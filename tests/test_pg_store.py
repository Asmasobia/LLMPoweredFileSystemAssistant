"""
Tests for the Postgres store (`pg_store.py`): schema, indexing, filters, and
that hybrid search really is RRF over the vector and keyword rankings.

They need a real Postgres with pgvector: DATABASE_URL if set (CI, Docker
Compose), otherwise the embedded pgserver in `.pgdata/`. If neither is
available the whole module is skipped, and `-rs` prints why.

Each test gets its own schema, created and dropped around it, so the tests never
see the development index or each other's rows.

The embedder is a fake: a bag of words hashed into 384 dimensions. Texts that
share words have similar vectors, it needs no model download, and it is
deterministic, so the rank of every chunk can be checked exactly.
"""

import re
import uuid
import zlib

import pytest

np = pytest.importorskip("numpy")

psycopg = pytest.importorskip("psycopg")
pg_store = pytest.importorskip("pg_store")


def fake_embed(texts):
    out = np.zeros((len(texts), pg_store.EMBEDDING_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9+#]+", text.lower()):
            out[row, zlib.crc32(word.encode()) % pg_store.EMBEDDING_DIM] += 1.0
        norm = np.linalg.norm(out[row])
        out[row] = out[row] / norm if norm else 1.0 / np.sqrt(pg_store.EMBEDDING_DIM)
    return out


RESUMES = {
    "resume_ana_kafka.txt": ("ANA KAFKA", 8, "Streaming engineer. Built Kafka pipelines on AWS.", "Python, Kafka, AWS"),
    "resume_ben_react.txt": ("BEN REACT", 3, "Frontend developer building React dashboards.", "JavaScript, React"),
    "resume_cy_kafka.txt": ("CY KAFKA", 2, "Junior engineer. Wrote Kafka consumers in Python.", "Python, Kafka"),
    "resume_dee_k8s.txt": ("DEE K8S", 6, "Platform engineer running Kubernetes and Terraform.", "Go, Kubernetes, Terraform"),
}


def resume_text(name, years, experience, skills):
    return (
        f"{name}\nEngineer\n\nSUMMARY\nEngineer with {years} years of experience.\n\n"
        f"EXPERIENCE\nEngineer | Co | 2020 - Present\n- {experience}\n\n"
        f"SKILLS\nLanguages: {skills}\n\nEDUCATION\nB.S. | Uni | 2015\n"
    )


@pytest.fixture(scope="module")
def db_url():
    try:
        url = pg_store.database_url()
        with psycopg.connect(url, connect_timeout=5) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as e:  # no pgserver, no DATABASE_URL, or it would not start
        pytest.skip(f"no Postgres with pgvector available: {e}")
    return url


@pytest.fixture
def conn(db_url):
    schema = f"test_{uuid.uuid4().hex[:12]}"
    c = pg_store.connect(db_url)
    c.execute(f"CREATE SCHEMA {schema}")
    # `public` stays on the path so the vector type, which lives there, resolves.
    c.execute(f"SET search_path TO {schema}, public")
    pg_store.create_schema(c)
    yield c
    c.execute(f"DROP SCHEMA {schema} CASCADE")
    c.close()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_ROOT", str(tmp_path))
    d = tmp_path / "resumes"
    d.mkdir()
    for filename, fields in RESUMES.items():
        (d / filename).write_text(resume_text(*fields), encoding="utf-8")
    # Not a résumé; must not be indexed.
    (d / "summary.txt").write_text("Summary for ANA KAFKA: Kafka Kafka Kafka.", encoding="utf-8")
    return d


@pytest.fixture
def indexed(conn, corpus):
    pg_store.index_corpus(conn, fake_embed, "section")
    pg_store.index_corpus(conn, fake_embed, "fixed")
    return conn


def sources(rows):
    return [r["source"].split("/")[-1] for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
#  Indexing
# ─────────────────────────────────────────────────────────────────────────────
def test_index_stores_each_resume_once_per_strategy_and_skips_summary(indexed):
    rows = indexed.execute(
        "SELECT strategy, count(DISTINCT source), bool_or(source LIKE '%%summary%%') "
        "FROM chunks GROUP BY strategy ORDER BY strategy"
    ).fetchall()
    assert rows == [("fixed", 4, False), ("section", 4, False)]


def test_reindexing_replaces_rather_than_duplicates(indexed):
    before = indexed.execute("SELECT count(*) FROM chunks").fetchone()[0]
    pg_store.index_corpus(indexed, fake_embed, "section")
    assert indexed.execute("SELECT count(*) FROM chunks").fetchone()[0] == before


def test_metadata_and_full_text_columns_are_populated(indexed):
    years, skills, tsv = indexed.execute(
        "SELECT experience_years, skills, tsv::text FROM chunks "
        "WHERE strategy = 'section' AND candidate_name = 'ANA KAFKA' AND section = 'EXPERIENCE'"
    ).fetchone()
    assert years == 8
    assert {"Python", "Kafka", "AWS"} <= set(skills)
    assert "'kafka'" in tsv and "'pipelin'" in tsv  # stemmed by the english config


# ─────────────────────────────────────────────────────────────────────────────
#  Search modes
# ─────────────────────────────────────────────────────────────────────────────
def test_keyword_mode_only_returns_chunks_containing_a_query_term(indexed):
    rows = pg_store.search(indexed, "Terraform", fake_embed, mode="keyword")
    assert rows and all("terraform" in r["content"].lower() for r in rows)
    assert all(r["vector_rank"] is None for r in rows)


def test_vector_mode_ranks_every_chunk(indexed):
    rows = pg_store.search(indexed, "React dashboards", fake_embed, mode="vector", k=100)
    total = indexed.execute("SELECT count(*) FROM chunks WHERE strategy = 'section'").fetchone()[0]
    assert len(rows) == total
    assert all(r["keyword_rank"] is None for r in rows)
    assert sources(rows)[0] == "resume_ben_react.txt"


def test_hybrid_score_is_reciprocal_rank_fusion_of_both_rankings(indexed):
    rows = pg_store.search(indexed, "Kafka pipelines", fake_embed, mode="hybrid", k=100)
    assert any(r["vector_rank"] and r["keyword_rank"] for r in rows)
    for r in rows:
        expected = sum(1.0 / (pg_store.RRF_K + rank)
                       for rank in (r["vector_rank"], r["keyword_rank"]) if rank)
        assert r["score"] == pytest.approx(expected)
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)


def test_keyword_terms_are_ored_not_anded(indexed):
    # No chunk contains both words. With AND semantics this would return nothing.
    rows = pg_store.search(indexed, "Terraform dashboards", fake_embed, mode="keyword")
    assert {"resume_dee_k8s.txt", "resume_ben_react.txt"} <= set(sources(rows))


def test_query_of_only_stop_words_still_returns_vector_results(indexed):
    rows = pg_store.search(indexed, "the and of", fake_embed, mode="hybrid")
    assert rows and all(r["keyword_rank"] is None for r in rows)


def test_unknown_mode_is_an_error(indexed):
    with pytest.raises(ValueError, match="Unknown mode"):
        pg_store.search(indexed, "x", fake_embed, mode="bm25")


# ─────────────────────────────────────────────────────────────────────────────
#  Filters, in the same statement
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", pg_store.SEARCH_MODES)
def test_min_years_filter_applies_in_every_mode(indexed, mode):
    rows = pg_store.search(indexed, "Kafka", fake_embed, mode=mode, k=100, min_years=5)
    assert rows
    assert all(r["experience_years"] >= 5 for r in rows)
    assert "resume_cy_kafka.txt" not in sources(rows)  # Kafka, but 2 years


def test_skill_filter_requires_every_listed_skill(indexed):
    rows = pg_store.search(indexed, "engineer", fake_embed, k=100, skills=["Python", "Kafka"])
    assert set(sources(rows)) == {"resume_ana_kafka.txt", "resume_cy_kafka.txt"}


def test_filter_that_matches_nothing_returns_nothing(indexed):
    assert pg_store.search(indexed, "Kafka", fake_embed, min_years=50) == []


def test_strategies_are_searched_separately(indexed):
    rows = pg_store.search(indexed, "Kafka", fake_embed, strategy="fixed", k=100)
    assert rows and {r["section"] for r in rows} == {"CHUNK"}


# ─────────────────────────────────────────────────────────────────────────────
#  File-level ranking
# ─────────────────────────────────────────────────────────────────────────────
def test_search_files_returns_distinct_files_in_best_chunk_order(indexed):
    files = pg_store.search_files(indexed, "Kafka pipelines AWS", embed=fake_embed, k=3)
    assert len(files) == len(set(files)) == 3
    assert files[0].endswith("resume_ana_kafka.txt")
