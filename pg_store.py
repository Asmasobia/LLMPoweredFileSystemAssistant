"""
Postgres + pgvector store for résumé chunks, with hybrid search.

CONCEPTS, IN THE ORDER THEY APPEAR BELOW
────────────────────────────────────────
pgvector   A Postgres extension that adds a `vector(n)` column type and distance
           operators. The cosine-distance operator returns 0 for the same
           direction and 2 for opposite. It turns "find chunks whose meaning is
           close to the query" into an ORDER BY on an ordinary table, so it can
           sit in the same SQL as WHERE clauses and joins.

Full-text  Postgres's built-in keyword search. `to_tsvector('english', text)`
search     reduces text to stemmed words ("developers" becomes "develop") with
           stop words removed; a `tsquery` is the same for the search terms; `@@`
           tests for a match and `ts_rank_cd` scores how well it matches. It
           catches exact terms (a skill name, "Kafka") that an embedding can
           blur into "something about data pipelines".

RRF        Reciprocal rank fusion: merges two ranked lists using only the ranks.
           Each chunk scores sum(1 / (k + rank)) over the lists it appears in,
           with k = 60 by convention. Ranks are used because the raw scores
           cannot be compared: a cosine distance and a ts_rank_cd value are on
           unrelated scales. A chunk near the top of both lists wins; a chunk
           that only one method found still gets credit.

Where the database comes from: DATABASE_URL if set (Docker Compose, CI),
otherwise an embedded Postgres started by `pgserver` in `.pgdata/`. Both run
Postgres 16 with pgvector 0.6.2.

    python pg_store.py index                    # chunk, embed and store the corpus
    python pg_store.py search "Kafka on AWS" --min-years 5
"""

import argparse
import os
from pathlib import Path

import numpy as np
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

import fs_tools
import resume_parsing

PROJECT_ROOT = Path(__file__).resolve().parent
PGDATA_DIR = PROJECT_ROOT / ".pgdata"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2, Chroma's default embedding model
RRF_K = 60
SEARCH_MODES = ("vector", "keyword", "hybrid")


# ─────────────────────────────────────────────────────────────────────────────
#  Connection
# ─────────────────────────────────────────────────────────────────────────────
_pgserver = None


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    global _pgserver
    if _pgserver is None:
        import pgserver  # only needed when there is no DATABASE_URL

        _pgserver = pgserver.get_server(str(PGDATA_DIR), cleanup_mode="stop")
    return _pgserver.get_uri()


def connect(url: str = None) -> psycopg.Connection:
    conn = psycopg.connect(url or database_url(), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets numpy arrays be passed as vector parameters
    return conn


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id               bigserial PRIMARY KEY,
    strategy         text NOT NULL,          -- 'section' or 'fixed'
    source           text NOT NULL,          -- e.g. resumes/resume_alex_kumar.txt
    chunk_index      int  NOT NULL,
    section          text NOT NULL,
    content          text NOT NULL,
    embedding        vector({EMBEDDING_DIM}) NOT NULL,
    -- Kept in sync by Postgres itself, so it can never disagree with content.
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    -- Per-résumé metadata, repeated on each chunk so filters need no join.
    candidate_name   text NOT NULL,
    experience_years int  NOT NULL,
    skills           text[] NOT NULL,
    education        text NOT NULL,
    UNIQUE (strategy, source, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_skills_idx ON chunks USING gin (skills);
-- No HNSW/IVFFlat index on `embedding`, on purpose. Those are approximate
-- nearest-neighbour indexes: faster, but they can miss true neighbours, and
-- combined with a WHERE filter they find the nearest N first and filter
-- afterwards, which can return fewer than N rows. At a few hundred chunks an
-- exact scan takes about a millisecond, and the evals should measure retrieval,
-- not index approximation. At scale you would add:
--   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def create_schema(conn) -> None:
    conn.execute(SCHEMA)


# ─────────────────────────────────────────────────────────────────────────────
#  Embeddings
# ─────────────────────────────────────────────────────────────────────────────
def default_embedder():
    """
    Chroma's default embedding function (all-MiniLM-L6-v2 via onnxruntime).

    The same model the Chroma baseline uses, so a Chroma-vs-Postgres comparison
    differs only in how retrieval is done, not in what the vectors are.
    Returns a function: list[str] -> np.ndarray of shape (n, 384), unit length.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    fn = DefaultEmbeddingFunction()
    return lambda texts: np.asarray(fn(list(texts)), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Indexing
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus(directory: str = "resumes") -> list:
    """Résumés only, read through the sandboxed tools like the rest of the app."""
    docs = []
    for entry in fs_tools.list_files(directory, ".txt"):
        if "path" not in entry or not resume_parsing.is_resume(entry["name"]):
            continue
        result = fs_tools.read_file(entry["path"])
        if result.get("success"):
            docs.append({"source": entry["path"], "name": entry["name"], "content": result["content"]})
    return docs


def index_corpus(conn, embed, strategy: str = "section", directory: str = "resumes") -> int:
    """Replace every chunk for `strategy` with a fresh index of the corpus."""
    rows = []
    for doc in load_corpus(directory):
        meta = resume_parsing.extract_metadata(doc["content"], doc["name"])
        for i, c in enumerate(resume_parsing.chunk(doc["content"], strategy)):
            rows.append((strategy, doc["source"], i, c["section"], c["text"],
                         meta["candidate_name"], meta["experience_years"],
                         meta["skills"], meta["education"]))
    vectors = embed([r[4] for r in rows]) if rows else []

    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (strategy, source, chunk_index, section, content,
                                       candidate_name, experience_years, skills, education,
                                       embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [row + (vec,) for row, vec in zip(rows, vectors)],
            )
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Search
# ─────────────────────────────────────────────────────────────────────────────
# One statement does the filtering, both searches and the fusion.
#
#   filtered  the metadata filters, applied BEFORE ranking, so both searches
#             only ever rank chunks that qualify. A NULL parameter means "no
#             filter", which keeps it one fixed statement.
#   vec       top `pool` chunks by cosine distance, numbered 1, 2, 3, ...
#   kw        top `pool` chunks by full-text rank, numbered the same way. The
#             query's words are OR-ed: plainto_tsquery would AND them, and a
#             six-word question then matches almost no single chunk.
#   fused     RRF over whichever of vec/kw the mode switched on.
SEARCH_SQL = """
WITH filtered AS (
    SELECT * FROM chunks
    WHERE strategy = %(strategy)s
      AND (%(min_years)s::int IS NULL OR experience_years >= %(min_years)s::int)
      AND (%(skills)s::text[] IS NULL OR skills @> %(skills)s::text[])
),
vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding OPERATOR(<=>) %(qvec)s, id) AS rank
    FROM filtered
    WHERE %(use_vector)s
    ORDER BY embedding OPERATOR(<=>) %(qvec)s, id
    LIMIT %(pool)s
),
q AS (
    SELECT replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')::tsquery AS tsq
),
kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q.tsq) DESC, id) AS rank
    FROM filtered, q
    WHERE %(use_keyword)s AND tsv @@ q.tsq
    ORDER BY ts_rank_cd(tsv, q.tsq) DESC, id
    LIMIT %(pool)s
),
fused AS (
    SELECT id, sum(1.0 / (%(rrf_k)s + rank)) AS score
    FROM (SELECT id, rank FROM vec UNION ALL SELECT id, rank FROM kw) ranked
    GROUP BY id
)
SELECT c.source, c.candidate_name, c.section, c.content,
       f.score::float AS score, v.rank AS vector_rank, k.rank AS keyword_rank,
       c.experience_years, c.skills
FROM fused f
JOIN chunks c USING (id)
LEFT JOIN vec v USING (id)
LEFT JOIN kw  k USING (id)
ORDER BY f.score DESC, c.id
LIMIT %(limit)s
"""


def search(conn, query: str, embed=None, mode: str = "hybrid", strategy: str = "section",
           k: int = 10, min_years: int = None, skills: list = None, pool: int = 50) -> list:
    """
    Top-k chunks for `query`. `mode` is vector, keyword or hybrid (RRF of both).
    `pool` is how many candidates each method contributes before fusion.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unknown mode {mode!r}; use one of {SEARCH_MODES}.")
    use_vector = mode in ("vector", "hybrid")
    qvec = embed([query])[0] if use_vector else np.zeros(EMBEDDING_DIM, dtype=np.float32)
    params = {
        "strategy": strategy, "min_years": min_years, "skills": skills or None,
        "qvec": qvec, "query": query, "pool": pool, "limit": k, "rrf_k": RRF_K,
        "use_vector": use_vector, "use_keyword": mode in ("keyword", "hybrid"),
    }
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(SEARCH_SQL, params)
        return cur.fetchall()


def rank_files(chunks: list) -> list:
    """
    Distinct résumé files in the order of their best chunk.

    Evals are judged per file ("which résumés are relevant?"), but the stores
    rank chunks, and one strong résumé can fill the top 5 with its own sections.
    """
    files = []
    for c in chunks:
        if c["source"] not in files:
            files.append(c["source"])
    return files


def search_files(conn, query: str, k: int = 5, pool: int = 50, **kwargs) -> list:
    """Top-k distinct résumé files for `query` (see rank_files)."""
    return rank_files(search(conn, query, k=pool, pool=pool, **kwargs))[:k]


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description="Postgres + pgvector résumé store.")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="chunk, embed and store the résumés")
    ix.add_argument("--strategy", choices=resume_parsing.CHUNKING_STRATEGIES + ("all",), default="all")
    s = sub.add_parser("search", help="hybrid search with optional filters")
    s.add_argument("query")
    s.add_argument("--mode", choices=SEARCH_MODES, default="hybrid")
    s.add_argument("--strategy", choices=resume_parsing.CHUNKING_STRATEGIES, default="section")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--min-years", type=int)
    s.add_argument("--skill", action="append", dest="skills", help="repeatable; all must match")
    args = p.parse_args(argv)

    conn = connect()
    create_schema(conn)
    embed = default_embedder()
    if args.cmd == "index":
        strategies = resume_parsing.CHUNKING_STRATEGIES if args.strategy == "all" else (args.strategy,)
        for strategy in strategies:
            print(f"{strategy}: {index_corpus(conn, embed, strategy)} chunks")
        return
    rows = search(conn, args.query, embed, mode=args.mode, strategy=args.strategy,
                  k=args.k, min_years=args.min_years, skills=args.skills)
    for i, r in enumerate(rows, 1):
        print(f"[{i}] {r['score']:.4f}  vec#{r['vector_rank'] or '-':<3} kw#{r['keyword_rank'] or '-':<3} "
              f"{r['candidate_name']} ({r['experience_years']} yrs) / {r['section']}  {r['source']}")


if __name__ == "__main__":
    main()
