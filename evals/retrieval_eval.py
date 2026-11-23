"""
Retrieval eval: recall@5 and MRR for every store x chunking x retrieval mode.

    python evals/retrieval_eval.py                       # print the table, write results JSON
    python evals/retrieval_eval.py --check evals/thresholds.json   # also fail on regression (CI)

What is compared, and why each axis exists:

  store      chroma = the original setup (ChromaDB, HNSW index, vector only) as
             the baseline; pg = Postgres + pgvector (exact search).
  chunking   section = one chunk per résumé section; fixed = 500-character
             windows with 100 characters of overlap.
  retrieval  vector = embedding similarity only; keyword = Postgres full-text
             only (shown for reference); hybrid = both, merged with RRF.

Every configuration indexes the same chunks with the same vectors (computed
once per chunking strategy and handed to both stores), so a difference between
two rows comes from the axis that differs and nothing else.

Results are judged per résumé file, not per chunk: the stores rank chunks, and
the ranked list of files is each file in the order of its best chunk.

The labels are in evals/labels.json. While any label is still a draft, the run
says so loudly and the results file records it: unchecked, model-written labels
make the numbers meaningless.
"""

import argparse
import datetime
import hashlib
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pg_store  # noqa: E402
import resume_parsing  # noqa: E402
from evals.metrics import mean, recall_at_k, reciprocal_rank  # noqa: E402

LABELS_PATH = PROJECT_ROOT / "evals" / "labels.json"
RESULTS_PATH = PROJECT_ROOT / "evals" / "results" / "retrieval.json"
K = 5
MRR_CUTOFF = 10

CONFIGS = [
    ("chroma", "section", "vector"),
    ("chroma", "fixed", "vector"),
    ("pg", "section", "vector"),
    ("pg", "section", "keyword"),
    ("pg", "section", "hybrid"),
    ("pg", "fixed", "vector"),
    ("pg", "fixed", "keyword"),
    ("pg", "fixed", "hybrid"),
]


def config_name(store, strategy, mode):
    return f"{store}/{strategy}/{mode}"


# ─────────────────────────────────────────────────────────────────────────────
#  Building the index once per chunking strategy
# ─────────────────────────────────────────────────────────────────────────────
def build_chunks(docs, strategy, embed):
    rows = []
    for doc in docs:
        meta = resume_parsing.extract_metadata(doc["content"], doc["name"])
        for i, c in enumerate(resume_parsing.chunk(doc["content"], strategy)):
            rows.append({"source": doc["source"], "chunk_index": i, "section": c["section"],
                         "text": c["text"], "meta": meta})
    vectors = embed([r["text"] for r in rows])
    return rows, vectors


class ChromaIndex:
    """The baseline: an in-memory Chroma collection, cosine HNSW, as resume_rag.py uses."""

    def __init__(self, strategy, rows, vectors):
        import chromadb

        client = chromadb.EphemeralClient()
        self.collection = client.create_collection(
            f"eval_{strategy}_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self.collection.add(
            ids=[f"{r['source']}#{r['chunk_index']}" for r in rows],
            embeddings=[v.tolist() for v in vectors],
            documents=[r["text"] for r in rows],
            metadatas=[{"source": r["source"], "experience_years": r["meta"]["experience_years"]}
                       for r in rows],
        )
        self.size = len(rows)

    def rank_files(self, qvec, min_years=None):
        kwargs = {"query_embeddings": [qvec.tolist()], "n_results": self.size}
        if min_years is not None:
            kwargs["where"] = {"experience_years": {"$gte": min_years}}
        res = self.collection.query(**kwargs)
        return pg_store.rank_files([{"source": m["source"]} for m in res["metadatas"][0]])


class PgIndex:
    """Postgres in a throwaway schema, so the eval never touches the dev index."""

    def __init__(self, conn):
        self.conn = conn
        self.schema = f"eval_{uuid.uuid4().hex[:12]}"
        conn.execute(f"CREATE SCHEMA {self.schema}")
        conn.execute(f"SET search_path TO {self.schema}, public")
        pg_store.create_schema(conn)

    def load(self, strategy, rows, vectors):
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (strategy, source, chunk_index, section, content,
                                       candidate_name, experience_years, skills, education, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(strategy, r["source"], r["chunk_index"], r["section"], r["text"],
                  r["meta"]["candidate_name"], r["meta"]["experience_years"],
                  r["meta"]["skills"], r["meta"]["education"], v)
                 for r, v in zip(rows, vectors)],
            )

    def rank_files(self, query, qvec, strategy, mode, min_years=None):
        chunks = pg_store.search(self.conn, query, embed=lambda _: [qvec], mode=mode,
                                 strategy=strategy, k=200, pool=200, min_years=min_years)
        return pg_store.rank_files(chunks)

    def drop(self):
        self.conn.execute(f"DROP SCHEMA {self.schema} CASCADE")


# ─────────────────────────────────────────────────────────────────────────────
#  Running and scoring
# ─────────────────────────────────────────────────────────────────────────────
def load_labels(path=LABELS_PATH):
    # Normalised so the hash is the same for CRLF and LF checkouts.
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    queries = json.loads(raw)["queries"]
    return queries, hashlib.sha256(raw).hexdigest()[:12]


def score(per_query, queries):
    """Aggregate per-query rankings into recall@K and MRR, overall and by kind."""
    out = {}
    groups = {"all": queries}
    for q in queries:
        groups.setdefault(q["kind"], []).append(q)
    for group, qs in groups.items():
        out[group] = {
            "n": len(qs),
            f"recall@{K}": mean(recall_at_k(per_query[q["id"]], q["relevant"], K) for q in qs),
            "mrr": mean(reciprocal_rank(per_query[q["id"]], q["relevant"], MRR_CUTOFF) for q in qs),
        }
    return out


def run(labels_path=LABELS_PATH):
    queries, labels_hash = load_labels(labels_path)
    embed = pg_store.default_embedder()
    docs = pg_store.load_corpus("resumes")
    qvecs = dict(zip((q["id"] for q in queries), embed([q["query"] for q in queries])))

    conn = pg_store.connect()
    pg = PgIndex(conn)
    chroma, chunk_counts = {}, {}
    try:
        for strategy in resume_parsing.CHUNKING_STRATEGIES:
            rows, vectors = build_chunks(docs, strategy, embed)
            chunk_counts[strategy] = len(rows)
            chroma[strategy] = ChromaIndex(strategy, rows, vectors)
            pg.load(strategy, rows, vectors)

        results = {}
        for store, strategy, mode in CONFIGS:
            per_query = {}
            for q in queries:
                ranked = (chroma[strategy].rank_files(qvecs[q["id"]], q.get("min_years"))
                          if store == "chroma" else
                          pg.rank_files(q["query"], qvecs[q["id"]], strategy, mode, q.get("min_years")))
                per_query[q["id"]] = [Path(s).name for s in ranked[:MRR_CUTOFF]]
            results[config_name(store, strategy, mode)] = {
                "metrics": score(per_query, queries), "rankings": per_query,
            }
    finally:
        pg.drop()
        conn.close()

    drafts = sum(1 for q in queries if q.get("status") != "reviewed")
    return {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": "python evals/retrieval_eval.py",
        "labels_sha256": labels_hash,
        "draft_labels": drafts,
        "counts": drafts == 0,
        "dataset": {"resumes": len(docs), "queries": len(queries), "chunks": chunk_counts,
                    "note": "synthetic résumés"},
        "embedding_model": "all-MiniLM-L6-v2 (Chroma default, ONNX)",
        "k": K, "mrr_cutoff": MRR_CUTOFF,
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Reporting and the CI gate
# ─────────────────────────────────────────────────────────────────────────────
def table(report) -> str:
    kinds = ["all", "keyword", "semantic", "filtered"]
    head = "| config | " + " | ".join(f"{g} R@{K} | {g} MRR" for g in kinds) + " |"
    lines = [head, "|" + "---|" * (1 + 2 * len(kinds))]
    for name, r in report["results"].items():
        m = r["metrics"]
        cells = []
        for g in kinds:
            cells += [f"{m[g][f'recall@{K}']:.3f}", f"{m[g]['mrr']:.3f}"] if g in m else ["-", "-"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check(report, thresholds_path) -> list:
    """
    Regression gate. thresholds.json maps config -> {metric: minimum}, for the
    "all" group. Returns a list of failures (empty = pass).
    """
    thresholds_path = Path(thresholds_path)
    if not thresholds_path.exists():
        return [f"{thresholds_path} does not exist yet. It is written from a run on reviewed labels."]
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    failures = []
    for name, minimums in thresholds["minimums"].items():
        got = report["results"][name]["metrics"]["all"]
        for metric, minimum in minimums.items():
            if got[metric] < minimum:
                failures.append(f"{name} {metric} = {got[metric]:.3f}, below {minimum:.3f}")
    return failures


# Below 1/53 (0.0189), what one query that loses everything costs the average,
# so losing a whole query fails the gate.
THRESHOLD_MARGIN = 0.01


def thresholds_from(report, margin=THRESHOLD_MARGIN) -> dict:
    """
    Minimums for the CI gate: each configuration's current recall@K and MRR,
    less a small margin. The retrieval is deterministic, so the margin is not
    for noise between runs; it lets an unrelated change that shuffles a tied
    rank or two pass, while a real regression still fails.
    """
    return {
        "about": f"Written by retrieval_eval.py --write-thresholds from labels {report['labels_sha256']}: "
                 f"each metric from that run minus {margin}. Regenerate when the labels change.",
        "minimums": {
            name: {m: round(max(0.0, r["metrics"]["all"][m] - margin), 3) for m in (f"recall@{K}", "mrr")}
            for name, r in report["results"].items()
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Retrieval eval: recall@5 and MRR.")
    p.add_argument("--check", metavar="THRESHOLDS", help="fail (exit 1) if any metric drops below its minimum")
    p.add_argument("--out", default=str(RESULTS_PATH))
    p.add_argument("--write-thresholds", metavar="PATH",
                   help="write CI minimums from this run (refused while labels are drafts)")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    report = run()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    d = report["dataset"]
    print(f"{d['resumes']} synthetic résumés, {d['queries']} queries, chunks: {d['chunks']}")
    if report["draft_labels"]:
        print(f"\n*** {report['draft_labels']} of {d['queries']} labels are unreviewed drafts. "
              f"These numbers do not count. ***\n")
    print(table(report))
    Path(args.out).with_suffix(".md").write_text(table(report) + "\n", encoding="utf-8")

    if args.write_thresholds:
        if report["draft_labels"]:
            print("Not writing thresholds: labels are still drafts.")
            return 1
        Path(args.write_thresholds).write_text(
            json.dumps(thresholds_from(report), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.write_thresholds}")

    if args.check:
        failures = check(report, args.check)
        if report["draft_labels"]:
            failures.append(f"{report['draft_labels']} labels are still drafts")
        for f in failures:
            print("REGRESSION:", f)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
