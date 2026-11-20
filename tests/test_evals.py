"""Tests for evals/metrics.py and the eval's scoring and regression gate."""

import json

import pytest

from evals.metrics import mean, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d", "e", "f", "g"]


def test_recall_counts_relevant_found_within_k():
    assert recall_at_k(RANKED, {"a"}, 5) == 1.0
    assert recall_at_k(RANKED, {"a", "f"}, 5) == 0.5   # f is 6th
    assert recall_at_k(RANKED, {"a", "f"}, 6) == 1.0
    assert recall_at_k(RANKED, {"z"}, 5) == 0.0


def test_recall_ignores_order_within_k():
    assert recall_at_k(RANKED, {"e"}, 5) == recall_at_k(RANKED, {"a"}, 5)


def test_recall_needs_at_least_one_relevant():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)


def test_reciprocal_rank_uses_the_first_relevant_hit():
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"c", "b"}) == 0.5
    assert reciprocal_rank(RANKED, {"e"}) == 0.2
    assert reciprocal_rank(RANKED, {"z"}) == 0.0


def test_reciprocal_rank_respects_the_cutoff():
    assert reciprocal_rank(RANKED, {"g"}, cutoff=5) == 0.0
    assert reciprocal_rank(RANKED, {"g"}, cutoff=7) == pytest.approx(1 / 7)


def test_mean_of_nothing_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 0.5]) == 0.75


# The runner imports pg_store, which needs psycopg; skip these where it is absent.
retrieval_eval = pytest.importorskip("evals.retrieval_eval", exc_type=ImportError)


def test_score_averages_per_query_and_by_kind():
    queries = [
        {"id": "q1", "kind": "keyword", "relevant": ["a"]},
        {"id": "q2", "kind": "semantic", "relevant": ["x", "c"]},
    ]
    per_query = {"q1": ["b", "a"], "q2": ["c", "b", "a"]}
    s = retrieval_eval.score(per_query, queries)
    assert s["keyword"] == {"n": 1, "recall@5": 1.0, "mrr": 0.5}
    assert s["semantic"] == {"n": 1, "recall@5": 0.5, "mrr": 1.0}
    assert s["all"] == {"n": 2, "recall@5": 0.75, "mrr": 0.75}


def test_check_reports_each_metric_below_its_minimum(tmp_path):
    report = {"results": {"pg/section/hybrid": {"metrics": {"all": {"recall@5": 0.8, "mrr": 0.6}}}}}
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"minimums": {"pg/section/hybrid": {"recall@5": 0.75, "mrr": 0.65}}}))
    failures = retrieval_eval.check(report, path)
    assert len(failures) == 1 and "mrr" in failures[0]


def test_labels_file_is_well_formed():
    """Every label names real résumé files, and filtered labels pass their own filter."""
    from pathlib import Path

    import resume_parsing

    queries, _ = retrieval_eval.load_labels()
    resumes = Path(retrieval_eval.PROJECT_ROOT, "resumes")
    years = {p.name: resume_parsing.extract_experience_years(p.read_text(encoding="utf-8"))
             for p in resumes.glob("resume_*.txt")}
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))
    for q in queries:
        assert q["kind"] in ("keyword", "semantic", "filtered"), q["id"]
        assert q["status"] in ("draft", "reviewed"), q["id"]
        assert q["relevant"], q["id"]
        for f in q["relevant"]:
            assert f in years, (q["id"], f)
            if "min_years" in q:
                assert years[f] >= q["min_years"], (q["id"], f)
