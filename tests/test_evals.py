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


# ─────────────────────────────────────────────────────────────────────────────
#  Grounding judge: parsing and agreement statistics (no network)
# ─────────────────────────────────────────────────────────────────────────────
from evals import grounding_eval as ge  # noqa: E402


def test_parse_verdict_accepts_a_consistent_reply():
    raw = '{"claims": [{"claim": "Ana knows Kafka", "supported": true}], "verdict": "grounded"}'
    assert ge.parse_verdict(raw)["verdict"] == "grounded"


@pytest.mark.parametrize("raw", [
    '{"verdict": "mostly"}',
    '{"claims": []}',
    '{"claims": [{"claim": "x", "supported": false}], "verdict": "grounded"}',
    "not json",
])
def test_parse_verdict_rejects_malformed_or_inconsistent_replies(raw):
    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError
        ge.parse_verdict(raw)


def test_kappa_is_one_for_perfect_agreement_and_zero_for_a_constant_judge():
    G, U = "grounded", "ungrounded"
    assert ge.cohens_kappa([(G, G), (U, U), (G, G)]) == 1.0
    # Human finds 2 of 10 ungrounded; judge always says grounded: 80% agreement, kappa 0.
    pairs = [(G, G)] * 8 + [(U, G)] * 2
    assert ge.agreement_report(dict(enumerate(h for h, _ in pairs)),
                               dict(enumerate(j for _, j in pairs)))["agreement"] == 0.8
    assert ge.cohens_kappa(pairs) == 0.0


def test_kappa_textbook_example():
    # 50 items: both say G on 20, both U on 15, human G / judge U 5, human U / judge G 10.
    # observed = 0.7; chance = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = 0.4.
    G, U = "grounded", "ungrounded"
    pairs = [(G, G)] * 20 + [(U, U)] * 15 + [(G, U)] * 5 + [(U, G)] * 10
    assert ge.cohens_kappa(pairs) == pytest.approx(0.4)


def test_agreement_skips_ungraded_answers_and_lists_disagreements():
    human = {"g1": "grounded", "g2": "ungrounded", "g3": None}
    judged = {"g1": "grounded", "g2": "grounded", "g3": "ungrounded"}
    r = ge.agreement_report(human, judged)
    assert r["graded"] == 2
    assert r["disagreements"] == ["g2"]
    assert r["confusion"]["human_ungrounded/judge_grounded"] == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Review sheet: marks in REVIEW.md land in the JSON files, and nothing else does
# ─────────────────────────────────────────────────────────────────────────────
def test_a_grade_given_for_different_text_is_stale():
    frozen = [{"id": "g01", "answer": "a", "context": "c"}, {"id": "g02", "answer": "new", "context": "c"}]
    human = [{"id": i, "answer": "a", "context": "c", "human": "grounded"} for i in ("g01", "g02", "g03")]
    assert ge.stale_grades(human, frozen) == ["g02", "g03"]


def test_replanting_keeps_grades_for_unchanged_text_only(tmp_path, monkeypatch):
    from evals import plant_errors as pe

    for name in ("ANSWERS", "GRADES"):
        copy = tmp_path / getattr(pe, name).name
        copy.write_text(getattr(pe, name).read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr(pe, name, copy)
    doc = json.loads(pe.GRADES.read_text(encoding="utf-8"))
    for g in doc["grades"]:
        if g["id"] == "p02":
            g["answer"] += " (graded on an older text)"
    pe.GRADES.write_text(json.dumps(doc), encoding="utf-8")

    pe.plant()

    grades = {g["id"]: g["human"] for g in json.loads(pe.GRADES.read_text(encoding="utf-8"))["grades"]}
    assert grades["p01"] == "ungrounded"
    assert grades["p02"] is None


def test_labels_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{"queries":\n []}\n')
    crlf.write_bytes(b'{"queries":\r\n []}\r\n')
    assert retrieval_eval.load_labels(lf) == retrieval_eval.load_labels(crlf)


import shutil  # noqa: E402

from evals import review_sheet as rs  # noqa: E402


@pytest.fixture
def sheet(tmp_path, monkeypatch):
    # Copies of the real files, reset to the unreviewed state, so these tests
    # do not depend on how far the real review has got.
    for name in ("LABELS", "GRADES"):
        src = getattr(rs, name)
        shutil.copy(src, tmp_path / src.name)
        monkeypatch.setattr(rs, name, tmp_path / src.name)
    labels = json.loads(rs.LABELS.read_text(encoding="utf-8"))
    for q in labels["queries"]:
        q["status"] = "draft"
    rs.LABELS.write_text(json.dumps(labels), encoding="utf-8")
    grades = json.loads(rs.GRADES.read_text(encoding="utf-8"))
    for g in grades["grades"]:
        g["human"], g["note"] = None, ""
    rs.GRADES.write_text(json.dumps(grades), encoding="utf-8")
    monkeypatch.setattr(rs, "SHEET", tmp_path / "REVIEW.md")
    rs.make()
    return tmp_path


def _labels(): return {q["id"]: q for q in json.loads(rs.LABELS.read_text(encoding="utf-8"))["queries"]}
def _grades(): return {g["id"]: g for g in json.loads(rs.GRADES.read_text(encoding="utf-8"))["grades"]}


def _edit(fn):
    rs.SHEET.write_text(fn(rs.SHEET.read_text(encoding="utf-8")), encoding="utf-8")


def _fix_row(text, qid, fix):
    return "\n".join(l[:-3] + f" {fix} |" if l.startswith(f"| {qid} |") else l for l in text.split("\n"))


def test_untouched_sheet_changes_nothing(sheet):
    before_l, before_g = rs.LABELS.read_text(encoding="utf-8"), rs.GRADES.read_text(encoding="utf-8")
    assert rs.apply() == 0
    assert all(q["status"] == "draft" for q in _labels().values())
    assert all(g["human"] is None for g in _grades().values())
    assert json.loads(rs.LABELS.read_text(encoding="utf-8")) == json.loads(before_l)
    assert json.loads(rs.GRADES.read_text(encoding="utf-8")) == json.loads(before_g)


def test_fixes_and_grades_are_applied_and_labels_count_only_once_confirmed(sheet):
    _edit(lambda t: _fix_row(t, "k09", "+nathan_patel"))
    _edit(lambda t: _fix_row(t, "s20", "-olivia_baker"))
    _edit(lambda t: t.replace("**Grade:** \n\n---\n\n### g02", "**Grade:** u - says 9 years\n\n---\n\n### g02", 1))
    _edit(lambda t: t.replace("**Grade:** \n\n---\n\n### g03", "**Grade:** G\n\n---\n\n### g03", 1))
    assert rs.apply() == 0
    labels, grades = _labels(), _grades()
    assert labels["k09"]["relevant"] == ["resume_christopher_lee2.txt", "resume_nathan_patel.txt"]
    assert "resume_olivia_baker.txt" not in labels["s20"]["relevant"]
    assert labels["k09"]["status"] == "draft"  # not confirmed yet
    assert (grades["g01"]["human"], grades["g01"]["note"]) == ("ungrounded", "says 9 years")
    assert grades["g02"]["human"] == "grounded"
    assert sum(bool(g["human"]) for g in grades.values()) == 2

    rs.make()  # regenerating keeps the grades already applied
    _edit(lambda t: t.replace("Part 1:** no", "Part 1:** yes", 1))
    assert rs.apply() == 0
    assert all(q["status"] == "reviewed" for q in _labels().values())
    assert _grades()["g01"]["human"] == "ungrounded"


@pytest.mark.parametrize("fix", ["+nobody", "nathan_patel", "-christopher_lee2"])
def test_bad_marks_are_rejected_without_writing_anything(sheet, fix):
    before = rs.LABELS.read_text(encoding="utf-8")
    _edit(lambda t: _fix_row(t, "k09", fix))  # the last case would leave k09 with no answer
    assert rs.apply() == 1
    assert rs.LABELS.read_text(encoding="utf-8") == before


def test_bad_grade_is_rejected(sheet):
    _edit(lambda t: t.replace("**Grade:** \n\n---\n\n### g02", "**Grade:** maybe\n\n---\n\n### g02", 1))
    assert rs.apply() == 1


def test_thresholds_sit_just_below_the_run_they_came_from():
    report = {"labels_sha256": "abc", "results": {
        "pg/section/hybrid": {"metrics": {"all": {"recall@5": 0.9, "mrr": 0.01}}}}}
    t = retrieval_eval.thresholds_from(report, margin=0.02)
    assert t["minimums"]["pg/section/hybrid"] == {"recall@5": 0.88, "mrr": 0.0}


def test_gate_fails_when_any_configuration_loses_one_whole_query():
    """Run against the committed results and thresholds, as CI uses them."""
    report = json.loads(retrieval_eval.RESULTS_PATH.read_text(encoding="utf-8"))
    path = retrieval_eval.PROJECT_ROOT / "evals" / "thresholds.json"
    assert retrieval_eval.check(report, path) == []
    for name, result in report["results"].items():
        metrics = result["metrics"]["all"]
        for metric in ("recall@5", "mrr"):
            before = metrics[metric]
            metrics[metric] = before - 1 / metrics["n"]
            assert any(name in f and metric in f for f in retrieval_eval.check(report, path))
            metrics[metric] = before
