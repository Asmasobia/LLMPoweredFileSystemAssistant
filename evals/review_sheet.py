"""
Turn labels.json and human_grades.json into one readable markdown sheet, and
read the reviewer's marks back.

    python evals/review_sheet.py make     # writes evals/REVIEW.md
    python evals/review_sheet.py apply    # copies the marks into the two JSON files

In REVIEW.md, the reviewer only types in two places:
  * labels:  the last table column. Blank or "ok" = correct as listed.
             "+name" adds a résumé, "-name" removes one, e.g. "+hannah_murphy -jane_smith".
  * answers: the "Grade:" line, "g" (grounded) or "u" (ungrounded). Text after
             the grade, separated by " - ", becomes the note.
"""

import json
import re
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent
LABELS = EVALS / "labels.json"
GRADES = EVALS / "human_grades.json"
SHEET = EVALS / "REVIEW.md"
REQUIRED_GRADES = 20


def short(f):
    return f.removeprefix("resume_").removesuffix(".txt")


def full(name):
    return f"resume_{name}.txt"


def make():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["queries"]
    grades = json.loads(GRADES.read_text(encoding="utf-8"))["grades"]
    # Judgement calls first, then the rest in file order.
    labels = sorted(labels, key=lambda q: not q["note"].startswith("CHECK"))

    out = [
        "# Eval review sheet",
        "",
        "Two parts. Type only in the **your fix** column and on the **Grade:** lines,",
        "then tell Claude, who runs `python evals/review_sheet.py apply`.",
        "",
        f"## Part 1: retrieval labels ({len(labels)} queries)",
        "",
        "Ask: *if a recruiter typed this, which résumés would they want to see?*",
        "Leave **your fix** blank if the list is right. Otherwise `+name` to add, `-name` to remove",
        "(names as in `resumes/`, without `resume_` and `.txt`). The ones marked CHECK are judgement calls.",
        "Résumé names: " + ", ".join(sorted(short(p.name) for p in (EVALS.parent / "resumes").glob("resume_*.txt"))),
        "",
        "**I have checked every row in Part 1:** no   ← change to `yes` when done (until then no label counts as reviewed)",
        "",
        "| id | query | filter | relevant | note | your fix |",
        "|---|---|---|---|---|---|",
    ]
    for q in labels:
        flt = f"≥{q['min_years']} yrs" if "min_years" in q else ""
        out.append(f"| {q['id']} | {q['query']} | {flt} | {', '.join(short(f) for f in q['relevant'])} "
                   f"| {q['note'].replace('|', '/')} |  |")

    out += [
        "",
        f"## Part 2: answer grading (grade at least {REQUIRED_GRADES} of {len(grades)})",
        "",
        "For each answer: is **every claim it makes about a candidate** stated in the excerpts below it?",
        "Judge against those excerpts only, not the full résumés. \"The résumés don't say\" counts as grounded.",
        "Write `g` (grounded) or `u` (ungrounded) after **Grade:**; optionally ` - reason`.",
        "",
    ]
    for g in grades:
        out += [
            "---",
            "",
            f"### {g['id']}: {g['question']}",
            "",
            "**Answer:**",
            "",
            "> " + g["answer"].strip().replace("\n", "\n> "),
            "",
            "<details><summary>Excerpts the model was given</summary>",
            "",
            "```",
            g["context"],
            "```",
            "",
            "</details>",
            "",
            f"**Grade:** {grade_mark(g)}",
            "",
        ]
    SHEET.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {SHEET.name}")


def grade_mark(g):
    if not g.get("human"):
        return ""
    return g["human"][0] + (f" - {g['note']}" if g.get("note") else "")


def apply():
    text = SHEET.read_text(encoding="utf-8")
    labels_doc = json.loads(LABELS.read_text(encoding="utf-8"))
    grades_doc = json.loads(GRADES.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in labels_doc["queries"]}
    known = {short(p.name) for p in (EVALS.parent / "resumes").glob("resume_*.txt")}
    problems = []
    confirmed = re.search(r"^\*\*I have checked every row in Part 1:\*\*\s*yes\b", text, re.M | re.I)

    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in by_id:
            continue
        q, fix = by_id[cells[0]], cells[5]
        relevant = [short(f) for f in q["relevant"]]
        for token in fix.split():
            if token.lower() == "ok":
                continue
            op, name = token[0], token[1:]
            if op not in "+-" or name not in known:
                problems.append(f"{q['id']}: can't read {token!r} (use +name or -name)")
            elif op == "+" and name not in relevant:
                relevant.append(name)
            elif op == "-" and name in relevant:
                relevant.remove(name)
        if not relevant:
            problems.append(f"{q['id']}: no relevant résumés left")
        q["relevant"] = [full(n) for n in relevant]
        if confirmed:
            q["status"] = "reviewed"

    grades = {g["id"]: g for g in grades_doc["grades"]}
    for gid, mark in re.findall(r"^### (g\d+):.*?^\*\*Grade:\*\*[ \t]*([^\n]*)$", text, re.M | re.S):
        mark = mark.strip()
        if not mark:
            continue
        verdict, _, note = mark.partition(" - ")
        verdict = {"g": "grounded", "u": "ungrounded"}.get(verdict.strip().lower()[:1])
        if verdict is None:
            problems.append(f"{gid}: grade must start with g or u, got {mark!r}")
            continue
        grades[gid]["human"], grades[gid]["note"] = verdict, note.strip()

    reviewed = sum(q["status"] == "reviewed" for q in labels_doc["queries"])
    graded = sum(bool(g["human"]) for g in grades_doc["grades"])
    if problems:
        print("Not applied; fix these in REVIEW.md first:\n  " + "\n  ".join(problems))
        return 1
    LABELS.write_text(json.dumps(labels_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    GRADES.write_text(json.dumps(grades_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not confirmed:
        print("Part 1 not confirmed, so fixes were copied but labels stay drafts.")
    print(f"labels reviewed: {reviewed}/{len(labels_doc['queries'])}; answers graded: {graded}/{len(grades)}"
          + ("" if graded >= REQUIRED_GRADES else f" (need {REQUIRED_GRADES})"))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"make": make, "apply": apply}.get(cmd, lambda: print(__doc__) or 2)() or 0)
