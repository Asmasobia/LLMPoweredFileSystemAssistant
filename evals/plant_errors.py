"""
Planted errors for the grounding judge.

All 24 real answers were graded grounded by hand, so on their own they cannot
show whether the judge catches an ungrounded answer: a judge that always said
"grounded" would agree 100%. This script makes 8 copies of real answers, each
with ONE claim added or changed that the excerpts do not support, keeping the
excerpts exactly as they were.

    python evals/plant_errors.py    # adds p01..p08 to answers.json and human_grades.json

The planted claims are listed here so anyone can see exactly what was changed.
They are still graded by hand like the rest: the human check confirms each
plant really is unsupported by its excerpts.
"""

import json
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent
ANSWERS = EVALS / "results" / "answers.json"
GRADES = EVALS / "human_grades.json"

# (new id, source id, text to find in the source answer, replacement, the false claim)
PLANTS = [
    ("p01", "g02", "**6 years of experience**", "**8 years of experience**",
     "8 years (the excerpt says 6)"),
    ("p02", "g04", "CompTIA Security+ and OSCP", "CompTIA Security+, CISSP and OSCP",
     "holds CISSP"),
    ("p03", "g03", "Brian Jackson and Chris Lee have both built iOS apps with SwiftUI.",
     "Brian Jackson and Chris Lee have both built iOS apps with SwiftUI, and Brian Jackson also built Android apps in Kotlin.",
     "Brian Jackson built Android apps in Kotlin"),
    ("p04", "g14", "** (resume_rachel_moore.txt).",
     "**, who built it while working at OpenAI (resume_rachel_moore.txt).",
     "Rachel Moore worked at OpenAI"),
    ("p05", "g16", "from Stanford University (2020)", "from MIT (2020)",
     "PhD from MIT (the excerpt says Stanford)"),
    ("p06", "g17", " has worked on autonomous vehicles (resume_victoria_ross.txt).",
     " has worked on autonomous vehicles, including two years on Tesla's Autopilot team (resume_victoria_ross.txt).",
     "two years on Tesla's Autopilot team"),
    ("p07", "g24", "using TensorRT and ONNX.", "using TensorRT and ONNX, cutting inference latency by 60%.",
     "cut inference latency by 60%"),
    ("p08", "g07", "“Migrated on‑premise applications to AWS”",
     "“Migrated on‑premise applications to AWS”, moving 200 servers in under three months",
     "200 servers in under three months"),
]


def plant():
    answers_doc = json.loads(ANSWERS.read_text(encoding="utf-8"))
    grades_doc = json.loads(GRADES.read_text(encoding="utf-8"))
    by_id = {a["id"]: a for a in answers_doc["answers"]}
    answers_doc["answers"] = [a for a in answers_doc["answers"] if not a.get("planted")]
    # A grade stays valid only for the exact text it was given for, so a rerun
    # keeps it when the planted answer and its excerpts come out unchanged.
    old_grades = {g["id"]: g for g in grades_doc["grades"] if g["id"].startswith("p")}
    grades_doc["grades"] = [g for g in grades_doc["grades"] if not g["id"].startswith("p")]

    for new_id, src_id, find, replace, claim in PLANTS:
        src = by_id[src_id]
        if find not in src["answer"]:
            raise SystemExit(f"{new_id}: {find!r} not found in {src_id}'s answer; the answers changed.")
        answer = src["answer"].replace(find, replace, 1)
        answers_doc["answers"].append({
            "id": new_id, "question": src["question"], "answerable": src["answerable"],
            "context": src["context"], "answer": answer,
            "planted": True, "planted_from": src_id, "planted_claim": claim,
        })
        old = old_grades.get(new_id, {})
        same = old.get("answer") == answer and old.get("context") == src["context"]
        grades_doc["grades"].append({
            "id": new_id, "question": src["question"], "context": src["context"],
            "answer": answer, "human": old["human"] if same else None,
            "note": old.get("note", "") if same else "",
        })

    for path, doc in ((ANSWERS, answers_doc), (GRADES, grades_doc)):
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"planted {len(PLANTS)} answers")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    plant()
