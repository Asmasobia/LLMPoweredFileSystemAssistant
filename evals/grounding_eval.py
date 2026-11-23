"""
Answer-grounding eval: does each RAG answer say only what its excerpts support?

"Grounded" means every factual claim in the answer is supported by the
excerpts the model was given. An answer can be grounded and still unhelpful
(e.g. "the résumés do not say" when they do); that is a retrieval or answer
quality problem, measured elsewhere. This eval asks one question: did the model
add anything the excerpts do not support?

Three steps, so the human grades and the judge look at exactly the same text:

    python evals/grounding_eval.py generate   # retrieve + answer every question; freeze to answers.json
    python evals/grounding_eval.py judge      # LLM judge labels each frozen answer
    python evals/grounding_eval.py agreement  # compare judge with human grades

The judge (openai/gpt-oss-120b) is a larger model than the one answering
(openai/gpt-oss-20b). It is only trusted as far as it agrees with a human on
the same answers: human grades go in evals/human_grades.json, and `agreement`
reports how often the judge matches them, and Cohen's kappa:

    kappa = (observed agreement - agreement expected by chance) / (1 - chance)

Chance agreement matters here because most answers are likely grounded: a
judge that always says "grounded" could agree 85% of the time while being
useless. Kappa scores that judge 0. Above ~0.6 is conventionally "substantial".
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVALS = PROJECT_ROOT / "evals"
QUESTIONS_PATH = EVALS / "grounding_questions.json"
ANSWERS_PATH = EVALS / "results" / "answers.json"
JUDGED_PATH = EVALS / "results" / "judged.json"
HUMAN_PATH = EVALS / "human_grades.json"
AGREEMENT_PATH = EVALS / "results" / "agreement.json"

JUDGE_MODEL = "openai/gpt-oss-120b"
VERDICTS = ("grounded", "ungrounded")

JUDGE_PROMPT = """You check whether an answer is supported by the résumé excerpts it was written from.

List every factual claim the answer makes about candidates (names, years, skills, employers, degrees, certifications, counts, comparisons).
For each, decide whether the excerpts state it or directly imply it. Background knowledge does not count.
Saying the excerpts do not contain something is not a claim about a candidate and is always supported.
The verdict is "grounded" only if every claim is supported; otherwise "ungrounded".

Reply with JSON only:
{"claims": [{"claim": str, "supported": bool}], "verdict": "grounded" | "ungrounded"}"""


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: generate and freeze answers
# ─────────────────────────────────────────────────────────────────────────────
def generate(k=5):
    import pg_store
    import rag_answer

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    conn = pg_store.connect()
    pg_store.create_schema(conn)
    if conn.execute("SELECT count(*) FROM chunks WHERE strategy = 'section'").fetchone()[0] == 0:
        raise SystemExit("The section index is empty. Run: python pg_store.py index")
    embed, client = pg_store.default_embedder(), rag_answer.groq_client()

    answers = []
    for q in questions:
        chunks = rag_answer.retrieve(conn, q["question"], embed, k)
        answers.append({
            "id": q["id"], "question": q["question"], "answerable": q["answerable"],
            "context": rag_answer.format_context(chunks),
            "answer": rag_answer.generate(client, q["question"], chunks),
        })
        print(f"{q['id']} answered")
    write_json(ANSWERS_PATH, {"answer_model": rag_answer.ANSWER_MODEL, "k": k, "answers": answers})


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: judge
# ─────────────────────────────────────────────────────────────────────────────
def judge_one(client, context, answer, model=JUDGE_MODEL) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"EXCERPTS:\n{context}\n\nANSWER:\n{answer}"},
        ],
    )
    return parse_verdict(response.choices[0].message.content)


def parse_verdict(raw: str) -> dict:
    """Validate the judge's JSON. A malformed reply is an error, never a silent 'grounded'."""
    data = json.loads(raw)
    if data.get("verdict") not in VERDICTS:
        raise ValueError(f"Judge returned no valid verdict: {raw[:200]}")
    claims = data.get("claims", [])
    if data["verdict"] == "grounded" and any(not c.get("supported") for c in claims):
        raise ValueError(f"Judge said grounded but marked a claim unsupported: {raw[:200]}")
    return {"verdict": data["verdict"], "claims": claims}


def judge():
    import rag_answer

    frozen = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    client = rag_answer.groq_client()
    judged = []
    for a in frozen["answers"]:
        result = judge_one(client, a["context"], a["answer"])
        judged.append({"id": a["id"], **result})
        print(f"{a['id']} {result['verdict']}")
    write_json(JUDGED_PATH, {"judge_model": JUDGE_MODEL, "answer_model": frozen["answer_model"],
                             "judged": judged})


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: agreement with human grades
# ─────────────────────────────────────────────────────────────────────────────
def cohens_kappa(pairs) -> float:
    """pairs: [(human, judge), ...] with values in VERDICTS."""
    n = len(pairs)
    if n == 0:
        raise ValueError("No graded pairs.")
    observed = sum(h == j for h, j in pairs) / n
    chance = sum(
        (sum(h == v for h, _ in pairs) / n) * (sum(j == v for _, j in pairs) / n) for v in VERDICTS
    )
    if chance == 1.0:
        return 1.0 if observed == 1.0 else 0.0  # both raters used one label throughout
    return (observed - chance) / (1 - chance)


def agreement_report(human: dict, judged: dict) -> dict:
    ids = sorted(i for i, v in human.items() if v in VERDICTS and i in judged)
    pairs = [(human[i], judged[i]) for i in ids]
    confusion = {f"human_{h}/judge_{j}": sum(p == (h, j) for p in pairs) for h in VERDICTS for j in VERDICTS}
    return {
        "graded": len(pairs),
        "agreement": sum(h == j for h, j in pairs) / len(pairs),
        "kappa": cohens_kappa(pairs),
        "confusion": confusion,
        "disagreements": [i for i in ids if human[i] != judged[i]],
    }


def stale_grades(human, answers) -> list:
    """Ids graded by hand on text that is no longer the frozen answer and excerpts."""
    frozen = {a["id"]: (a["answer"], a["context"]) for a in answers}
    return [g["id"] for g in human if g["human"] in VERDICTS
            and frozen.get(g["id"]) != (g["answer"], g["context"])]


def agreement():
    human = json.loads(HUMAN_PATH.read_text(encoding="utf-8"))["grades"]
    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))["answers"]
    stale = stale_grades(human, answers)
    if stale:
        raise SystemExit(f"These were graded on different text than answers.json now holds; "
                         f"regrade them first: {', '.join(stale)}")
    judged = {j["id"]: j["verdict"] for j in json.loads(JUDGED_PATH.read_text(encoding="utf-8"))["judged"]}
    report = agreement_report({g["id"]: g["human"] for g in human}, judged)
    print(json.dumps(report, indent=2))
    write_json(AGREEMENT_PATH, report)
    return report


def human_template():
    """Write human_grades.json with every frozen answer and an empty grade, for grading by hand."""
    if HUMAN_PATH.exists():
        raise SystemExit(f"{HUMAN_PATH} exists; not overwriting grades.")
    frozen = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    write_json(HUMAN_PATH, {
        "about": "Grade each answer against ITS OWN context only: 'grounded' if every claim about "
                 "a candidate is supported by the context, else 'ungrounded'. Leave null to skip. "
                 "Grade before looking at evals/results/judged.json.",
        "grades": [{"id": a["id"], "question": a["question"], "context": a["context"],
                    "answer": a["answer"], "human": None, "note": ""} for a in frozen["answers"]],
    })


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Answer-grounding eval.")
    p.add_argument("step", choices=["generate", "template", "judge", "agreement"])
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    {"generate": generate, "template": human_template, "judge": judge, "agreement": agreement}[args.step]()


if __name__ == "__main__":
    main()
