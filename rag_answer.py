"""
Answer a question about the candidates from retrieved résumé chunks (RAG).

    python rag_answer.py "Which candidates have worked with Kafka?"

Retrieval is pg_store's hybrid search over section chunks; the top-k chunks go
to the model as numbered excerpts, and the model is told to use only those.
Whether it actually does is what evals/grounding_eval.py measures.
"""

import argparse
import os
import sys

import pg_store

ANSWER_MODEL = "openai/gpt-oss-20b"

ANSWER_PROMPT = """You answer a recruiter's questions about job candidates.
Use ONLY the résumé excerpts provided. Cite the file name for each fact, e.g. (resume_jane_smith.txt).
If the excerpts do not contain the answer, say that the résumés provided do not say. Do not guess."""


def format_context(chunks) -> str:
    return "\n\n".join(
        f"[{i}] {c['source'].split('/')[-1]} ({c['section']})\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    )


def retrieve(conn, question, embed, k=5):
    return pg_store.search(conn, question, embed, mode="hybrid", strategy="section", k=k)


def generate(client, question, chunks, model=ANSWER_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"EXCERPTS:\n{format_context(chunks)}\n\nQUESTION: {question}"},
        ],
    )
    return response.choices[0].message.content


def groq_client():
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Answer a question from retrieved résumé chunks.")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    conn = pg_store.connect()
    chunks = retrieve(conn, args.question, pg_store.default_embedder(), args.k)
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c['source']} ({c['section']})")
    print()
    print(generate(groq_client(), args.question, chunks))


if __name__ == "__main__":
    main()
