"""
Retrieval metrics. Pure functions over ranked lists of file names.

recall@k   Of the résumés that should be found for a query, the fraction that
           appear in the top k results. 1.0 means none were missed; 0.5 means
           half were. It ignores order within the top k.

MRR        Mean reciprocal rank. For each query take 1 / (rank of the first
           relevant result): 1.0 if it came first, 0.5 if second, 0.2 if fifth,
           0 if it never appeared within the cutoff. Then average over queries.
           It rewards putting a right answer at the very top, which recall@k
           does not.

Both are averaged over queries, so each query counts equally however many
relevant résumés it has.
"""


def recall_at_k(ranked: list, relevant, k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        raise ValueError("A query with no relevant documents has no recall.")
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list, relevant, cutoff: int = None) -> float:
    relevant = set(relevant)
    for rank, doc in enumerate(ranked[:cutoff] if cutoff else ranked, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
