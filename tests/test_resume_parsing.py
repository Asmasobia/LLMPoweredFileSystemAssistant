"""
Tests for chunking and metadata extraction (`resume_parsing.py`).

These matter to the evals more than they look: `experience_years` is what a
filter like `--min-years 5` compares against, and the chunks are what every
store ranks. A wrong number here makes a filter silently drop the right
candidate, and no retrieval metric would point back to the cause.
"""

from pathlib import Path

import pytest

import resume_parsing as rp

RESUMES = Path(__file__).resolve().parent.parent / "resumes"

SAMPLE = """JANE TEST
Data Engineer

CONTACT
Email: jane@email.com

SUMMARY
Data engineer with 7 years of experience in batch and streaming pipelines.

EXPERIENCE
Senior Data Engineer | Acme | 2021 - Present
- Built Kafka pipelines on AWS

SKILLS
Languages: Python, SQL, Go

EDUCATION
B.S. Computer Science | State University | 2016
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Which files are résumés
# ─────────────────────────────────────────────────────────────────────────────
def test_summary_file_is_not_a_resume():
    assert not rp.is_resume("summary.txt")
    assert rp.is_resume("resume_alex_kumar.txt")


# ─────────────────────────────────────────────────────────────────────────────
#  Chunking
# ─────────────────────────────────────────────────────────────────────────────
def test_section_chunks_split_on_headers_and_keep_the_name_as_header():
    chunks = rp.chunk(SAMPLE, "section")
    assert [c["section"] for c in chunks] == ["HEADER", "CONTACT", "SUMMARY", "EXPERIENCE", "SKILLS", "EDUCATION"]
    assert chunks[0]["text"] == "JANE TEST\nData Engineer"
    assert "Kafka" in chunks[3]["text"]


def test_section_chunking_falls_back_to_fixed_without_headers():
    text = "no headers here " * 100
    assert {c["section"] for c in rp.chunk(text, "section")} == {"CHUNK"}


def test_fixed_chunks_overlap_and_cover_the_whole_text():
    text = "".join(chr(ord("a") + i % 26) for i in range(1234))
    chunks = rp.fixed_size_chunks(text, size=500, overlap=100)
    assert [len(c["text"]) for c in chunks] == [500, 500, 434]
    # Each chunk starts 400 characters after the previous one.
    assert chunks[1]["text"][:100] == chunks[0]["text"][-100:]
    assert text.endswith(chunks[-1]["text"])


def test_fixed_chunks_do_not_emit_a_trailing_chunk_already_covered():
    # 500 characters exactly: one chunk, not a second one holding the last 100.
    assert len(rp.fixed_size_chunks("x" * 500, size=500, overlap=100)) == 1


def test_unknown_strategy_is_an_error():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        rp.chunk(SAMPLE, "semantic")


# ─────────────────────────────────────────────────────────────────────────────
#  Experience years
# ─────────────────────────────────────────────────────────────────────────────
def test_stated_years_in_summary_win_over_date_ranges():
    # 2021 - Present would be 5 in 2026 and 6 in 2027; the résumé says 7.
    assert rp.extract_experience_years(SAMPLE, as_of_year=2026) == 7
    assert rp.extract_experience_years(SAMPLE, as_of_year=2030) == 7


def test_fallback_counts_overlapping_jobs_once():
    text = """SUMMARY
Engineer.

EXPERIENCE
Job A | 2015 - 2020
Job B | 2018 - 2022
Job C | 2024 - Present

EDUCATION
B.S. | 2010 - 2014
"""
    # 2015-2022 is 7 years (not 5 + 4), plus 2024-2026 is 2. The degree is not experience.
    assert rp.extract_experience_years(text, as_of_year=2026) == 9


def test_every_resume_in_the_corpus_matches_the_years_it_states():
    """
    The corpus states its own experience figure in each SUMMARY. The old code
    summed date ranges up to the current year, and was 2 years high on all 31.
    """
    resumes = sorted(RESUMES.glob("resume_*.txt"))
    assert len(resumes) == 31
    years = {p.name: rp.extract_experience_years(p.read_text(encoding="utf-8")) for p in resumes}
    assert years["resume_andrew_young.txt"] == 8
    assert years["resume_alex_kumar.txt"] == 3
    assert years["resume_john_doe.txt"] == 5  # "5+ years"
    assert all(1 <= y <= 15 for y in years.values())


# ─────────────────────────────────────────────────────────────────────────────
#  Skills
# ─────────────────────────────────────────────────────────────────────────────
def test_skills_with_punctuation_are_found():
    skills = rp.extract_skills("Languages: C++, Node.js, CI/CD, Vue.js")
    assert {"C++", "Node.js", "CI/CD", "Vue.js"} <= set(skills)


def test_short_skill_names_are_not_matched_inside_ordinary_words():
    skills = rp.extract_skills("Ready to go live. Wrote C code, not C++ or C#. Javanese Java.")
    assert "Go" not in skills
    assert "R" not in skills
    assert "C++" in skills
    assert "JavaScript" not in skills
    assert "Java" in skills


def test_java_is_not_found_inside_javascript():
    assert rp.extract_skills("Languages: JavaScript") == ["JavaScript"]


def test_go_is_matched_case_sensitively():
    assert "Go" in rp.extract_skills("Languages: Go, Python")
    assert "Go" not in rp.extract_skills("we go to production weekly")


def test_metadata_for_sample():
    meta = rp.extract_metadata(SAMPLE, "resume_jane_test.txt")
    assert meta["candidate_name"] == "JANE TEST"
    assert meta["experience_years"] == 7
    assert {"Python", "SQL", "Go", "Kafka", "AWS"} <= set(meta["skills"])
    assert meta["education"] == "B.S. Computer Science | State University | 2016"
