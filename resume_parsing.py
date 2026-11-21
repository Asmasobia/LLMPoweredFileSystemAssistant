"""
Chunking and metadata extraction, shared by every store.

Both the ChromaDB baseline (`resume_rag.py`) and the Postgres store
(`pg_store.py`) index the output of these functions. That is what makes the
comparison between them fair: if each store chunked differently, a difference in
recall could come from the chunks rather than from the retrieval method, and
there would be no way to tell which.
"""

import datetime
import re

# Only files named like this are résumés. `resumes/` also holds `summary.txt`,
# an assistant-written summary of John Doe; indexing it would put a second
# "John Doe" document in the corpus and distort every eval that touches him.
RESUME_PREFIX = "resume_"

SECTION_HEADERS = [
    "CONTACT", "SUMMARY", "OBJECTIVE", "EXPERIENCE", "WORK EXPERIENCE",
    "EDUCATION", "SKILLS", "TECHNICAL SKILLS", "PROJECTS", "CERTIFICATIONS",
    "ACHIEVEMENTS", "AWARDS", "PUBLICATIONS", "LANGUAGES", "INTERESTS",
    "REFERENCES", "PROFESSIONAL EXPERIENCE",
]
_HEADER_RE = re.compile(
    r"(?i)^(" + "|".join(re.escape(s) for s in SECTION_HEADERS) + r")\s*$"
)

FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 100

CHUNKING_STRATEGIES = ("section", "fixed")


def is_resume(filename: str) -> bool:
    return filename.startswith(RESUME_PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
#  Chunking
# ─────────────────────────────────────────────────────────────────────────────
def chunk(content: str, strategy: str = "section") -> list:
    """Split a résumé into chunks: [{"section": ..., "text": ...}, ...]."""
    if strategy == "section":
        return section_chunks(content)
    if strategy == "fixed":
        return fixed_size_chunks(content)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. Use one of {CHUNKING_STRATEGIES}.")


def section_chunks(content: str) -> list:
    """
    One chunk per section header (EXPERIENCE, SKILLS, ...). Text before the
    first header (name and title) becomes a HEADER chunk.

    Falls back to fixed-size chunks when one chunk or fewer is found, i.e. the
    document has no recognisable headers.
    """
    chunks = []
    current_section = "HEADER"
    current_lines = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append({"section": current_section, "text": text})

    for line in content.split("\n"):
        match = _HEADER_RE.match(line.strip())
        if match:
            flush()
            current_section = match.group(1).upper()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if len(chunks) <= 1:
        return fixed_size_chunks(content)
    return chunks


def fixed_size_chunks(content: str, size: int = FIXED_CHUNK_SIZE,
                      overlap: int = FIXED_CHUNK_OVERLAP) -> list:
    """Chunks of `size` characters, each overlapping the previous by `overlap`."""
    chunks = []
    for start in range(0, len(content), size - overlap):
        text = content[start:start + size].strip()
        if text:
            chunks.append({"section": "CHUNK", "text": text})
        if start + size >= len(content):
            break
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata
# ─────────────────────────────────────────────────────────────────────────────
COMMON_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "Go", "Rust", "R", "SQL",
    "React", "Node.js", "Django", "Flask", "FastAPI", "Spring Boot", "Angular", "Vue.js",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Ansible",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "PostgreSQL", "MongoDB", "Redis", "MySQL", "Elasticsearch",
    "Git", "CI/CD", "Jenkins", "Kafka", "GraphQL", "REST",
    "Swift", "Kotlin", "Flutter", "React Native",
    "Figma", "Tableau", "Power BI", "Spark", "Airflow",
]

# Matched case-sensitively: lower-cased, "go" and "r" are ordinary words ("go
# live", "R&D"), and every résumé containing them was tagged as a Go or R
# developer.
_CASE_SENSITIVE_SKILLS = {"Go", "R"}


def _skill_pattern(skill: str) -> re.Pattern:
    # \b does not work next to "+" (C++) or "." (Node.js), so use lookarounds.
    flags = 0 if skill in _CASE_SENSITIVE_SKILLS else re.IGNORECASE
    return re.compile(r"(?<![\w+#])" + re.escape(skill) + r"(?![\w+#])", flags)


_SKILL_PATTERNS = [(s, _skill_pattern(s)) for s in COMMON_SKILLS]


def extract_skills(content: str) -> list:
    return [skill for skill, pattern in _SKILL_PATTERNS if pattern.search(content)]


_STATED_YEARS_RE = re.compile(r"(?i)\b(\d{1,2})\+?\s+years?\b")
_DATE_RANGE_RE = re.compile(r"(\d{4})\s*[-–]\s*(Present|\d{4})")


def extract_experience_years(content: str, as_of_year: int = None) -> int:
    """
    Years of experience, preferring the figure the résumé states in its SUMMARY.

    Why not just add up the job date ranges, as this used to: "Present" has to
    be resolved against some year, and using the current year makes the number
    grow every January. A résumé that says "8 years" became 10 in 2026, so a
    filter like experience_years >= 9 returned different candidates depending on
    when it ran. Adding ranges also double-counts overlapping jobs.

    Fallback when no figure is stated: the union of the date ranges, so overlaps
    count once, with "Present" resolved to `as_of_year` (default: this year).
    """
    summary = _section_text(content, "SUMMARY")
    stated = _STATED_YEARS_RE.search(summary)
    if stated:
        return int(stated.group(1))

    as_of_year = as_of_year or datetime.datetime.now().year
    experience = _section_text(content, "EXPERIENCE") or content
    ranges = sorted(
        (int(start), as_of_year if end == "Present" else int(end))
        for start, end in _DATE_RANGE_RE.findall(experience)
    )
    total, cur_start, cur_end = 0, None, None
    for start, end in ranges:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def _section_text(content: str, name: str) -> str:
    for c in section_chunks(content):
        if c["section"] == name:
            return c["text"]
    return ""


def extract_metadata(content: str, filename: str) -> dict:
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    education = _section_text(content, "EDUCATION")
    return {
        "candidate_name": lines[0] if lines else "Unknown",
        "skills": extract_skills(content),
        "experience_years": extract_experience_years(content),
        "education": "; ".join(l.strip() for l in education.split("\n") if l.strip()),
        "source_file": filename,
    }
