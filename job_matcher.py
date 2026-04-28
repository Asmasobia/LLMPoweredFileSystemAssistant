import json
import re
import time
from resume_rag import ResumeRAG


SAMPLE_JOB_DESCRIPTIONS = {
    "ml_engineer": {
        "title": "Machine Learning Engineer",
        "description": "We are looking for a Machine Learning Engineer with 3+ years of experience. "
                       "Must have strong Python skills and experience with PyTorch or TensorFlow. "
                       "Experience with NLP, model deployment, and MLOps is preferred. "
                       "Familiarity with cloud platforms (AWS/GCP) is a plus.",
        "must_have": ["Python", "Machine Learning"],
        "nice_to_have": ["PyTorch", "TensorFlow", "NLP", "AWS", "GCP"],
        "min_experience_years": 3,
    },
    "backend_engineer": {
        "title": "Senior Backend Engineer",
        "description": "Seeking a Senior Backend Engineer with 5+ years of experience "
                       "building scalable APIs and microservices. Must be proficient in Java or Python. "
                       "Experience with Spring Boot, Django, or FastAPI required. "
                       "Database expertise (PostgreSQL, Redis) and Docker/Kubernetes knowledge needed.",
        "must_have": ["Python", "SQL"],
        "nice_to_have": ["Java", "Spring Boot", "Django", "FastAPI", "Docker", "Kubernetes", "PostgreSQL", "Redis"],
        "min_experience_years": 5,
    },
    "devops_engineer": {
        "title": "DevOps Engineer",
        "description": "Looking for a DevOps Engineer with experience in CI/CD pipelines, "
                       "container orchestration, and cloud infrastructure. Must have hands-on experience "
                       "with Docker, Kubernetes, and Terraform. AWS certification preferred. "
                       "Scripting skills in Python or Bash required.",
        "must_have": ["Docker", "Kubernetes", "Terraform"],
        "nice_to_have": ["AWS", "Python", "CI/CD", "Ansible", "Jenkins"],
        "min_experience_years": 3,
    },
    "frontend_developer": {
        "title": "Frontend Developer",
        "description": "We need a Frontend Developer skilled in React and TypeScript. "
                       "Experience with responsive design, component libraries, and state management. "
                       "Familiarity with Next.js, testing frameworks, and CI/CD pipelines is a plus. "
                       "Strong UI/UX sensibility preferred.",
        "must_have": ["React", "JavaScript"],
        "nice_to_have": ["TypeScript", "Next.js", "Vue.js", "Figma", "CI/CD"],
        "min_experience_years": 2,
    },
    "data_scientist": {
        "title": "Data Scientist",
        "description": "Hiring a Data Scientist with expertise in statistical analysis, "
                       "machine learning, and data visualization. Must be proficient in Python and SQL. "
                       "Experience with Pandas, Scikit-learn, and Tableau or Power BI required. "
                       "Advanced degree in a quantitative field preferred.",
        "must_have": ["Python", "SQL", "Machine Learning"],
        "nice_to_have": ["Pandas", "Scikit-learn", "Tableau", "Power BI", "TensorFlow", "R"],
        "min_experience_years": 3,
    },
}


class JobMatcher:
    def __init__(self):
        self.rag = ResumeRAG()

    def keyword_search(self, candidate_skills, must_have, nice_to_have):
        """Check keyword matches for must-have and nice-to-have skills."""
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        must_matches = [s for s in must_have if s.lower() in candidate_skills_lower]
        nice_matches = [s for s in nice_to_have if s.lower() in candidate_skills_lower]
        must_have_met = len(must_matches) == len(must_have)
        return {
            "must_have_matches": must_matches,
            "nice_to_have_matches": nice_matches,
            "all_must_have_met": must_have_met,
        }

    def compute_score(self, semantic_score, keyword_result,
                      experience_years, min_experience,
                      must_have, nice_to_have):
        """Compute match score 0-100."""
        sem_score = max(semantic_score, 0) * 40

        if must_have:
            must_ratio = len(keyword_result["must_have_matches"]) / len(must_have)
        else:
            must_ratio = 1.0
        must_score = must_ratio * 30

        if nice_to_have:
            nice_ratio = len(keyword_result["nice_to_have_matches"]) / len(nice_to_have)
        else:
            nice_ratio = 1.0
        nice_score = nice_ratio * 15

        if min_experience > 0:
            exp_ratio = min(experience_years / min_experience, 1.5) / 1.5
        else:
            exp_ratio = 1.0
        exp_score = exp_ratio * 15

        total = sem_score + must_score + nice_score + exp_score
        return min(int(round(total)), 100)

    def generate_reasoning(self, candidate_name, keyword_result,
                           experience_years, min_experience, score):
        """Generate human-readable match reasoning."""
        reasons = []

        if keyword_result["all_must_have_met"]:
            reasons.append("Meets all must-have requirements: {}".format(
                ", ".join(keyword_result["must_have_matches"])))
        else:
            reasons.append("Must-have skills matched: {}".format(
                ", ".join(keyword_result["must_have_matches"]) or "None"))

        if keyword_result["nice_to_have_matches"]:
            reasons.append("Nice-to-have skills: {}".format(
                ", ".join(keyword_result["nice_to_have_matches"])))

        if experience_years >= min_experience:
            reasons.append("Experience ({} yrs) meets minimum ({} yrs).".format(
                experience_years, min_experience))
        else:
            reasons.append("Experience ({} yrs) below minimum ({} yrs).".format(
                experience_years, min_experience))

        if score >= 80:
            reasons.append("Strong overall match.")
        elif score >= 60:
            reasons.append("Moderate match.")
        else:
            reasons.append("Weak match.")

        return " ".join(reasons)

    def match_job(self, job_key=None, job_description=None, top_k=10):
        """Match a job description against stored resumes."""
        if job_key and job_key in SAMPLE_JOB_DESCRIPTIONS:
            jd = SAMPLE_JOB_DESCRIPTIONS[job_key]
        elif job_description:
            jd = job_description
        else:
            return {"error": "Job key '{}' not found.".format(job_key)}

        start_time = time.time()

        results = self.rag.query(jd["description"], n_results=top_k * 3)

        candidates = {}
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            name = meta["candidate_name"]
            semantic_score = 1 - distance

            if name not in candidates:
                skills = json.loads(meta.get("skills", "[]"))
                candidates[name] = {
                    "candidate_name": name,
                    "resume_path": meta["source"],
                    "skills": skills,
                    "experience_years": meta.get("experience_years", 0),
                    "education": meta.get("education", ""),
                    "semantic_scores": [],
                    "relevant_excerpts": [],
                }

            candidates[name]["semantic_scores"].append(semantic_score)
            if len(candidates[name]["relevant_excerpts"]) < 3:
                candidates[name]["relevant_excerpts"].append(doc[:300])

        top_matches = []
        must_have = jd.get("must_have", [])
        nice_to_have = jd.get("nice_to_have", [])
        min_exp = jd.get("min_experience_years", 0)

        for name, cand in candidates.items():
            avg_semantic = sum(cand["semantic_scores"]) / len(cand["semantic_scores"])
            keyword_result = self.keyword_search(cand["skills"], must_have, nice_to_have)
            score = self.compute_score(
                avg_semantic, keyword_result,
                cand["experience_years"], min_exp,
                must_have, nice_to_have,
            )
            reasoning = self.generate_reasoning(
                name, keyword_result,
                cand["experience_years"], min_exp, score,
            )
            top_matches.append({
                "candidate_name": name,
                "resume_path": cand["resume_path"],
                "match_score": score,
                "matched_skills": keyword_result["must_have_matches"] + keyword_result["nice_to_have_matches"],
                "experience_years": cand["experience_years"],
                "education": cand["education"],
                "relevant_excerpts": cand["relevant_excerpts"],
                "reasoning": reasoning,
            })

        top_matches.sort(key=lambda x: x["match_score"], reverse=True)
        top_matches = top_matches[:top_k]

        latency = time.time() - start_time

        return {
            "job_title": jd["title"],
            "job_description": jd["description"][:200] + "...",
            "must_have": must_have,
            "nice_to_have": nice_to_have,
            "min_experience_years": min_exp,
            "top_matches": top_matches,
            "latency_seconds": round(latency, 3),
        }


def print_results(result):
    """Pretty print match results."""
    print("\n" + "=" * 70)
    print("JOB: {}".format(result["job_title"]))
    print("Must-have: {}".format(", ".join(result["must_have"])))
    print("Nice-to-have: {}".format(", ".join(result["nice_to_have"])))
    print("Min Experience: {} years".format(result["min_experience_years"]))
    print("Latency: {}s".format(result["latency_seconds"]))
    print("=" * 70)

    for i, match in enumerate(result["top_matches"], 1):
        print("\n  [{}] {} - Score: {}/100".format(i, match["candidate_name"], match["match_score"]))
        print("      File: {}".format(match["resume_path"]))
        print("      Skills Matched: {}".format(", ".join(match["matched_skills"])))
        print("      Experience: {} years".format(match["experience_years"]))
        print("      Education: {}".format(match["education"][:100]))
        print("      Reasoning: {}".format(match["reasoning"]))
        if match["relevant_excerpts"]:
            print("      Excerpt: {}...".format(match["relevant_excerpts"][0][:150]))


if __name__ == "__main__":
    matcher = JobMatcher()

    for job_key in SAMPLE_JOB_DESCRIPTIONS:
        result = matcher.match_job(job_key)
        print_results(result)

    # Save results to JSON
    all_results = {}
    for job_key in SAMPLE_JOB_DESCRIPTIONS:
        all_results[job_key] = matcher.match_job(job_key)

    with open("match_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to match_results.json")