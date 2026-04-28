import os
import re
import json
import time
import datetime
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import fs_tools

load_dotenv()

# Try ChromaDB, fall back to TF-IDF
USE_CHROMA = False
try:
    import chromadb
    USE_CHROMA = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "resumes"
VECTOR_STORE_PATH = "vector_store.json"


class ResumeRAG:
    def __init__(self):
        print("Initializing RAG system...")

        if USE_CHROMA:
            print("Using ChromaDB with built-in embeddings...")
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            # Uses ChromaDB's default embedding function (no PyTorch needed)
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print("ChromaDB initialized. Chunks in DB: {}".format(self.collection.count()))
        else:
            print("Using TF-IDF vectorizer (fallback)...")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            self.documents = []
            self.metadatas = []
            self.tfidf_matrix = None
            self._load_json_store()

        print("RAG system initialized.")

    def _load_json_store(self):
        """Load existing JSON vector store."""
        if os.path.exists(VECTOR_STORE_PATH):
            with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
                self.metadatas = data.get("metadatas", [])
                if self.documents:
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def _save_json_store(self):
        """Save vector store to JSON file."""
        data = {
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ---- Document Processing Pipeline ----

    def load_resumes(self, directory="resumes"):
        """Load all resumes using fs_tools from Milestone 1."""
        files = fs_tools.list_files(directory)
        resumes = []
        for f in files:
            if isinstance(f, dict) and "path" in f:
                result = fs_tools.read_file(f["path"])
                if result.get("success"):
                    resumes.append({
                        "path": f["path"],
                        "name": f["name"],
                        "content": result["content"],
                        "metadata": result["metadata"],
                    })
        print("Loaded {} resumes.".format(len(resumes)))
        return resumes

    def chunk_resume(self, content, filename):
        """Chunk resume by sections (Education, Experience, Skills, etc.)."""
        sections = [
            "CONTACT", "SUMMARY", "OBJECTIVE", "EXPERIENCE", "WORK EXPERIENCE",
            "EDUCATION", "SKILLS", "TECHNICAL SKILLS", "PROJECTS", "CERTIFICATIONS",
            "ACHIEVEMENTS", "AWARDS", "PUBLICATIONS", "LANGUAGES", "INTERESTS",
            "REFERENCES", "PROFESSIONAL EXPERIENCE",
        ]

        pattern = r"(?i)^(" + "|".join(re.escape(s) for s in sections) + r")\s*$"
        lines = content.split("\n")
        chunks = []
        current_section = "HEADER"
        current_lines = []

        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                if current_lines:
                    chunk_text = "\n".join(current_lines).strip()
                    if chunk_text:
                        chunks.append({
                            "section": current_section,
                            "text": chunk_text,
                            "source": filename,
                        })
                current_section = match.group(1).upper()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append({
                    "section": current_section,
                    "text": chunk_text,
                    "source": filename,
                })

        if len(chunks) <= 1:
            chunks = self._fixed_size_chunk(content, filename)

        return chunks

    def _fixed_size_chunk(self, content, filename, chunk_size=500, overlap=100):
        """Fallback: chunk by character count with overlap."""
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            text = content[i:i + chunk_size].strip()
            if text:
                chunks.append({
                    "section": "CHUNK",
                    "text": text,
                    "source": filename,
                })
        return chunks

    def extract_metadata(self, content, filename):
        """Extract key fields from resume content."""
        lines = [l.strip() for l in content.split("\n") if l.strip()]

        name = lines[0] if lines else "Unknown"

        skills_section = ""
        in_skills = False
        for line in lines:
            if re.match(r"(?i)^(SKILLS|TECHNICAL SKILLS)", line):
                in_skills = True
                continue
            elif re.match(r"(?i)^(EDUCATION|EXPERIENCE|PROJECTS|CERTIFICATIONS|SUMMARY)", line):
                in_skills = False
            elif in_skills:
                skills_section += line + " "

        skills = []
        common_skills = [
            "Python", "Java", "JavaScript", "TypeScript", "C\\+\\+", "Go", "Rust", "R", "SQL",
            "React", "Node.js", "Django", "Flask", "FastAPI", "Spring Boot", "Angular", "Vue.js",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Ansible",
            "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "PostgreSQL", "MongoDB", "Redis", "MySQL", "Elasticsearch",
            "Git", "CI/CD", "Jenkins", "Kafka", "GraphQL", "REST",
            "Swift", "Kotlin", "Flutter", "React Native",
            "Figma", "Tableau", "Power BI", "Spark", "Airflow",
        ]
        full_text = content.lower()
        for skill in common_skills:
            if re.search(r"\b{}\b".format(skill.lower()), full_text):
                skills.append(skill.replace("\\+\\+", "++"))

        year_pattern = r"(\d{4})\s*[-\u2013]\s*(Present|\d{4})"
        matches = re.findall(year_pattern, content)
        total_years = 0
        current_year = datetime.datetime.now().year
        for start, end in matches:
            end_year = current_year if end == "Present" else int(end)
            total_years += end_year - int(start)

        education = ""
        in_edu = False
        for line in lines:
            if re.match(r"(?i)^EDUCATION", line):
                in_edu = True
                continue
            elif re.match(r"(?i)^(SKILLS|EXPERIENCE|PROJECTS|CERTIFICATIONS)", line):
                in_edu = False
            elif in_edu and line:
                education += line + "; "

        return {
            "candidate_name": name,
            "skills": skills,
            "experience_years": total_years,
            "education": education.strip("; "),
            "source_file": filename,
        }

    # ---- Store in Vector DB ----

    def process_and_store(self, directory="resumes"):
        """Full pipeline: load, chunk, embed, store."""
        resumes = self.load_resumes(directory)

        all_ids = []
        all_texts = []
        all_metadatas = []

        for resume in resumes:
            chunks = self.chunk_resume(resume["content"], resume["name"])
            metadata = self.extract_metadata(resume["content"], resume["name"])

            for i, chunk in enumerate(chunks):
                doc_id = "{}_{}_{}".format(resume["name"], chunk["section"], i)
                doc_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", doc_id)
                all_ids.append(doc_id)
                all_texts.append(chunk["text"])
                all_metadatas.append({
                    "source": resume["path"],
                    "section": chunk["section"],
                    "candidate_name": metadata["candidate_name"],
                    "skills": json.dumps(metadata["skills"]),
                    "experience_years": metadata["experience_years"],
                    "education": metadata["education"],
                })

        if not all_texts:
            print("No texts to store.")
            return

        if USE_CHROMA:
            print("Storing {} chunks in ChromaDB (auto-embedding)...".format(len(all_texts)))
            batch_size = 50
            for i in range(0, len(all_texts), batch_size):
                end = min(i + batch_size, len(all_texts))
                self.collection.upsert(
                    ids=all_ids[i:end],
                    documents=all_texts[i:end],
                    metadatas=all_metadatas[i:end],
                )
                print("  Batch {}/{} stored.".format(
                    (i // batch_size) + 1,
                    (len(all_texts) // batch_size) + 1
                ))
            print("Stored {} chunks from {} resumes in ChromaDB.".format(len(all_texts), len(resumes)))
        else:
            self.documents = all_texts
            self.metadatas = all_metadatas
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self._save_json_store()
            print("Stored {} chunks from {} resumes in JSON store.".format(len(all_texts), len(resumes)))

    def query(self, query_text, n_results=10, where=None):
        """Semantic search over stored resumes."""
        if USE_CHROMA:
            count = self.collection.count()
            if count == 0:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            kwargs = {
                "query_texts": [query_text],
                "n_results": min(n_results, count),
            }
            if where:
                kwargs["where"] = where
            return self.collection.query(**kwargs)
        else:
            if not self.documents or self.tfidf_matrix is None:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            query_vec = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[::-1][:n_results]

            documents = [self.documents[i] for i in top_indices]
            metadatas = [self.metadatas[i] for i in top_indices]
            distances = [float(1 - similarities[i]) for i in top_indices]

            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
            }

    def get_all_candidates(self):
        """Get a summary of all candidates in the database."""
        if USE_CHROMA:
            all_data = self.collection.get()
            meta_list = all_data["metadatas"]
        else:
            meta_list = self.metadatas

        candidates = {}
        for meta in meta_list:
            name = meta["candidate_name"]
            if name not in candidates:
                candidates[name] = {
                    "name": name,
                    "source": meta["source"],
                    "skills": json.loads(meta.get("skills", "[]")),
                    "experience_years": meta.get("experience_years", 0),
                    "education": meta.get("education", ""),
                }
        return candidates

    def count(self):
        """Return total number of chunks."""
        if USE_CHROMA:
            return self.collection.count()
        return len(self.documents)

    def reset(self):
        """Delete and recreate the collection/store."""
        if USE_CHROMA:
            try:
                self.chroma_client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self.documents = []
            self.metadatas = []
            self.tfidf_matrix = None
            if os.path.exists(VECTOR_STORE_PATH):
                os.remove(VECTOR_STORE_PATH)
        print("Vector store reset.")


if __name__ == "__main__":
    rag = ResumeRAG()
    rag.reset()

    # Index all resumes
    start = time.time()
    rag.process_and_store("resumes")
    elapsed = time.time() - start
    print("\nIndexing time: {:.2f}s".format(elapsed))

    # Show all candidates
    print("\n--- All Candidates ---")
    candidates = rag.get_all_candidates()
    for name, info in candidates.items():
        print("  {} | {} yrs exp | Skills: {}".format(
            name, info["experience_years"], ", ".join(info["skills"][:5])
        ))

    # Test queries
    test_queries = [
        "Python developer with machine learning experience",
        "Frontend React developer",
        "DevOps engineer with Kubernetes and AWS",
        "Data scientist with SQL and visualization",
        "Mobile app developer with Swift",
    ]

    for query in test_queries:
        print("\n--- Query: {} ---".format(query))
        results = rag.query(query, n_results=5)
        if results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                print("  [{}] {} (similarity: {:.3f}) - {}".format(
                    i + 1, meta["candidate_name"], 1 - dist, meta["section"]
                ))
                print("      Excerpt: {}...".format(doc[:100]))
        else:
            print("  No results found.")

    # Summary stats
    print("\n--- Summary ---")
    print("Total candidates: {}".format(len(candidates)))
    print("Total chunks: {}".format(rag.count()))
    print("Indexing time: {:.2f}s".format(elapsed))
    print("Backend: {}".format("ChromaDB (vector DB)" if USE_CHROMA else "TF-IDF + JSON"))