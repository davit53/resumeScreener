import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_emb: np.ndarray, job_emb: np.ndarray) -> float:
    """Return cosine similarity score (0–100)."""
    score = cosine_similarity([resume_emb], [job_emb])[0][0]
    return round(float(score) * 100, 2)

def compare_skills(resume_skills: list, job_skills: list) -> tuple:
    matched = sorted(set(resume_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(resume_skills))
    return matched, missing

def generate_suggestions(missing_skills: list) -> list:
    suggestions = []
    for skill in missing_skills:
        suggestions.append(f"- Add or highlight experience with **{skill}**")
    return suggestions
