import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_emb: np.ndarray, job_emb: np.ndarray) -> float:
    """Return cosine similarity score (0–100) between embeddings."""
    score = cosine_similarity([resume_emb], [job_emb])[0][0]
    return round(float(score) * 100, 2)

def compare_skills(resume_skills: list, job_skills: list) -> tuple:
    """
    Compare extracted skills from resume and job description.
    Returns (matched, missing).
    """
    matched = sorted(set(resume_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(resume_skills))
    return matched, missing

def skill_overlap_score(matched_skills: list, job_skills: list) -> float:
    """
    Compute a skill overlap score (0–100) based on how many job skills
    are covered by the resume.
    """
    if not job_skills:
        return 50.0  # Neutral if we couldn't detect job skills
    ratio = len(matched_skills) / len(job_skills)
    return round(ratio * 100, 2)

def combined_score(semantic_score: float, skill_score: float) -> float:
    """
    Combine semantic similarity and skill overlap into a single score.
    You can tweak the weights based on experimentation.
    """
    # Weight semantic similarity more heavily
    w_semantic = 0.7
    w_skills = 0.3
    final = w_semantic * semantic_score + w_skills * skill_score
    return round(final, 2)

def generate_suggestions(missing_skills: list) -> list:
    """
    Generate human-readable suggestions based on missing skills.
    """
    suggestions = []
    for skill in missing_skills:
        suggestions.append(f"- Add or highlight experience with {skill}")
    if not suggestions:
        suggestions.append("- Your skills already match the job requirements quite well. Consider emphasizing your strongest ones.")
    return suggestions
