from sentence_transformers import CrossEncoder

# This model directly scores (query, document) pairs.
# You can swap it for another CrossEncoder if you want.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_score(job_text: str, resume_text: str) -> float:
    """
    Use a CrossEncoder to score how well the resume matches the job description.
    Returns a 0–100 score.
    """
    # CrossEncoder expects a list of (job, resume) pairs.
    raw_score = cross_encoder.predict([(job_text, resume_text)])[0]

    # The raw scores are typically in some unbounded range.
    # We'll squash them into a 0–100 range with a simple transformation.
    # You can tune this later based on real data.
    # Here we assume scores roughly fall in [-5, 5].
    min_val, max_val = -5.0, 5.0
    clipped = max(min(raw_score, max_val), min_val)
    normalized = (clipped - min_val) / (max_val - min_val)  # 0–1
    return round(normalized * 100, 2)
