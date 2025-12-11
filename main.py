import argparse
import pdfplumber

from preprocess import clean_text, extract_skills
from embeddings import get_embedding
from scorer import (
    compute_similarity,
    compare_skills,
    skill_overlap_score,
    combined_score,
    generate_suggestions,
)

# Advanced model is optional and only used in advanced mode
try:
    from advanced_model import cross_encoder_score
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

def read_file(path: str) -> str:
    """Reads .txt or .pdf files depending on extension."""
    if path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="AI Resume Screener")
    parser.add_argument("--resume", required=True, help="Path to resume file (.txt or .pdf)")
    parser.add_argument("--job", required=True, help="Path to job description (.txt or .pdf)")
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced"],
        default="basic",
        help="Scoring mode: 'basic' (faster) or 'advanced' (uses a CrossEncoder model).",
    )
    args = parser.parse_args()

    if args.mode == "advanced" and not ADVANCED_AVAILABLE:
        print("⚠ Advanced mode requested, but advanced_model could not be imported.")
        print("  Make sure advanced_model.py exists and dependencies are installed.")
        print("  Falling back to basic mode.\n")
        args.mode = "basic"

    print("\n🔍 Loading files...")
    resume_raw = read_file(args.resume)
    job_raw = read_file(args.job)

    print("🧹 Cleaning text...")
    resume_clean = clean_text(resume_raw)
    job_clean = clean_text(job_raw)

    print("📘 Extracting skills...")
    resume_skills = extract_skills(resume_clean)
    job_skills = extract_skills(job_clean)

    print("📐 Generating embeddings...")
    resume_emb = get_embedding(resume_clean)
    job_emb = get_embedding(job_clean)

    print("📊 Calculating scores...")
    semantic_score = compute_similarity(resume_emb, job_emb)
    overlap_score = skill_overlap_score(resume_skills, job_skills)
    combined = combined_score(semantic_score, overlap_score)

    matched, missing = compare_skills(resume_skills, job_skills)
    suggestions = generate_suggestions(missing)

    advanced_score = None
    if args.mode == "advanced":
        print("🤖 Running advanced CrossEncoder model...")
        advanced_score = cross_encoder_score(job_clean, resume_clean)

    print("\n==============================")
    print("       AI Resume Results      ")
    print("==============================\n")

    print(f"Semantic Match Score      : {semantic_score} / 100")
    print(f"Skill Overlap Score       : {overlap_score} / 100")
    print(f"Combined Match Score      : {combined} / 100")

    if advanced_score is not None:
        print(f"Advanced Model Match Score: {advanced_score} / 100")

    print("\nMatched Skills:", matched)
    print("Missing Skills:", missing)

    if suggestions:
        print("\nSuggestions:")
        for s in suggestions:
            print(s)

    print("\nDone.\n")

if __name__ == "__main__":
    main()
