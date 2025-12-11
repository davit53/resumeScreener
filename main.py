import argparse
import pdfplumber
from preprocess import clean_text, extract_skills
from embeddings import get_embedding
from scorer import compute_similarity, compare_skills, generate_suggestions

def read_file(path: str) -> str:
    """Reads .txt or .pdf files depending on extension."""
    if path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="AI Resume Screener")
    parser.add_argument("--resume", required=True, help="Path to resume file (.txt or .pdf)")
    parser.add_argument("--job", required=True, help="Path to job description (.txt)")
    args = parser.parse_args()

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

    print("📊 Calculating score...")
    match_score = compute_similarity(resume_emb, job_emb)

    matched, missing = compare_skills(resume_skills, job_skills)
    suggestions = generate_suggestions(missing)

    print("\n==============================")
    print("       AI Resume Results      ")
    print("==============================")
    print(f"\nMatch Score: {match_score} / 100\n")

    print("Matched Skills:", matched)
    print("Missing Skills:", missing)

    if suggestions:
        print("\nSuggestions:")
        for s in suggestions:
            print(s)

    print("\nDone.\n")

if __name__ == "__main__":
    main()
