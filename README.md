# resumeScreener

An AI-powered command-line tool that evaluates how well a candidate’s résumé matches a job description using modern NLP, semantic embeddings, skill extraction, and an optional CrossEncoder relevance model.

# ✨ **Features**

  - Résumé & job description parsing (supports .txt and .pdf)

  - Text cleaning & preprocessing using spaCy

  - Semantic similarity scoring with SentenceTransformers (MiniLM)

  - Skill extraction & comparison

  - Hybrid scoring model (semantic score + skill-overlap score)

  - Advanced BERT-based CrossEncoder for pairwise relevance ranking

  - Automated improvement suggestions for missing skills

  - Modular, production-style Python architecture

  - CLI interface for simple and fast usage


# 🛠 **Installation**

Clone the repository:

git clone https://github.com/davit53/resumeScreener.git

cd resumeScreener


Create a virtual environment:

python -m venv venv


// Mac/Linux

source venv/bin/activate         

// Windows

venv\Scripts\activate            


Install dependencies:

pip install -r requirements.txt

python -m spacy download en_core_web_sm

# 🚀 **Usage**
Basic Mode (fastest)

python main.py --resume sample_data/resume.txt --job sample_data/job.txt --mode basic

Advanced Mode (uses CrossEncoder model)

python main.py --resume sample_data/resume.txt --job sample_data/job.txt --mode advanced

Using PDF Résumés

python main.py --resume myResume.pdf --job jobDescription.txt --mode basic

# 🧠 **Scoring Model Overview**
1. Semantic Similarity (Embeddings)

Uses all-MiniLM-L6-v2 to convert résumé and job text into vector embeddings, then computes cosine similarity.

2. Skill Extraction & Overlap Score
spaCy model + keyword matching to detect skills like:

  - Python, SQL, Java, Machine Learning

  - TensorFlow, Keras

  - AWS, Docker

  - React, JavaScript, etc.

Skill match ratio → 0–100 score.

3. Combined Score

Weighted hybrid:

final = 0.7 * semantic_similarity + 0.3 * skill_overlap

4. Advanced Model (CrossEncoder)

A BERT-based model (ms-marco-MiniLM-L-6-v2) that scores the pair (job, résumé) directly for improved accuracy.
