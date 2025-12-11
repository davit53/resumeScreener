"""Author: Davit Najaryan"""
"""The prupose of this class is to clean the user inputed resume and extract key words from the clean text"""

#imports
import re
import spacy

nlp = spacy.load("en_core_web_sm")

#this fucntion is responsible for cleaning the text
#Args: text (string)
#returns: string 
def clean_text(text: str) -> str:
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#this function is resposible for extracting key words from a string of text
#args: text (string)
#returns: list containing key words
def extract_skills(text: str) -> list:

    skill_keywords = [
        "python", "java", "c++", "sql", "machine learning", "deep learning",
        "tensorflow", "keras", "aws", "docker", "linux", "react",
        "javascript", "node", "html", "css", "nlp", "data analysis",
        "data science", "git", "pandas", "numpy"
    ]

    doc = nlp(text)
    found = set()

    for keyword in skill_keywords:
        if keyword in text:
            found.add(keyword)

    return sorted(found)
