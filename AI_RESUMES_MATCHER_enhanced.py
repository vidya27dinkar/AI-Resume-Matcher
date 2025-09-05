import os
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any

import docx
import PyPDF2
import pandas as pd
import numpy as np
from tqdm import tqdm

# NLP & embeddings
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load small english model for NER/lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # User should run: python -m spacy download en_core_web_sm
    raise

# Load an embedding model (change if you prefer different model)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# A starter skills list — extend this with a larger list or load from external file
DEFAULT_SKILLS = {
    "python", "java", "c++", "c#", "javascript", "react", "node", "django",
    "flask", "sql", "nosql", "postgresql", "mongodb", "aws", "azure", "gcp",
    "docker", "kubernetes", "ml", "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "scikit-learn", "git", "rest api", "html", "css",
    "excel", "tableau", "power bi"
}

# Regex patterns for quick extraction
YEARS_PATTERN = re.compile(r"(\d{4})")
EXP_PATTERN = re.compile(r"(\d+)\+?\s*(years|yrs|year)", flags=re.I)
EDU_TERMS = ["bachelor", "master", "b.sc", "m.sc", "b.tech", "m.tech", "phd", "mba", "degree"]


#Text extraction utilities

def extract_text_from_pdf(path: str) -> str:
    text = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                # fallback: skip page if any issue
                continue
    return "\n".join(text)


def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_text(path: str) -> str:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix == '.pdf':
        return extract_text_from_pdf(path)
    elif suffix in ['.docx', '.doc']:
        return extract_text_from_docx(path)
    elif suffix in ['.txt', '.md']:
        return extract_text_from_txt(path)
    else:
        # attempt to read as text
        try:
            return extract_text_from_txt(path)
        except Exception:
            return ""


#Feature extraction

def extract_skills(text: str, skills_vocab: set = None) -> List[str]:
    """Return a list of detected skills (lowercased) using token matching and simple phrase matching."""
    if skills_vocab is None:
        skills_vocab = DEFAULT_SKILLS
    text_low = text.lower()
    found = set()
    # direct substring match for each skill
    for s in skills_vocab:
        if s in text_low:
            found.add(s)
    # Additional heuristic: look for common skill patterns
  
    patterns = [r"experience with ([a-zA-Z0-9\-\+\. #]+)",
                r"proficient in ([a-zA-Z0-9\-\+\. #]+)",
                r"knowledge of ([a-zA-Z0-9\-\+\. #]+)"]
    for pat in patterns:
        for m in re.findall(pat, text_low):
            token = m.strip().split('\n')[0]
            # keep short tokens
            token = token.split(',')[0].strip()
            if 1 < len(token) < 40:
                # break multi-word into words and check
                words = token.split()
                for w in words:
                    if w in skills_vocab:
                        found.add(w)
                if token in skills_vocab:
                    found.add(token)
    return sorted(found)


def extract_education(text: str) -> List[str]:
    text_low = text.lower()
    found = []
    for term in EDU_TERMS:
        if term in text_low:
            # capture line(s) around the term for context
            idx = text_low.find(term)
            start = max(0, idx - 100)
            snippet = text_low[start: idx + 200]
            found.append(snippet.strip())
    return found


def extract_experience_years(text: str) -> Tuple[float, List[str]]:
    """Try to find explicit years of experience e.g., '5 years' or infer from dates in job history."""
    text_low = text.lower()
    m = EXP_PATTERN.search(text_low)
    if m:
        try:
            years = float(m.group(1))
            return years, [m.group(0)]
        except:
            pass
    # fallback: find date ranges like 2018-2022 or 2018 to 2022
    years = re.findall(r"(\b19\d{2}\b|\b20\d{2}\b)", text)
    yrs_found = []
    if years:
        yrs_found = years
        try:
            years_int = sorted(set([int(y) for y in years if y.isdigit()]))
            if len(years_int) >= 2:
                est = abs(years_int[-1] - years_int[0])
                return float(est), yrs_found
        except:
            pass
    return 0.0, yrs_found


def extract_projects(text: str) -> List[str]:
    """Return chunks that look like project descriptions. Heuristic: lines containing 'project' or 'built' or 'developed'."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    proj = []
    for ln in lines:
        if re.search(r"project|projects|developed|built|implementation|implemented", ln, flags=re.I):
            proj.append(ln)
    # return top 5 project-like lines
    return proj[:10]


#  Matching & Scoring 

def embed_texts(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def semantic_similarity(a: str, b: str) -> float:
    v = embed_texts([a, b])
    sim = cosine_similarity([v[0]], [v[1]])[0][0]
    return float(sim)


def skills_match_score(jd_skills: List[str], resume_skills: List[str]) -> float:
    if not jd_skills:
        return 0.0
    matched = set([s.lower() for s in resume_skills]) & set([s.lower() for s in jd_skills])
    return float(len(matched)) / float(len(set(jd_skills)))


def education_score(jd_education: List[str], resume_education: List[str]) -> float:
    # simple heuristic: if resume mentions any EDU_TERMS and JD does too -> partial match
    if not jd_education:
        return 0.0
    jd_join = " ".join(jd_education).lower()
    res_join = " ".join(resume_education).lower()
    # check for degree words
    for term in EDU_TERMS:
        if term in jd_join and term in res_join:
            return 1.0
    # otherwise partial via semantic similarity
    try:
        return semantic_similarity(jd_join, res_join)
    except Exception:
        return 0.0


def projects_score(jd_text: str, resume_projects: List[str]) -> float:
    if not resume_projects:
        return 0.0
    # compute similarity between JD and each project snippet, take best
    sims = []
    for p in resume_projects:
        try:
            sims.append(semantic_similarity(jd_text, p))
        except Exception:
            sims.append(0.0)
    return float(max(sims)) if sims else 0.0


def experience_score(jd_min_years: float, resume_years: float) -> float:
    if jd_min_years <= 0:
        return 0.0
    # if resume_years >= jd_min_years -> full score 1, else proportion
    return float(min(resume_years / jd_min_years, 1.0))


def compute_overall_score(features: Dict[str, Any], weights: Dict[str, float]) -> float:
    total = 0.0
    for k, w in weights.items():
        total += features.get(k, 0.0) * w
    return float(total)


#  High level pipeline 

def parse_jd(jd_text: str, skills_vocab: set = None) -> Dict[str, Any]:
    """Extract skills, education hints, projects hints and min experience from a JD text."""
    if skills_vocab is None:
        skills_vocab = DEFAULT_SKILLS
    skills = extract_skills(jd_text, skills_vocab)
    education = extract_education(jd_text)
    projects = []
    # minimal attempt: look for 'project' or 'responsibilities' block
    projects = extract_projects(jd_text)
    # try to find 'X years' in JD
    m = EXP_PATTERN.search(jd_text.lower())
    jd_years = float(m.group(1)) if m else 0.0
    return {
        'text': jd_text,
        'skills': skills,
        'education': education,
        'projects': projects,
        'min_experience': jd_years
    }


def parse_resume(text: str, skills_vocab: set = None) -> Dict[str, Any]:
    if skills_vocab is None:
        skills_vocab = DEFAULT_SKILLS
    skills = extract_skills(text, skills_vocab)
    education = extract_education(text)
    exp_years, years_found = extract_experience_years(text)
    projects = extract_projects(text)
    return {
        'text': text,
        'skills': skills,
        'education': education,
        'projects': projects,
        'experience_years': exp_years,
        'years_found': years_found
    }


def score_resume_against_jd(jd: Dict[str, Any], resume: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    # features
    f_skills = skills_match_score(jd.get('skills', []), resume.get('skills', []))
    f_edu = education_score(jd.get('education', []), resume.get('education', []))
    f_proj = projects_score(jd.get('text', ''), resume.get('projects', []))
    f_exp = experience_score(jd.get('min_experience', 0.0), resume.get('experience_years', 0.0))
    features = {
        'skills': f_skills,
        'education': f_edu,
        'projects': f_proj,
        'experience': f_exp
    }
    overall = compute_overall_score(features, weights)
    out = {
        'features': features,
        'overall_score': overall
    }
    return out


def generate_ranked_matches(resume_folder: str, jd_file: str, out_csv: str = 'ranked_resumes.csv',
                            skills_vocab: set = None, weights: Dict[str, float] = None) -> pd.DataFrame:
    """Main utility: reads the JD and all resumes in a folder and returns a ranked DataFrame."""
    if skills_vocab is None:
        skills_vocab = DEFAULT_SKILLS
    if weights is None:
        weights = {'skills': 0.4, 'education': 0.2, 'projects': 0.2, 'experience': 0.2}

    # read JD
    jd_text = extract_text(jd_file) if Path(jd_file).exists() else str(jd_file)
    jd = parse_jd(jd_text, skills_vocab)

    records = []
    files = [p for p in Path(resume_folder).glob('*') if p.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt']]

    for f in tqdm(files, desc='Scoring resumes'):
        txt = extract_text(str(f))
        parsed = parse_resume(txt, skills_vocab)
        scored = score_resume_against_jd(jd, parsed, weights)
        rec = {
            'file': str(f.name),
            'path': str(f),
            'overall_score': scored['overall_score'],
            'skills_match': scored['features']['skills'],
            'education_match': scored['features']['education'],
            'projects_match': scored['features']['projects'],
            'experience_match': scored['features']['experience'],
            'resume_skills': ','.join(parsed['skills']) if parsed['skills'] else '',
            'resume_education': ';'.join(parsed['education']) if parsed['education'] else '',
            'resume_projects': ';'.join(parsed['projects']) if parsed['projects'] else '',
            'resume_experience_years': parsed['experience_years']
        }
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        print("No resumes found in folder:", resume_folder)
        return df
    df = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.to_csv(out_csv, index_label='rank')
    print(f"Saved ranked results to {out_csv}")
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rank resumes against a JD')
    parser.add_argument('--resumes', '-r', type=str, default='resumes/', help='Folder with resumes (pdf/docx/txt)')
    parser.add_argument('--jd', '-j', type=str, required=True, help='Job description file (txt/pdf/docx)')
    parser.add_argument('--out', '-o', type=str, default='ranked_resumes.csv', help='Output CSV file')
    args = parser.parse_args()

    df = generate_ranked_matches(args.resumes, args.jd, args.out)
    if not df.empty:
        print(df.head(10).to_string())



