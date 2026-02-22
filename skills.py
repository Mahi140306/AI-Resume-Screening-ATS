import re

# You can expand this list anytime
SKILLS = [
    "python", "java", "c", "c++", "javascript", "html", "css",
    "sql", "mysql", "mongodb",
    "machine learning", "deep learning", "nlp", "data science",
    "tensorflow", "keras", "pytorch",
    "flask", "django", "streamlit",
    "git", "github",
    "docker", "kubernetes",
    "aws", "azure", "gcp",
    "excel", "power bi", "tableau"
]

def normalize_text(text: str) -> str:
    """Normalize text for skill matching (keep it simple and consistent)."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_skills(text: str, skills_list=SKILLS):
    """
    Extracts skills present in the given text using phrase matching.
    Returns a sorted list of matched skills.
    """
    text = normalize_text(text)

    matched = set()
    for skill in skills_list:
        # word boundary safe matching for single words, phrase matching for multi-words
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            matched.add(skill)

    return sorted(matched)

def skill_gap_analysis(resume_text: str, jd_text: str):
    """
    Returns:
    - resume_skills
    - jd_skills
    - matched_skills
    - missing_skills
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    matched_skills = sorted(resume_skills.intersection(jd_skills))
    missing_skills = sorted(jd_skills - resume_skills)

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
