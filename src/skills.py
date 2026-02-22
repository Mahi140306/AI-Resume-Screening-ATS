from typing import List, Dict

DEFAULT_SKILLS = [
    "python", "java", "c", "c++", "c#", "sql", "mysql", "mongodb",
    "html", "css", "javascript", "react", "node", "express",
    "machine learning", "deep learning", "nlp", "tensorflow", "keras", "pytorch",
    "data science", "pandas", "numpy", "scikit-learn", "streamlit",
    "flask", "django", "git", "github", "api", "rest", "linux"
]

def extract_skills(text: str, skill_list: List[str] = None) -> List[str]:
    if skill_list is None:
        skill_list = DEFAULT_SKILLS

    text = text.lower()
    found = []
    for skill in skill_list:
        if skill.lower() in text:
            found.append(skill.lower())
    return sorted(list(set(found)))


def skill_gap_analysis(resume_text: str, jd_text: str, skill_list: List[str] = None) -> Dict:
    resume_skills = extract_skills(resume_text, skill_list)
    jd_skills = extract_skills(jd_text, skill_list)

    matched = sorted(list(set(resume_skills).intersection(set(jd_skills))))
    missing = sorted(list(set(jd_skills) - set(resume_skills)))

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "jd_skill_count": len(jd_skills),
        "matched_skill_count": len(matched),
        "missing_skill_count": len(missing),
        "skill_match_ratio": round(len(matched) / len(jd_skills), 4) if len(jd_skills) > 0 else 0.0
    }
