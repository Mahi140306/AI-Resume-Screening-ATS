from preprocessing import clean_text
from similarity import get_match_score
from skills import skill_gap_analysis

def build_features(resume_raw: str, jd_raw: str) -> dict:
    """
    Create explainable numeric features for ML classification.
    """
    # similarity score uses cleaned text
    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)
    sim_score = get_match_score(resume_clean, jd_clean)  # percentage

    # skill gap uses raw text
    report = skill_gap_analysis(resume_raw, jd_raw)

    jd_skill_count = len(report["jd_skills"])
    matched_skill_count = len(report["matched_skills"])
    missing_skill_count = len(report["missing_skills"])

    # Avoid division by zero
    match_ratio = (matched_skill_count / jd_skill_count) if jd_skill_count > 0 else 0.0
    

    return {
        "similarity_score": sim_score,
        "jd_skill_count": jd_skill_count,
        "matched_skill_count": matched_skill_count,
        "missing_skill_count": missing_skill_count,
        "skill_match_ratio": round(match_ratio, 4)
    }

