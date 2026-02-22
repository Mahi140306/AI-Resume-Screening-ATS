from src.resume_parser import split_into_sections
from src.similarity import get_similarity_score
from src.skills import skill_gap_analysis


def compute_ats_score(resume_text: str, jd_text: str, method="sbert"):
    """
    ATS-style weighted scoring
    """

    sections = split_into_sections(resume_text)

    # Similarity per section
    skill_sim = get_similarity_score(sections["skills"], jd_text, method)
    exp_sim = get_similarity_score(sections["experience"], jd_text, method)
    proj_sim = get_similarity_score(sections["projects"], jd_text, method)

    # Skill gap
    gap = skill_gap_analysis(resume_text, jd_text)
    skill_ratio = gap["skill_match_ratio"] * 100

    # Weighted ATS scoring
    final_score = (
        0.40 * skill_ratio +   # Skills most important
        0.30 * exp_sim +       # Experience
        0.20 * skill_sim +     # Skills text similarity
        0.10 * proj_sim        # Projects
    )

    return round(final_score, 2), gap
