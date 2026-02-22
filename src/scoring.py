def compute_final_score(similarity_score: float, skill_match_ratio: float) -> float:
    """
    Weighted scoring:
    - skill match is more important than raw similarity
    """
    skill_score = skill_match_ratio * 100
    final_score = (0.4 * similarity_score) + (0.6 * skill_score)
    return round(final_score, 2)


def final_decision(final_score: float, threshold: float = 55.0) -> str:
    return "Suitable ✅" if final_score >= threshold else "Not Suitable ❌"
