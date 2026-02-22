def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def compute_final_score(similarity_score,
                        skill_match_ratio,
                        missing_skill_count,
                        w_sim=0.45,
                        w_skill=0.55,
                        missing_penalty=7):
    """
    similarity_score: float (0-100)
    skill_match_ratio: float (0-1)
    missing_skill_count: int

    Returns:
        final_score (0-100)
        skill_score (0-100)
    """

    # Skill score from ratio
    skill_score = (skill_match_ratio * 100.0)

    # Penalize missing skills
    skill_score = skill_score - (missing_skill_count * missing_penalty)
    skill_score = clamp(skill_score)

    # Final combined score
    final_score = (w_sim * similarity_score) + (w_skill * skill_score)
    final_score = clamp(final_score)

    return final_score, skill_score
