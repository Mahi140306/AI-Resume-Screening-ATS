import pandas as pd
from src.preprocessing import clean_text
from src.similarity import get_similarity_score
from src.skills import skill_gap_analysis
from src.scoring import compute_final_score
from src.ats_scoring import compute_ats_score


def rank_resumes(resume_texts: list, jd_text: str, method="sbert") -> pd.DataFrame:
    jd_clean = clean_text(jd_text)

    results = []
    for idx, rtext in enumerate(resume_texts, start=1):
        r_clean = clean_text(rtext)

        sim = get_similarity_score(r_clean, jd_clean, method=method)
        gap = skill_gap_analysis(r_clean, jd_clean)

        final_score = compute_final_score(sim, gap["skill_match_ratio"])

        results.append({
            "resume_no": idx,
            "similarity_score": sim,
            "matched_skill_count": gap["matched_skill_count"],
            "missing_skill_count": gap["missing_skill_count"],
            "skill_match_ratio": gap["skill_match_ratio"],
            "final_score": final_score
        })

    df = pd.DataFrame(results).sort_values(by="final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df[["rank", "resume_no", "final_score", "similarity_score",
               "skill_match_ratio", "matched_skill_count", "missing_skill_count"]]
