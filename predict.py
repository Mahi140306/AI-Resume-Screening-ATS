import joblib
import pandas as pd

from features import build_features
from skills import skill_gap_analysis
from preprocessing import clean_text
from similarity import get_match_score


# ----------------------------
# Final Combined Score Helpers
# ----------------------------
def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def compute_final_score(similarity_score, skill_match_ratio, missing_skill_count,
                        w_sim=0.45, w_skill=0.55, missing_penalty=7):
    """
    similarity_score: float (0-100)
    skill_match_ratio: float (0-1)
    missing_skill_count: int

    Returns:
        final_score (0-100)
        skill_score (0-100)
    """
    # Skill score from ratio
    skill_score = skill_match_ratio * 100.0

    # Penalty for missing skills
    skill_score = skill_score - (missing_skill_count * missing_penalty)
    skill_score = clamp(skill_score)

    # Combined final score
    final_score = (w_sim * similarity_score) + (w_skill * skill_score)
    final_score = clamp(final_score)

    return final_score, skill_score


def predict_suitability(resume_text: str, jd_text: str):
    pack = joblib.load("models/saved_model.pkl")
    model = pack["model"]
    cols = pack["feature_columns"]

    feats = build_features(resume_text, jd_text)
    X = pd.DataFrame([feats])[cols]

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    if prob is not None:
        prob = min(prob, 0.95)

    return pred, prob, feats


if __name__ == "__main__":
    # Example input
    resume_raw = open("resume.txt", "r", encoding="utf-8").read()
    jd_raw = open("jd.txt", "r", encoding="utf-8").read()

    pred, prob, feats = predict_suitability(resume_raw, jd_raw)

    print("========== Prediction ==========")
    print("Features Used:", feats)

    # Explainability output
    report = skill_gap_analysis(resume_raw, jd_raw)

    # Similarity score (your existing method)
    sim = get_match_score(clean_text(resume_raw), clean_text(jd_raw))

    # ----------------------------
    # Final Combined Score Output
    # ----------------------------
    final_score, skill_score = compute_final_score(
        similarity_score=sim,
        skill_match_ratio=feats["skill_match_ratio"],
        missing_skill_count=feats["missing_skill_count"]
    )

    print("\n========== Scoring ==========")
    print(f"Text Similarity Score: {sim:.2f}%")
    print(f"Skill Score: {skill_score:.2f}%")
    print(f"Final Combined Score: {final_score:.2f}%")

    # ----------------------------
    # Final Decision (better logic)
    # ----------------------------
    # ML prediction + score threshold
    final_decision = "Not Suitable ❌"
    if pred == 1 and final_score >= 55:
        final_decision = "Suitable ✅"

    print("\n========== Final Decision ==========")
    print("Final Decision:", final_decision)

    # Still show ML output
    print("\n========== ML Model Output ==========")
    print("Suitability (Model):", "Suitable ✅" if pred == 1 else "Not Suitable ❌")
    
    def confidence_label(p):
        if p >= 0.85:
         return "High"
        elif p >= 0.65:
         return "Medium"
        else:
         return "Low"
    
    if prob is not None:
       prob_display = min(prob, 0.95)
       print(f"Model Probability: {prob_display*100:.2f}% ({confidence_label(prob_display)})")


    # Skill report
    print("\n========== Skill Gap Analysis ==========")
    print("Matched Skills:", report["matched_skills"])
    print("Missing Skills:", report["missing_skills"])
