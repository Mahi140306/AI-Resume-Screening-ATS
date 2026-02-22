from preprocessing import clean_text
from similarity import get_match_score
from skills import skill_gap_analysis

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    resume_raw = read_text_file("resume.txt")
    jd_raw = read_text_file("jd.txt")

    # Preprocess (for similarity)
    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)

    # Similarity score
    score = get_match_score(resume_clean, jd_clean)

    # Skill analysis (use raw text because skill phrases may get altered in cleaning)
    skill_report = skill_gap_analysis(resume_raw, jd_raw)

    print("========== Intelligent Resume Screening ==========")
    print(f"\nMatch Score (TF-IDF + Cosine Similarity): {score}%")

    print("\n--- Skill Report ---")
    print("Resume Skills:", ", ".join(skill_report["resume_skills"]) if skill_report["resume_skills"] else "None")
    print("JD Skills:", ", ".join(skill_report["jd_skills"]) if skill_report["jd_skills"] else "None")

    print("\nMatched Skills:", ", ".join(skill_report["matched_skills"]) if skill_report["matched_skills"] else "None")
    print("Missing Skills:", ", ".join(skill_report["missing_skills"]) if skill_report["missing_skills"] else "None")
