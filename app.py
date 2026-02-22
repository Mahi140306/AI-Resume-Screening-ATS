import streamlit as st
import pandas as pd

from src.resume_parser import extract_resume_text
from src.preprocessing import clean_text
from src.similarity import get_similarity_score
from src.skills import skill_gap_analysis
from src.scoring import compute_final_score, final_decision
from src.ranker import rank_resumes


st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screening System (SBERT + Skill Gap + Scoring)")
st.write("Upload resumes and paste Job Description to rank candidates automatically.")

# ---------------------------
# Sidebar Settings
# ---------------------------
st.sidebar.header("⚙ Settings")

similarity_method = st.sidebar.selectbox(
    "Similarity Method",
    ["sbert", "tfidf"],
    index=0
)

threshold = st.sidebar.slider("Suitability Threshold", 0, 100, 55)

st.sidebar.info(
    "🔹 Final Score = 0.4 * Similarity + 0.6 * Skill Score\n\n"
    "Skill Score = skill_match_ratio * 100"
)

# ---------------------------
# Inputs
# ---------------------------
st.subheader("📝 Job Description")
jd_text = st.text_area("Paste Job Description Here", height=220)

st.subheader("📌 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload multiple resumes (PDF / DOCX / TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# ---------------------------
# Process
# ---------------------------
if st.button("🚀 Screen Resumes"):
    if not jd_text.strip():
        st.error("❌ Please paste a Job Description first.")
        st.stop()

    if not uploaded_files:
        st.error("❌ Please upload at least one resume.")
        st.stop()

    jd_clean = clean_text(jd_text)

    resume_texts = []
    resume_names = []

    # Read resumes
    for file in uploaded_files:
        # Save temporarily
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        text = extract_resume_text(temp_path)
        resume_texts.append(text)
        resume_names.append(file.name)

    # Rank resumes
    df_rank = rank_resumes(resume_texts, jd_text, method=similarity_method)

    # Add Resume Names
    df_rank["resume_name"] = [resume_names[i - 1] for i in df_rank["resume_no"]]
    df_rank = df_rank[["rank", "resume_name", "final_score", "similarity_score",
                       "skill_match_ratio", "matched_skill_count", "missing_skill_count"]]

    st.success("✅ Screening Completed!")

    st.subheader("🏆 Ranked Resumes")
    st.dataframe(df_rank, use_container_width=True)

    # Top resume analysis
    top_resume_no = int(df_rank.iloc[0]["rank"])  # rank 1
    top_resume_index = int(df_rank.iloc[0]["resume_name"] in resume_names)

    st.subheader("🔍 Detailed Analysis (Top Resume)")

    # Find the top resume based on rank
    top_resume_name = df_rank.iloc[0]["resume_name"]
    top_resume_idx = resume_names.index(top_resume_name)

    resume_clean = clean_text(resume_texts[top_resume_idx])

    sim_score = get_similarity_score(resume_clean, jd_clean, method=similarity_method)
    gap_report = skill_gap_analysis(resume_clean, jd_clean)

    final_score = compute_final_score(sim_score, gap_report["skill_match_ratio"])
    decision = final_decision(final_score, threshold=threshold)

    col1, col2, col3 = st.columns(3)

    col1.metric("📌 Similarity Score", f"{sim_score}%")
    col2.metric("🛠 Skill Match Ratio", f"{gap_report['skill_match_ratio']*100:.2f}%")
    col3.metric("⭐ Final Score", f"{final_score}%")

    st.markdown(f"### ✅ Final Decision: **{decision}**")

    st.subheader("📌 Skill Gap Analysis")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("### ✅ Matched Skills")
        st.write(gap_report["matched_skills"])

    with colB:
        st.markdown("### ❌ Missing Skills")
        st.write(gap_report["missing_skills"])

    st.info("Tip: Add more skills to skill list for better accuracy.")
