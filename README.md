# AI Resume Screening System (ATS-Based)

An AI-powered Resume Screening and Ranking System inspired by real Applicant Tracking Systems (ATS).  
The system analyzes resumes against a Job Description (JD) using NLP, semantic similarity, and skill-gap analysis.

---

## Features

- Resume parsing (PDF, DOCX, TXT)
- Job Description skill extraction
- ATS-style weighted scoring
- Skill gap analysis (Matched & Missing skills)
- Resume rejection rules (Low skill match / No skills section)
- Semantic similarity using SBERT
- Fallback TF-IDF similarity
- Candidate ranking system
- Streamlit interactive UI

---

##  How It Works

1. Extract resume text  
2. Parse sections (Skills, Experience, Projects, Education)  
3. Extract required skills from Job Description  
4. Compute:
   - Skill Match Ratio
   - Section-based Similarity
   - Weighted ATS Score  
5. Apply rejection rules  
6. Rank candidates based on final score  

Final Score =  
0.40 × Skill Match +  
0.30 × Experience Similarity +  
0.20 × Skills Similarity +  
0.10 × Projects Similarity  

---

##  Tech Stack

- Python
- Streamlit
- Sentence-Transformers (SBERT)
- Scikit-learn
- Pandas
- PyPDF2 / python-docx

---

## ▶ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
