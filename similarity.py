from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_match_score(resume_text: str, jd_text: str) -> float:
    """
    Returns similarity score between resume and job description in percentage.
    """
    corpus = [resume_text, jd_text]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # cosine similarity between resume (0) and jd (1)
    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return round(sim * 100, 2)
