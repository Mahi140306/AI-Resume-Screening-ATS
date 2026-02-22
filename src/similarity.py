from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_similarity(a: str, b: str) -> float:
    vec = TfidfVectorizer()
    mat = vec.fit_transform([a, b])
    sim = cosine_similarity(mat[0], mat[1])[0][0]
    return round(sim * 100, 2)


def get_similarity_score(resume_clean: str, jd_clean: str, method="sbert") -> float:
    if method == "tfidf":
        return tfidf_similarity(resume_clean, jd_clean)

    # SBERT (try)
    try:
        from sentence_transformers import SentenceTransformer, util

        # load model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        emb1 = model.encode(resume_clean, convert_to_tensor=True)
        emb2 = model.encode(jd_clean, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).item()
        return round(sim * 100, 2)

    except Exception as e:
        print("⚠ SBERT failed, switching to TF-IDF. Reason:", e)
        return tfidf_similarity(resume_clean, jd_clean)
