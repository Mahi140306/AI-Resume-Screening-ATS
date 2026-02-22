import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Cleans and preprocesses text for NLP tasks.
    Returns a single cleaned string.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs and emails
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep only alphabets and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize (simple split is enough here)
    tokens = text.split()

    # Remove stopwords + lemmatize
    processed = []
    for word in tokens:
        if word not in _stopwords and len(word) > 2:
            processed.append(_lemmatizer.lemmatize(word))

    return " ".join(processed)
