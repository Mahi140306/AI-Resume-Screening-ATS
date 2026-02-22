import os
from typing import Optional
from docx import Document
import PyPDF2


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def extract_resume_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".pdf":
        return extract_text_from_pdf(path)
    else:
        raise ValueError("Unsupported file type. Use PDF / DOCX / TXT")

SECTION_HEADERS = {
    "education": ["education", "academic", "qualification"],
    "skills": ["skills", "technical skills", "technologies"],
    "experience": ["experience", "work experience", "employment"],
    "projects": ["projects", "academic projects"]
}


def split_into_sections(text: str):
    """
    Split resume text into major sections
    """
    text_lower = text.lower()
    sections = {k: "" for k in SECTION_HEADERS}

    current_section = None
    lines = text.split("\n")

    for line in lines:
        line_lower = line.lower().strip()

        for sec, keywords in SECTION_HEADERS.items():
            if any(k in line_lower for k in keywords):
                current_section = sec
                break

        if current_section:
            sections[current_section] += " " + line

    return sections