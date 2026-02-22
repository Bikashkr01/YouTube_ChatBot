import re

def contains_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(re.search(rf"\b{re.escape(term.lower())}\b", t) for term in terms)

def topic_guard(transcript_text: str) -> str:
    """
    Returns: 'project' if it looks like LangChain/RAG project video,
             'general' otherwise.
    """
    project_terms = ["langchain", "rag", "retrieval", "vector", "faiss", "embedding", "youtube chatbot"]
    return "project" if contains_any(transcript_text, project_terms) else "general"