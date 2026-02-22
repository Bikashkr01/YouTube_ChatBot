import re
from typing import List
from langchain_core.documents import Document

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def compress_docs_extractive(docs: List[Document], question: str) -> List[Document]:
    """
    Extractive compression: keeps only sentences containing keywords from the question.
    No LLM -> no hallucination.
    """
    q = question.lower()
    # simple keyword set
    keywords = {w for w in re.findall(r"[a-zA-Z]{3,}", q)}
    out = []
    for d in docs:
        sents = _SENT_SPLIT.split(d.page_content.replace("\n", " "))
        kept = []
        for s in sents:
            s_low = s.lower()
            if any(k in s_low for k in keywords):
                kept.append(s.strip())
        text = " ".join(kept).strip()
        if text:
            out.append(Document(page_content=text, metadata=d.metadata))
        else:
            # keep original if nothing matched (so we don't drop all context)
            out.append(d)
    return out