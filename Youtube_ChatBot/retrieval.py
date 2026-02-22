from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import re
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]


@dataclass
class HybridRetriever:
    """
    Hybrid retrieval:
    - Dense: FAISS similarity / MMR (via vectorstore)
    - Sparse: BM25 over the same chunks
    Then merge + deduplicate by content.
    """
    vector_store: any
    bm25: BM25Okapi
    bm25_docs: List[Document]

    @classmethod
    def from_vector_store(cls, vector_store, docs_for_bm25: List[Document]):
        corpus = [_tokenize(d.page_content) for d in docs_for_bm25]
        bm25 = BM25Okapi(corpus)
        return cls(vector_store=vector_store, bm25=bm25, bm25_docs=docs_for_bm25)

    def _bm25_search(self, query: str, k: int = 6) -> List[Document]:
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.bm25_docs[i] for i in top_idx]

    def _dense_search(self, query: str, k: int = 6, mmr: bool = True) -> List[Document]:
        retriever = self.vector_store.as_retriever(
            search_type="mmr" if mmr else "similarity",
            search_kwargs={"k": k, "fetch_k": max(20, k * 4)} if mmr else {"k": k},
        )
        return retriever.invoke(query)

    def invoke(self, query: str, k: int = 8) -> List[Document]:
        dense = self._dense_search(query, k=max(4, k // 2), mmr=True)
        sparse = self._bm25_search(query, k=max(4, k // 2))

        # merge & dedupe by page_content
        seen = set()
        merged: List[Document] = []
        for d in dense + sparse:
            key = d.page_content.strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(d)

        return merged[:k]


# ✅ STRICTER MULTI-QUERY REWRITER
def make_multi_query_rewriter(model: str = "llama-3.3-70b-versatile", n: int = 3):
    llm = ChatGroq(model=model, temperature=0.2)

    def _clean(line: str) -> str:
        line = line.strip()
        # remove leading numbering/bullets like "1.", "2)", "-", "•"
        line = re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", line)
        return line.strip('"').strip()

    def _rewrite(q: str):
        prompt = (
            "You are a search query optimizer. Given a user question, generate {n} short, effective search queries to retrieve relevant transcript chunks.\n"
            "Rules:\n"
            "- Output EXACTLY {n} lines.\n"
            "- Each line MUST be a standalone search query.\n"
            "- NO conversational filler, NO numbering, NO bullets, NO quotes.\n\n"
            "Question: {q}"
        ).format(n=n, q=q)

        raw = llm.invoke(prompt).content.splitlines()
        queries = [_clean(x) for x in raw if _clean(x)]

        # Filter out lines that look like conversational filler
        filtered = []
        for q_line in queries:
            if len(q_line.split()) > 10: continue # too long for a 'query'
            if any(word in q_line.lower() for word in ["sure", "here", "queries", "search"]): continue
            filtered.append(q_line)

        # always include original query
        final = [q] + [x for x in filtered if x.lower() != q.lower()]
        return final[: n + 1]

    return RunnableLambda(_rewrite)