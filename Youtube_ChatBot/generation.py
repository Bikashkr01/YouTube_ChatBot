# generation.py

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# --------------------------
# Timestamp formatting helper
# --------------------------
def _sec_to_mmss(sec: float) -> str:
    sec = int(sec)
    m = sec // 60
    s = sec % 60
    return f"{m}:{s:02d}"


# --------------------------
# Format evidence for prompt
# --------------------------
def format_evidence(docs: List[Document]) -> str:
    """
    Converts retrieved documents into clean evidence blocks with timestamps.

    Example:
    (0:57–1:02) Some transcript text...
    """
    blocks = []

    for d in docs:
        start = d.metadata.get("start")
        end = d.metadata.get("end")

        if start is not None and end is not None:
            ts = f"({_sec_to_mmss(start)}–{_sec_to_mmss(end)})"
        elif start is not None:
            ts = f"({_sec_to_mmss(start)})"
        else:
            ts = ""

        text = d.page_content.strip().replace("\n", " ")
        blocks.append(f"{ts} {text}")

    return "\n\n".join(blocks)


# --------------------------
# Make answer chain with strict hallucination control
# --------------------------
def make_answer_chain(model: str = "mistral"):
    llm = ChatOllama(
        model=model,
        temperature=0,      # ✅ reduce hallucination by disabling randomness
        num_predict=350,
    )

    system = """You are a YouTube video assistant.

STRICT RULES (must follow):
- Answer ONLY from the provided transcript evidence.
- Do NOT use outside knowledge. Do NOT guess.
- If the evidence does not contain the answer, say exactly:
  "Not discussed in the video."
- Every claim must be backed by at least one timestamp (m:ss or m:ss–m:ss).
- If timestamps are missing in evidence, say "Not discussed in the video."
"""

    human = """EVIDENCE (timestamped transcript chunks):
{evidence}

USER QUESTION:
{question}

Return the answer in this format exactly:

### Status
[Discussed] OR [Not Discussed]

### Answer
<2-4 sentences with timestamps. If not discussed, say: "The provided transcript does not contain information to answer this question.">

### Highlight
> <A concise block highlighting the most relevant part of the discussion with timestamps. If not discussed, say: "N/A">
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),   # ✅ evidence is passed here
    ])

    return prompt | llm


# --------------------------
# Make general knowledge chain (FALLBACK)
# --------------------------
def make_general_knowledge_chain(model: str = "mistral"):
    llm = ChatOllama(
        model=model,
        temperature=0.7,    # ✅ more creative for general knowledge
        num_predict=350,
    )

    system = """You are a helpful AI assistant.
The user asked a question that was not discussed in the YouTube video they are watching.
Provide a clear, helpful answer based on your general knowledge.
"""

    human = """USER QUESTION:
{question}

Answer concisely in 2-4 sentences.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])

    return prompt | llm