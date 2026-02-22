import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from retrieval import HybridRetriever, make_multi_query_rewriter
from generation import make_answer_chain, format_evidence, make_general_knowledge_chain
from compression import compress_docs_extractive

# Setup
video_id = "J5_-l7WIO_w"
faiss_path = Path("cache") / video_id / "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"Loading index from {faiss_path}...")
vs = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
all_docs = list(vs.docstore._dict.values())
retriever = HybridRetriever.from_vector_store(vs, all_docs)

rewriter = make_multi_query_rewriter(model="mistral", n=1)
chain = make_answer_chain(model="mistral")

def run_test(question):
    print(f"\n--- TESTING: {question} ---")
    queries = rewriter.invoke(question)
    docs = []
    seen = set()
    for q in queries:
        for d in retriever.invoke(q, k=4):
            key = d.page_content.strip()
            if key and key not in seen:
                seen.add(key)
                docs.append(d)
    docs = docs[:5]
    docs = compress_docs_extractive(docs, question)
    evidence = format_evidence(docs)
    answer = chain.invoke({"evidence": evidence, "question": question}).content
    print("VIDEO ANSWER:")
    print(answer)

    if "[Discussed]" not in answer:
        print("\nFALLBACK ANSWER:")
        gen_chain = make_general_knowledge_chain(model="mistral")
        fallback = gen_chain.invoke({"question": question}).content
        print(fallback)
    print("-" * 30)

# Case 1: Discussed
run_test("is there any discussion about chatbot")

# Case 2: Not Discussed
run_test("does the video talk about cooking pizza?")
