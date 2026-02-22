# test_rag.py
from ingestion import download_audio, get_segments, extract_video_id, create_vector_store_from_segments
from retrieval import HybridRetriever, make_multi_query_rewriter
from generation import make_answer_chain, format_evidence
from compression import compress_docs_extractive
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

video_url = "https://www.youtube.com/watch?v=J5_-l7WIO_w"

# 1) Ingest with timestamped segments
audio = download_audio(video_url)
# Fixed: ingestion.py uses (audio_path, model_size, device, compute_type)
_, _, audio_path_str = audio # download_audio returns (video_id, title, audio_path)
segments = get_segments(extract_video_id(video_url), audio_path_str) # Use get_segments to handle captions/fallback
vs = create_vector_store_from_segments(segments, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# 2) Build BM25 docs from FAISS docstore
all_docs = list(vs.docstore._dict.values())
hybrid = HybridRetriever.from_vector_store(vs, all_docs)

# 3) Multi-query rewriting
rewriter = make_multi_query_rewriter(model="mistral", n=3)
question = "What is DeepMind and what are the key achievements discussed?"
queries = rewriter.invoke(question)
print("Rewritten queries:", queries)

# 4) Retrieve across all queries
docs = []
seen = set()
for q in queries:
    for d in hybrid.invoke(q, k=6):
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            docs.append(d)
docs = docs[:8]

# 5) Extractive compression (no hallucinations)
docs = compress_docs_extractive(docs, question)

# 6) Format context with timestamps
evidence = format_evidence(docs)

# 7) Generate answer
chain = make_answer_chain(model="mistral")
resp = chain.invoke({"evidence": evidence, "question": question})

print(resp.content)