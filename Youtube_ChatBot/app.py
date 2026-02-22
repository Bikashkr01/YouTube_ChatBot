# app.py

import warnings
from pathlib import Path
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------
# Imports
# --------------------------
from ingestion import (
    extract_video_id,
    download_audio,
    get_segments,
    create_vector_store_from_segments,
    get_video_title,
    get_device,
)
from retrieval import HybridRetriever, make_multi_query_rewriter
from generation import make_answer_chain, format_evidence, make_general_knowledge_chain
from compression import compress_docs_extractive
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title("🎥 YouTube Chatbot")

# Hardware Status in Sidebar
device, dtype = get_device()
with st.sidebar:
    st.header("⚙️ System Status")
    st.write(f"**Device:** {device.upper()}")
    st.write(f"**Compute:** {dtype}")
    st.success("⚡ Powered by Groq AI (Llama 3.3)")


# --------------------------
# Cache directory
# --------------------------
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# --------------------------
# Cached embeddings
# --------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------
# BUILD INDEX
# --------------------------
@st.cache_resource(show_spinner=False)
def build_index(video_url: str):

    video_id = extract_video_id(video_url)
    video_cache_dir = CACHE_DIR / video_id
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = video_cache_dir / "faiss_index"
    embeddings = get_embeddings()

    with st.status("Processing Video...", expanded=True) as status:
        
        st.write("🔍 Fetching video metadata...")
        title = get_video_title(video_url)
        
        if (faiss_path / "index.faiss").exists():
            st.write("📦 Loading cached data...")
            vs = FAISS.load_local(
                str(faiss_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            method = "Cached Index"
        else:
            st.write("📥 Downloading audio...")
            _, _, audio_path = download_audio(video_url)

            st.write(f"🎙️ Getting transcript... (using {device.upper()})")
            segments, method = get_segments(video_id, audio_path)

            st.write("🧠 Organizing knowledge...")
            vs = create_vector_store_from_segments(segments, embeddings)
            vs.save_local(str(faiss_path))
        
        status.update(label=f"✅ Ready: {title}", state="complete", expanded=False)

    all_docs = list(vs.docstore._dict.values())
    retriever = HybridRetriever.from_vector_store(vs, all_docs)

    return retriever, title, method


# --------------------------
# QA PIPELINE
# --------------------------
def run_qa(retriever, question: str):
    rewriter = make_multi_query_rewriter(model="llama-3.3-70b-versatile", n=1)
    raw_queries = rewriter.invoke(question)

    queries = []
    for q in raw_queries:
        if len(q.split()) > 12: continue
        if any(w in q.lower() for w in ["sure", "here", "queries"]): continue
        queries.append(q)

    if not queries: queries = [question]

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

    if not docs:
        return "Not discussed in the video.", queries, ""

    evidence = format_evidence(docs)
    if len(evidence.strip()) < 60:
        return "Not discussed in the video.", queries, evidence

    chain = make_answer_chain(model="llama-3.3-70b-versatile")
    answer = chain.invoke({"evidence": evidence, "question": question}).content
    return answer, queries, evidence


# --------------------------
# UI LAYOUT
# --------------------------
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("1. Input 🎥")
    url = st.text_input("YouTube URL")
    
    if st.button("Start Processing", use_container_width=True):
        if not url:
            st.warning("Please enter a URL.")
        else:
            retriever, title, method = build_index(url)
            st.session_state["retriever"] = retriever
            st.session_state["title"] = title
            st.session_state["method"] = method
            st.session_state["ready"] = True

    if st.session_state.get("ready"):
        st.success(f"**Loaded:** {st.session_state['title']}")
        st.caption(f"Source: {st.session_state['method']}")
    else:
        st.info("Paste a URL and start.")

with col2:
    st.header("2. Ask ✅")
    ready = st.session_state.get("ready")
    question = st.text_input("What is discussed in the video?", disabled=not ready)

    if st.button("Ask", use_container_width=True, disabled=not ready):
        if not question:
            st.warning("Ask something first!")
        else:
            retriever = st.session_state["retriever"]
            with st.spinner("Thinking..."):
                answer, queries, evidence = run_qa(retriever, question)

            is_discussed = "[Discussed]" in answer
            
            st.markdown("### 📌 Result")
            st.write(answer)

            if not is_discussed:
                st.write("---")
                st.markdown("### 🌐 General Knowledge Fallback")
                with st.spinner("Searching..."):
                    gen_chain = make_general_knowledge_chain(model="llama-3.3-70b-versatile")
                    fallback_answer = gen_chain.invoke({"question": question}).content
                    st.write(fallback_answer)
            
            with st.expander("Show Technical Details"):
                st.write("**Queries:**", queries)
                if evidence:
                    st.text(evidence)
                else:
                    st.write("No direct video evidence.")