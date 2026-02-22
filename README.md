# 🎥 YouTube ChatBot (RAG + Whisper)

A powerful Streamlit-based chatbot that allows you to chat with any YouTube video. It uses **Faster-Whisper** for high-accuracy transcription (especially for Hindi/English) and **RAG (Retrieval-Augmented Generation)** with **Ollama** for intelligent answering.

## ✨ Features
- **Instant Transcription**: Automatically fetches YouTube captions or uses AI fallback (Whisper tiny) to transcribe audio.
- **Hindi & English Support**: Capable of handling bilingual content.
- **RAG Architecture**: Uses FAISS vector store and Hybrid Retrieval for precise timestamp-backed answers.
- **General Knowledge Fallback**: If the video doesn't have the answer, the AI can still help you using its internal knowledge.
- **Hardware Acceleration**: Automatically detects and uses CUDA GPU if available.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) (running locally)
- FFmpeg (for audio processing)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Bikashkr01/YouTube_ChatBot.git
   cd YouTube_ChatBot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the `mistral` model in Ollama:
   ```bash
   ollama pull mistral
   ```

### Running the App
```bash
streamlit run Youtube_ChatBot/app.py
```

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Transcription**: Faster-Whisper
- **Orchestration**: LangChain
- **Embeddings**: HuggingFace (MiniLM-L6-v2)
- **Vector DB**: FAISS
- **LLM**: Ollama (Mistral)
