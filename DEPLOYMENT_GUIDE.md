# 🌐 YouTube ChatBot Deployment Guide

This guide explains how to deploy your YouTube ChatBot to a cloud server (VPS) using Docker. This is the most reliable method as it bundles the application with its AI dependencies (Ollama and Whisper).

## 📋 Prerequisites
- A cloud server (VPS) with at least **8GB RAM** (e.g., DigitalOcean, AWS, Google Cloud).
- **Docker** and **Docker Compose** installed on the server.
- (Optional but recommended) An **NVIDIA GPU** for faster transcription and responses.

---

## 🛠️ One-Click Deployment (Docker Compose)

The easiest way to deploy is using the provided `docker-compose.yml` file.

### 1. Upload your code to the server
```bash
git clone https://github.com/Bikashkr01/YouTube_ChatBot.git
cd YouTube_ChatBot
```

### 2. Configure your API Key
Create a `.env` file in the project folder and add your key:
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### 3. Start the application
```bash
docker-compose up -d
```

### 4. Access the App
Open your browser and go to `http://your-server-ip:8501`.

---

## ☁️ Option 2: Streamlit Cloud (Quick but Limited)

You *can* deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting your GitHub repo, but there are major limitations:
- **No Ollama**: Streamlit Cloud doesn't support running local LLMs like Ollama. You would need to modify the code to use an API (like OpenAI or Groq).
- **Low Resources**: Whisper transcription might be very slow or crash on the free tier.

**Recommendation**: Use a VPS with Docker for the best experience.

---

## 🔒 Security Tips
- **Firewall**: Ensure port `8501` is open in your server's firewall (ufw/iptables).
- **Reverse Proxy**: For production, use Nginx with SSL (Let's Encrypt) to serve the app over `https://your-domain.com`.

## 🚀 Post-Deployment
- If transcription is slow, check if the server has a GPU.
- You can monitor logs using: `docker logs -f youtube_chatbot`
