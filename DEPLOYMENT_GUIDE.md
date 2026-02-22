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

## ☁️ Option 2: Streamlit Community Cloud (Best for "Web App")

This is the recommended way to have a public link for your bot.

### 1. Account Setup
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"Create app"**.
3. Select your repository: `Bikashkr01/YouTube_ChatBot`.
4. Set the Main file path: `Youtube_ChatBot/app.py`.

### 2. Configure Secrets (CRITICAL)
Before clicking "Deploy", click on **"Advanced settings"** and enter your Groq API key in the **Secrets** box exactly like this:

```toml
GROQ_API_KEY = "your_key_here"
```

### 3. Deploy
1. Click **"Deploy!"**.
2. Streamlit will install `ffmpeg` (from `packages.txt`) and your Python libraries automatically.
3. Your app will be live at a URL like `https://youtube-chatbot-bikash.streamlit.app`.

---

## 🔒 Security Tips
- **Firewall**: Ensure port `8501` is open in your server's firewall (ufw/iptables).
- **Reverse Proxy**: For production, use Nginx with SSL (Let's Encrypt) to serve the app over `https://your-domain.com`.

## 🚀 Post-Deployment
- If transcription is slow, check if the server has a GPU.
- You can monitor logs using: `docker logs -f youtube_chatbot`
