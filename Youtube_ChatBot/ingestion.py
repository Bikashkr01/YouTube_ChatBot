# ingestion.py

import os
import sys
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import yt_dlp
from faster_whisper import WhisperModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------
# Cache / Global Variables
# --------------------------
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

_WHISPER_MODEL = None

# --------------------------
# Hardware Detection
# --------------------------
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"

# --------------------------
# 1) YouTube URL helpers
# --------------------------
def extract_video_id(url: str) -> str:
    url = url.strip()
    if "youtube.com" in url or "youtu.be" in url:
        parsed = urlparse(url)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/")
        qs = parse_qs(parsed.query)
        v = qs.get("v", [""])[0]
        if v: return v
        path = parsed.path.split("/")
        if "shorts" in path:
            return path[path.index("shorts") + 1]
    return url


def normalize_youtube_url(url: str) -> str:
    vid = extract_video_id(url)
    if not vid:
        raise ValueError("Could not extract video id from URL.")
    return f"https://www.youtube.com/watch?v={vid}"


def get_video_dir(video_id: str) -> Path:
    d = CACHE_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_video_title(url: str) -> str:
    try:
        url = normalize_youtube_url(url)
        ydl_opts = {
            "quiet": True, 
            "skip_download": True, 
            "noplaylist": True,
            "nocheckcertificate": True,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info.get("title") or "Unknown title"
    except Exception:
        return "YouTube Video"


# --------------------------
# 2) Download audio
# --------------------------
def download_audio(url: str) -> tuple[str, str, str]:
    url = normalize_youtube_url(url)
    video_id = extract_video_id(url)

    vdir = get_video_dir(video_id)
    audio_path = vdir / "audio.webm"

    if audio_path.exists() and audio_path.stat().st_size > 1024 * 100:
        return video_id, "", str(audio_path)

    if audio_path.exists():
        audio_path.unlink()

    outtmpl = str(vdir / "audio.%(ext)s")
    ydl_opts = {
        "format": "worstaudio/best",
        "outtmpl": outtmpl,
        "nocheckcertificate": True,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded = next(vdir.glob("audio.*"), None)
    if downloaded is None or downloaded.stat().st_size < 1024 * 50:
        raise RuntimeError("Audio download failed.")

    if downloaded.suffix != ".webm":
        try:
            downloaded.replace(audio_path)
        except Exception:
            audio_path = downloaded

    return video_id, "", str(audio_path)


# --------------------------
# 3) YouTube captions (SAFE VERSION)
# --------------------------
def fetch_captions_segments(video_id: str):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # 1. Try to list all transcripts to find the best match
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # 2. Try English (manual then generated)
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except:
            # 3. Fallback: Take the first available transcript and translate to English
            try:
                # Get the first available transcript (any language)
                first_transcript = next(iter(transcript_list))
                transcript = first_transcript.translate('en')
            except:
                # Last resort: just get anything
                transcript = transcript_list.find_generated_transcript(['en', 'hi', 'es', 'fr'])

        caps = transcript.fetch()
            
        return [
            {
                "start": c["start"],
                "end": c["start"] + c.get("duration", 0.0),
                "text": c["text"],
            }
            for c in caps
            if c.get("text", "").strip()
        ]
    except Exception as e:
        print(f"[DEBUG] Captions not available via API: {e}")
        return None


# --------------------------
# 4) Whisper Transcription
# --------------------------
def transcribe_audio_segments(
    audio_path: str,
    model_size="tiny",
):
    global _WHISPER_MODEL
    device, compute_type = get_device()
    
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() or 4,
            download_root="models",
        )

    segments, info = _WHISPER_MODEL.transcribe(
        audio_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=1000),
    )

    seg_list = []
    for s in segments:
        seg_list.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip()
        })

    return seg_list


# --------------------------
# 5) Create Vector Store
# --------------------------
def create_vector_store_from_segments(
    segments,
    embeddings,
    chunk_size=1200,
    chunk_overlap=150,
):
    docs = [
        Document(
            page_content=seg["text"],
            metadata={"start": seg["start"], "end": seg["end"]}
        )
        for seg in segments
        if seg["text"].strip()
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked = splitter.split_documents(docs)
    return FAISS.from_documents(chunked, embeddings)


# --------------------------
# 6) Combined with Segment Cache
# --------------------------
def get_segments(video_id: str, audio_path: str = None):
    vdir = get_video_dir(video_id)
    cache_file = vdir / "segments.json"

    # 1. Load from cache if exists
    if cache_file.exists():
        print(f"[DEBUG] Loading cached segments for {video_id}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f), "Cached AI Transcription"

    # 2. Try YouTube API
    print(f"[DEBUG] Fetching segments for {video_id}...")
    segments = fetch_captions_segments(video_id)
    if segments:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(segments, f)
        return segments, "YouTube Captions"

    # 3. AI Fallback (ONLY if audio_path is provided)
    if audio_path:
        print(f"[DEBUG] Falling back to Whisper transcription...")
        segments = transcribe_audio_segments(audio_path)
        
        # Save immediately after transcription finishes
        if segments:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(segments, f)
            return segments, "Whisper Transcription (AI)"
        
    return None, None