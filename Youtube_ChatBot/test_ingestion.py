from ingestion import download_audio, transcribe_audio, create_vector_store

video_url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"

print("Downloading...")
audio = download_audio(video_url)
print("Audio:", audio)

print("Transcribing (FAST MODE)...")
transcript = transcribe_audio(audio, language="en", fast=True)
print("Transcript preview:", transcript[:400])

print("Indexing...")
vs = create_vector_store(transcript)
print("DONE ✅ vectors:", vs.index.ntotal)