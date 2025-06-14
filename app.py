import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="🎬 Transcriber", layout="centered")
st.title("🎬 Video to Transcript")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "wav", "mp3"])

if video_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.info("Transcribing, please wait (can take a minute)...")

        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        st.success("Transcription complete!")
        st.text_area("Transcript", result["text"], height=300)

        # optional: save output
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(result["text"])

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please upload a video/audio file to start.")
