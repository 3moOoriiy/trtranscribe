import streamlit as st
import whisper
import tempfile

st.title("🎬 Video to Transcript")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "wav", "mp3"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    st.info("Transcribing, please wait...")
    model = whisper.load_model("base")
    result = model.transcribe(tmp_path)
    st.success("Done!")
    st.text_area("Transcript", result['text'], height=300)
