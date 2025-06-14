import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="🎬 Transcriber", layout="centered")
st.title("🎬 Video to Transcript")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "wav", "mp3", "m4a", "flac"])

if video_file is not None:
    try:
        # Get file extension
        file_ext = os.path.splitext(video_file.name)[1]
        
        # Create temp file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        
        st.info("Transcribing, please wait...")
        
        # Load model with caching
        @st.cache_resource
        def load_whisper_model(model_name="base"):
            return whisper.load_model(model_name)
        
        model = load_whisper_model()
        result = model.transcribe(tmp_path)
        
        st.success("Transcription complete!")
        st.text_area("Transcript", result["text"], height=300)
        
        # Download button for transcript
        st.download_button(
            label="Download Transcript",
            data=result["text"],
            file_name="transcript.txt",
            mime="text/plain"
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error: {e}")
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
else:
    st.warning("Please upload a video/audio file to start.")
