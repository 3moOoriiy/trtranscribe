import os
from pytube import YouTube
import whisper

def download_video(url: str, output_dir: str = "downloads") -> str:
    """Download YouTube video as MP3"""
    os.makedirs(output_dir, exist_ok=True)
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    out_file = audio.download(output_path=output_dir)
    
    # Convert to mp3
    base, _ = os.path.splitext(out_file)
    mp3_file = base + '.mp3'
    os.rename(out_file, mp3_file)
    
    return mp3_file

def transcribe_audio(file_path: str, model: str = "base") -> str:
    """Transcribe audio using Whisper"""
    print("Loading Whisper model...")
    model = whisper.load_model(model)
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    return result["text"]

def save_transcript(text: str, filename: str = "transcript.txt"):
    """Save transcript to file"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Video Transcriber")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    args = parser.parse_args()
    
    print(f"Downloading video from {args.url}...")
    audio_file = download_video(args.url)
    
    print("Starting transcription...")
    transcript = transcribe_audio(audio_file, args.model)
    
    output_file = f"transcript_{os.path.basename(audio_file)}.txt"
    save_transcript(transcript, output_file)
    
    print(f"Transcript saved to {output_file}")
