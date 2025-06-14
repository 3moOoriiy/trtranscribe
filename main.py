import whisper
import os

def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result['text']

if __name__ == "__main__":
    video_file = input("Enter path to video file: ")
    if os.path.exists(video_file):
        transcript = transcribe_video(video_file)
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        print("Transcript saved to transcript.txt")
    else:
        print("File not found.")
