import whisper
import json
import os 


model = whisper.load_model("medium")

audios = os.listdir('audios')
for audio in audios:
    print(audio)
    title = audio.split('.mp3')[0]

    
    
    
    
result = model.transcribe( audio = f"audios/{audio}", 
                          language="en", fp16=False, task="transcribe", 
                          verbose=True, best_of=5, beam_size=5, temperature=0.0,word_timestamps=False)



def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"



for audio_file in audios.glob("*.mp3"):
    output_file = transcripts.glob / f"{audio_file.stem}.json"

    if output_file.exists():
        print(f"Skipping already done: {audio_file.name}")
        continue

    print(f"Transcribing: {audio_file.name}")

    result = model.transcribe(
        str(audio_file),
        language="en",
        task="transcribe",
        verbose=True
    )




chunks = []
 
for segment in result["segments"]:
        chunks.append({
            "id": segment["id"],
            "source_file": audio_file.name,
            "video_title": audio_file.stem,
            "start": segment["start"],
            "end": segment["end"],
            "start_time": format_timestamp(segment["start"]),
            "end_time": format_timestamp(segment["end"]),
            "text": segment["text"].strip()
        })