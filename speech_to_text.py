import whisper
import json


model = whisper.load_model("medium")
result = model.transcribe( audio = "audios/sample_clip.mp3", 
                          language="en", fp16=False, task="transcribe", 
                          verbose=True, best_of=5, beam_size=5, temperature=0.0)


chunks = []
for segment in result["segments"]:
    chunks.append({
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        })
        
    print(chunks)
    
with open('output.json', 'w') as f:    
    json.dump(chunks, f)