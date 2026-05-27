#lets  convert video to MP3
import os
import subprocess

files = os.listdir('videos')
for file in files:
    print(f'Converting {file} to MP3...')
    # Use ffmpeg to convert video to MP3
    tutorial_number = file.split(" - ")[1]       
    
    file_name = file.split(' - ')[0].strip()
    print(f'Tutorial number: {tutorial_number}, File name: {file_name}')
    subprocess.run(['ffmpeg', '-i', f'videos/{file}', f'audios/{file_name}.mp3'])    