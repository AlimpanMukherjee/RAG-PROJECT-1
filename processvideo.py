import os
import subprocess
files=os.listdir('vdoResource')
for file in files:
    tutorial_number=file.split('_')[0]
    print(tutorial_number)
    file_name=file.split('_')[1].split('.')[0]
    print(file_name)
    FFMPEG_PATH = r"C:\Users\KIIT0001\Desktop\New folder (2)\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    subprocess.run([FFMPEG_PATH, '-i', f'vdoResource/{file}', f'audio/{tutorial_number}_{file_name}audio.mp3'])
 