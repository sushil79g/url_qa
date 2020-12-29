import os

import subprocess
from pytube import YouTube
 

def download_wav_file(video_url="https://www.youtube.com/watch?v=WVPcKah4CbA", video_name="abc", video_path="../dataset/"):
    yt = YouTube(video_url)
    tube_wav = yt.streams.filter(only_audio=True, file_extension='mp4')
    tube_wav[0].download(video_path)
    old_filename = os.path.join(video_path, yt.streams.first().default_filename)
    new_filename = os.path.join(video_path,video_name)+str(".wav")
    command2wav = "ffmpeg -i " + old_filename +" "+ new_filename
    subprocess.call(["ffmpeg","-i", old_filename, new_filename])
    subprocess.call(["rm","-rf", old_filename])
