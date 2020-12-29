import os

import torch
from glob import glob

import subprocess
import streamlit as st
from pytube import YouTube
 
device = torch.device('cpu')
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', 
                                       device=device)

def download_wav_file(video_url, video_name, video_path):
    st.write("Download starting..")
    yt = YouTube(video_url)
    tube_wav = yt.streams.filter(only_audio=True, file_extension='mp4')
    tube_wav[0].download(video_path)
    old_filename = os.path.join(video_path, yt.streams.first().default_filename)
    new_filename = os.path.join(video_path,video_name)
    command2wav = "ffmpeg -i " + old_filename +" "+ new_filename
    subprocess.call(["ffmpeg","-i", old_filename, new_filename])
    subprocess.call(["rm","-rf", old_filename])
    st.write("Download Complete")
    return os.path.join(video_path, video_name)



def extract_text(video_url="https://www.youtube.com/watch?v=WVPcKah4CbA", video_name="abc.wav", video_path="../dataset/", util = utils):
    st.text("Extracting text from video")
    new_filepath = download_wav_file(video_url, video_name, video_path)
    (read_batch, split_into_batches, read_audio, prepare_model_input) = utils
    test_files = glob(new_filepath)
    batches = split_into_batches(test_files, batch_size=1)
    input_set = prepare_model_input(read_batch(batches[0]),device=device)
    output_set = model(input_set)
    for example in output_set:
        text = decoder(example.cpu())
    return text


