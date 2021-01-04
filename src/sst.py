import os

import torch
from glob import glob

import subprocess
import streamlit as st
from pytube import YouTube

import dask
from .utils import prepare_model_input, read_batch, Decoder

# @st.cache
def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load("silero-model/sst_model.pt", map_location=device)
    decoder = Decoder(model.labels)
    return model, decoder, device

@dask.delayed
def predict(input_batch, model, decoder):
    with torch.no_grad():
        out = model(input_batch)
        for example in out:
            text = decoder(example.cpu())
    return text

# @st.cache
def download_wav_file(video_url, video_name, video_path):
    yt = YouTube(video_url)
    tube_wav = yt.streams.filter(only_audio=True, file_extension='mp4')
    tube_wav[0].download(video_path)
    old_filename = os.path.join(video_path, yt.streams.first().default_filename)
    new_filename = os.path.join(video_path,video_name)
    command2wav = "ffmpeg -i " + old_filename +" "+ new_filename
    subprocess.call(["ffmpeg","-i", old_filename, new_filename])
    subprocess.call(["rm","-rf", old_filename])
    return os.path.join(video_path, video_name)



def extract_text(video_url="https://www.youtube.com/watch?v=WVPcKah4CbA", video_name="abc.wav", video_path="../dataset/"):
    model, decoder, device = load_model()
    new_filepath = os.path.join(video_path, video_name)
    if not os.path.exists("../dataset/"+video_name):
        new_filepath = download_wav_file(video_url, video_name, video_path)
    input_set = prepare_model_input(read_batch([new_filepath]),device=device)
    dmodel = dask.delayed(model.cpu())
    predictions = [predict(input_set, dmodel, decoder)]
    text = dask.compute(*predictions)[0]
    return text
    
    