import os

import torch
from glob import glob

import os
import re
import subprocess
import streamlit as st
from pytube import YouTube
import gdown
import dask
from .third_party.utils import prepare_model_input, read_batch, Decoder


# @st.cache
def load_sst_model(model_dir, model_name,link_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(os.path.join(model_dir,model_name)):
        url =  "https://drive.google.com/uc?id="
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        link = str(url)+str(link_id)
        gdown.download(link, os.path.join(model_dir, model_name), quiet=False)
    model = torch.jit.load(os.path.join(model_dir,model_name), map_location=device)
    
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
def download_wav_file(video_url, video_dir):
    yt = YouTube(video_url)
    old_filename = os.path.join(video_dir, yt.streams.first().default_filename)
    new_filename = re.sub(r'.mp4', '', old_filename)+str(".wav")
    if not os.path.exists(new_filename):
        tube_wav = yt.streams.filter(only_audio=True, file_extension='mp4')
        tube_wav[0].download(video_dir)
        subprocess.call(["ffmpeg","-i", old_filename, new_filename])
        subprocess.call(["rm","-rf", old_filename])
    return new_filename

    


def extract_text(model_dir, model_name, video_url, video_dir,link_id,text_dir):
    if not os.path.exists(text_dir):
        model, decoder, device = load_sst_model(model_dir, model_name, link_id)
        new_filepath = download_wav_file(video_url, video_dir)
        input_set = prepare_model_input(read_batch([new_filepath]),device=device)
        dmodel = dask.delayed(model.cpu())
        predictions = [predict(input_set, dmodel, decoder)]
        text = dask.compute(*predictions)[0]
        file_txt = open(text_dir,"w")
        file_txt.write(text)
        file_txt.close()
    else:
        file_txt = open(text_dir,"r")
        text = file_txt.read()

    return text
    

