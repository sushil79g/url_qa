import os

import torch
from glob import glob

import os
import re
import subprocess
import streamlit as st
from pytube import YouTube
import gdown
import pdb
import dask
from .third_party.utils import prepare_model_input, read_batch, Decoder


<<<<<<< HEAD
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
=======
# @st.cache(suppress_st_warning=True)
def load_model(model_path, model_name):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not os.path.exists(os.path.join(model_path,model_name)):
        url =  "https://drive.google.com/uc?id="
        file_id = "1Vz4Sl9wbrl5B1rs2ACQImCL58t6kUiUZ" # go to the sharable link parse the id and put it here
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        link = str(url)+str(file_id)
        gdown.download(link, os.path.join(model_path, model_name), quiet=False)
    # pdb.set_trace()
    model = torch.jit.load("silero-model/sst_model.pt", map_location=device)
>>>>>>> 79e710451a43feea23e5644ddf86d366092c8bf6
    
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
<<<<<<< HEAD
    old_filename = os.path.join(video_dir, yt.streams.first().default_filename)
    wav_filename = re.sub(r'.mp4', '', old_filename)+str(".wav")
    text_filename = re.sub(r'.mp4', '', old_filename)+str(".txt")
    if not os.path.exists(wav_filename):
        tube_wav = yt.streams.filter(only_audio=True, file_extension='mp4')
        tube_wav[0].download(video_dir)
        subprocess.call(["ffmpeg","-i", old_filename, wav_filename])
        subprocess.call(["rm","-rf", old_filename])
    return wav_filename, text_filename

    


def extract_text(model_dir, model_name, video_url, video_dir,link_id):
    
    model, decoder, device = load_sst_model(model_dir, model_name, link_id)
    wav_filepath, text_filepath = download_wav_file(video_url, video_dir)    
    if not os.path.exists(text_filepath):
        input_set = prepare_model_input(read_batch([wav_filepath]),device=device)
        dmodel = dask.delayed(model.cpu())
        predictions = [predict(input_set, dmodel, decoder)]
        text = dask.compute(*predictions)[0]
        file_txt = open(text_filepath,"w")
        file_txt.write(text)
        file_txt.close()
    else:
        file_txt = open(text_filepath,"r")
        text = file_txt.read()

=======
    tube_wav = yt.streams.filter(only_audio=True, file_extension='mp4')
    tube_wav[0].download(video_path)
    old_filename = os.path.join(video_path, yt.streams.first().default_filename)
    new_filename = os.path.join(video_path, video_name)
    command2wav = "ffmpeg -i " + old_filename +" "+ new_filename
    subprocess.call(["ffmpeg","-i", old_filename, new_filename])
    subprocess.call(["rm","-rf", old_filename])
    return os.path.join(video_path, video_name)


# @st.cache(suppress_st_warning=True, hash_funcs={toolz.functoolz.curry: hash}) 
def extract_text(video_url="https://www.youtube.com/watch?v=WVPcKah4CbA", video_path="dataset", video_name="abc.wav", model_path="silero-model", model_name="sst_model.pt"):
    model, decoder, device = load_model(model_path, model_name)
    new_filepath = os.path.join(video_path, video_name)
    if not os.path.exists(os.path.join(video_path,video_name)):
        new_filepath = download_wav_file(video_url, video_path, video_name)
    input_set = prepare_model_input(read_batch([new_filepath]),device=device)
    dmodel = dask.delayed(model.cpu())
    predictions = [predict(input_set, dmodel, decoder)]
    text = dask.compute(*predictions)[0]
>>>>>>> 79e710451a43feea23e5644ddf86d366092c8bf6
    return text
    

