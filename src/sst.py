import os

import torch
from glob import glob

<<<<<<< HEAD
import os
import subprocess
import streamlit as st
from pytube import YouTube
import gdown
import dask
from .utils import prepare_model_input, read_batch, Decoder



# @st.cache
def load_model(model_path, model_name):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not os.path.exists(os.path.join(model_path,model_name)):
        url =  "https://drive.google.com/uc?id="
        file_id = "FILE-ID" # go to the sharable link parse the id and put it here
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        link = str(url)+str(file_id)
        gdown.download(link, os.path.join(model_path, model_name), quiet=False)
    model = torch.jit.load("../silero-model/sst_model.pt", map_location=device)
    
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
def download_wav_file(video_url, video_path, video_name):
=======
import subprocess
import streamlit as st
from pytube import YouTube
 
# @st.cache
def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                        model='silero_stt',
                                        language='en', 
                                        device=device)
    return model, decoder, utils, device

# @st.cache
def download_wav_file(video_url, video_name, video_path):
>>>>>>> 00003f051b63de405197eac806408e34bd4dbe03
    yt = YouTube(video_url)
    tube_wav = yt.streams.filter(only_audio=True, file_extension='mp4')
    tube_wav[0].download(video_path)
    old_filename = os.path.join(video_path, yt.streams.first().default_filename)
<<<<<<< HEAD
    new_filename = os.path.join(video_path, video_name)
=======
    new_filename = os.path.join(video_path,video_name)
>>>>>>> 00003f051b63de405197eac806408e34bd4dbe03
    command2wav = "ffmpeg -i " + old_filename +" "+ new_filename
    subprocess.call(["ffmpeg","-i", old_filename, new_filename])
    subprocess.call(["rm","-rf", old_filename])
    return os.path.join(video_path, video_name)



<<<<<<< HEAD
def extract_text(video_url="https://www.youtube.com/watch?v=WVPcKah4CbA", video_path="dataset", video_name="abc.wav", model_path="silero-model", model_name="sst_model.pt"):
    model, decoder, device = load_model(model_path, model_name)
    new_filepath = os.path.join(video_path, video_name)
    if not os.path.exists(os.path.join(video_path,video_name)):
        new_filepath = download_wav_file(video_url, video_path, video_name)
    input_set = prepare_model_input(read_batch([new_filepath]),device=device)
    dmodel = dask.delayed(model.cpu())
    predictions = [predict(input_set, dmodel, decoder)]
    text = dask.compute(*predictions)[0]
    return text
    
=======
def extract_text( model, decoder, utils, device,  video_url="https://www.youtube.com/watch?v=WVPcKah4CbA", video_name="abc.wav", video_path="../dataset/"):
    # model, decoder, utils, device = load_model()
    new_filepath = os.path.join(video_path, video_name)
    if not os.path.exists("../dataset/"+video_name):
        new_filepath = download_wav_file(video_url, video_name, video_path)
    (read_batch, split_into_batches, _ , prepare_model_input) = utils
    test_files = glob(new_filepath)
    batches = split_into_batches(test_files, batch_size=1)
    input_set = prepare_model_input(read_batch(batches[0]),device=device)
    output_set = model(input_set)
    for example in output_set:
        text = decoder(example.cpu())
    return text


>>>>>>> 00003f051b63de405197eac806408e34bd4dbe03
