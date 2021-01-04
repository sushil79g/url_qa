import torch
from omegaconf import OmegaConf

import os
import subprocess
from glob import glob
from pytube import YouTube

import pdb
import dask

from utils import split_into_batches, prepare_model_input, read_batch, init_jit_model, Decoder
device = torch.device('cpu')
model = torch.jit.load("../silero-model/sst_model.pt", map_location=device)
decoder = Decoder(model.labels)
# models = OmegaConf.load('latest_silero_models.yml')
# model, decoder = init_jit_model(models.stt_models.en.latest.jit, device=device)
# torch.jit.save(model, "../silero-model/sst_model.pt")

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


@dask.delayed
def predict(input_batch, model, decoder):
    with torch.no_grad():
        out = model(input_batch)
        for example in out:
            text = decoder(example.cpu())
    return text



def extract_text(video_url="https://www.youtube.com/watch?v=WVPcKah4CbA", video_name="abc.wav", video_path="../dataset/"):
    # new_filepath = os.path.join(video_path,video_name)
    new_filepath = download_wav_file(video_url, video_name, video_path)
    # test_files = glob(new_filepath)
    input_set = prepare_model_input(read_batch([new_filepath]),device=device)
    # output_set = model(input_set)
    dmodel = dask.delayed(model.cpu())
    predictions = [predict(input_set, dmodel, decoder)]
    text = dask.compute(*predictions)
    return text


