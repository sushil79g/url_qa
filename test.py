import re
import streamlit as st
import time

from src.sst import download_wav_file, extract_text, load_model
from src.qna import answer_ques

from farm.infer import Inferencer

# def download_tube(link):
#     st.write("Downloading Video")
#     time.sleep(2)
#     st.write("Download complete, Converting video to text")
#     return "here"

# def convert_text(file):
#     time.sleep(2)
#     st.write("successfully converted")
#     return "i am speech to text"

def qna(text, nlp):
    st.text("AI video reading complete")
    question = st.text_input('Ask question related to video input', "")
    if not question:
        st.stop()
    st.write("analysing answer, Machine learning on work!")
    answer = answer_ques(text, question, nlp)
    st.write("Answer is:::", answer)

st.title("YOUtube Question Answering: Awesome project")
st.write("Making Speech to text model ready...")
model, decoder, device = load_model(model_path="silero-model", model_name="sst_model.pt")
st.write("SST model loading complete")
time.sleep(3)
st.write("Extract Text from Speech")
title = st.text_input('YOutube video link', "")
if not title:
    st.stop()
# name = re.sub(r'[^\w\s]','_',title)+".wav"
text = extract_text(title)
time.sleep(3)
st.text("Downloading the video and extracting the text Completed")
model_name="deepset/roberta-base-squad2"
st.text("Load Q aand A model")
if device == "gpu":
    nlp = Inferencer.load(model_name, task_type="question_answering", gpu=True)
else:
    nlp = Inferencer.load(model_name, task_type="question_answering", gpu=False)
st.text("Ask Me Anything")
qna(text, nlp)