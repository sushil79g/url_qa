import streamlit as st
import time

from src.sst import download_wav_file, extract_text
from src.qna import answer_ques

# def download_tube(link):
#     st.write("Downloading Video")
#     time.sleep(2)
#     st.write("Download complete, Converting video to text")
#     return "here"

# def convert_text(file):
#     time.sleep(2)
#     st.write("successfully converted")
#     return "i am speech to text"

def qna(text):
    st.text("AI video reading complete")
    question = st.text_input('Ask question related to video input', "")
    if not question:
        st.stop()
    st.write("analysing answer, Machine learning on work!")
    answer = answer_ques(text, question)
    st.write("Answer is:::", answer)
    return "yes! you are right"

st.title("YOUtube Question Answering: Awesome project")
title = st.text_input('YOutube video link', "")
if not title:
    st.stop()
text = extract_text(title)
answer = qna(text)