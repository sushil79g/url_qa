import yaml
import streamlit as st
from src.sst import extract_text
from src.qna import load_qna_model, answer_ques

st.title("YOUtube Question Answering: Awesome project")

if st.checkbox('Initialze Configuration File'):
    with open('config.yaml') as f:
        config_dict = yaml.safe_load(f)
    st.info("Success!!! CONFIG YAML INITIALIZED")

url_link = st.text_input('Please Provide URL Link Here', "")
if url_link:
    st.info("Success!!! Link Detected")


sst_button, qna_button = st.beta_columns(2)
# if data_button.checkbox("Load Data"):
#     filepath = download_wav_file(video_url=url_link, video_dir=config_dict['video_dir'])
#     st.info("Success!!! Data Loaded")

if sst_button.checkbox("Load & Run SST Model over data"):
    sst_text = extract_text(
        model_dir=config_dict['sst_model_dir'],
        model_name=config_dict['sst_model_name'],
        video_url=url_link,
        video_dir=config_dict['video_dir'],
        link_id= config_dict['link_id']
    )
    st.info("SST Prediction complete")

if qna_button.checkbox("Load QNA Model & Ask Me anything"):
    model_qna = load_qna_model(model_dir=config_dict['qna_model_dir'])
    if st.info("QNA model loading Completed"):
        question = st.text_input('Ask question related to video input', "")
        if question:
            answer = answer_ques(sst_text, question, model_qna)
            if answer:
                st.write("Answer is:::", answer)

