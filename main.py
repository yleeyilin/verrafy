import os
import streamlit as st
from apikey import HUGGINGFACEHUB_API_TOKEN
from chat import normal_chat
from langchain.llms import HuggingFaceHub
from reader import pdf_to_txt

# ENV and LLM SETUP
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
repo_id = "google/flan-t5-xxl"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)

# TITLE
st.title("Verrafy")
st.markdown("<h1 style='text-align: left; font-size:20px;'>Learn more about Verra's carbon credit methodlogies and guidelines here.</h1>", unsafe_allow_html=True)

# FILE UPLOADER + LOCAL DB 
uploaded_file = st.file_uploader('Upload PDF file', type='pdf')
if uploaded_file is not None:
    db_path = "/Users/leeyilin/Downloads/verrafy/localdb"
    documents = pdf_to_txt(uploaded_file, db_path)
    normal_chat(llm, documents)