# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI

# App framework
os.environ['OPENAI)API_KEY'] = apikey
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here: ')