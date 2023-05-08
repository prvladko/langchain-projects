# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI

# App framework
os.environ['OPENAI)API_KEY'] = apikey
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here: ')

# LLMs
llm = OpenAI(temperature=0.9)

# Show stuff to the screen if there's a prompt
if prompt:
    response = llm(prompt)
    st.write(response)