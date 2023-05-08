# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain # SequentialChain - for multiple sets of outputs
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# App framework
os.environ['OPENAI)API_KEY'] = apikey
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here: ')

# Prompt templates
# Generate title
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write me a youtube video title about {topic}'
)
# Generate script
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template='write me a youtube video script based on this TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

# Wikipedia
wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    #response = sequential_chain({'topic': prompt})
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wiki Research'):
        st.info(wiki_research)