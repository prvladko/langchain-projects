import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap

# --------------------------------------------------------------
# Load the HuggingFaceHub API token from the .env file
# --------------------------------------------------------------

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]