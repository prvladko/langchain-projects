import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt

# Load the environment variables from the .env file
load_dotenv()

def train():
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the training folder
    training_folder_path = os.path.join(current_dir, "training")

    # Get the training data
    training_data = list(Path(training_folder_path).glob("**/*.*"))

    if len(training_data) < 1:
        print("No files in the training folder",
            file=sys.stderr)
        exit()

    data = []
    for training in training_data:
        with open(training) as f:
            print(f"Add {f.name} to dataset")
            data.append(f.read())

    text_splitter = CharacterTextSplitter(chunk_size=2000, separator="\n")

    docs = []
    for sets in data:
        docs.extend(text_splitter.split_text(sets))

    store = FAISS.from_texts(docs, OpenAIEmbeddings())
    faiss.write_index(store.index, "trainig.index")
    store.index = None

    with open("faiss.pkl", "wb") as f:
        store = pickle.dump(store, f)

with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

store.index = faiss.read_index("trainig.index")

masterPrompt = """You are a legal assistant bot, you will use the embeddings data provided and always answer honestly. 
It is imperitative that DO NOT answer questions that you DO NOT know the answer to. 
Be polite, professional and answer as if you were a UK employment solicitorm giving advice to a client.

Use the following pieces of MemoryContext to answer the questions at the end. 
Also remember ConversationHistory is a list of Conversation objects.
---
ConversationHistory: {history}
---
MemoryContext: {context}
---
Human: {question}
Bot:
"""

prompt = Prompt(template=masterPrompt, input_variables=["history", "context", "question"])

llmChain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0))

def onMessage(question, history):
    docs = store.similarity_search(question)
    contexts = []
    for i, doc in enumerate(docs):
        contexts.append(f"Context {i}:\n{doc.page_content}")
    #answer = llmChain.predict(question=question, context="\n\n".join(contexts), history=history)
    answer = llmChain.predict(question=question, context="\n\n".join(contexts), history="\n".join(history))
    return answer

history = []
while True:
    question = input("Ask a question > ")
    answer = onMessage(question, history)
    print(f"Bot: {answer}")
    history.append(f"Human: {question}")
    history.append(f"Bot: {answer}")