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
        pickle.dump(store, f)

# Check if the "faiss.pkl" file exists before attempting to load it
if not os.path.exists("faiss.pkl"):
    print("The faiss.pkl file does not exist. Running the train() function to generate it.")
    train()

# Load the "faiss.pkl" file
with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

# Load the faiss index
store.index = faiss.read_index("trainig.index")

masterPrompt = """You are an Elixir software engineer bot, using your knowledge and experience in Elixir programming to help users solve programming problems. 
Always answer honestly and only answer questions that you know the answer to. 
Be polite, professional, and provide assistance as if you were a Senior Elixir software engineer helping a colleague.

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


## Slack Bot
#import os
from flask import Flask
from slackeventsapi import SlackEventAdapter
from slack_sdk import WebClient
#from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()

app = Flask(__name__)

history = {} # history[userid] = {[historyList]}

eventAdapter = SlackEventAdapter(os.environ['SLACK_SIGNING_SECRET'], "/slack/events", app)

client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])

bot = client.api_call("auth.test")['user_id']

@eventAdapter.on("message")
def onSlackMessage(message):
    print(message)
    event = message.get('event', {})
    channel = event.get('channel')
    user = event.get('user')
    text = event.get('text')
    ts = event.get('ts')
    if bot in text:
        print("FOR ME!")
        if history.get(user)==None:
            history[user] = []
        answer = onMessage(text, history[user])
        history[user].append(f"Human: {text}")
        history[user].append(f"Bot: {answer}")
        client.chat_postMessage(channel=channel, thread_ts=ts, text=answer)


app.run(host='0.0.0.0', port=3000, debug=True)