import os
from flask import Flask
from slackeventsapi import SlackEventAdapter
from slack_sdk import WebClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

eventAdapter = SlackEventAdapter(os.environ['SLACK_SIGNING_SECRET'], "/slack/events", app)

client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])

bot = client.api_call("auth.test")['user_id']

@eventAdapter.on("message")
def onMessage(message):
    print(message)
    event = message.get('event', {})
    channel = event.get('channel')
    user = event.get('user')
    text = event.get('text')
    if bot in text:
        print("FOR ME!")


app.run(host='0.0.0.0', port=3000, debug=True)

