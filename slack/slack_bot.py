import os
# Use the package we installed
from slack_bolt import App
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initializes your app with your bot token and signing secret
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Add functionality here
# @app.event("app_home_opened") etc

# Start your app
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))