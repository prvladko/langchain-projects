import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def scrape_elixir_school(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = []

    # Extract the main content
    main_content = soup.find('article')

    # Extract headings and paragraphs
    headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    paragraphs = main_content.find_all('p')

    # Zip together headings and paragraphs
    for heading, paragraph in zip(headings, paragraphs):
        content.append(f"{heading.text}\n{paragraph.text}")

    return "\n".join(content)

def summarize_text(text):
    summarizer = pipeline("summarization")
    input_length = len(text.split())
    
    if input_length < 30:
        return text
    
    max_length_value = max(min(150, int(input_length * 0.5)), 30)
    summary = summarizer(text, max_length=max_length_value, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def save_training_data(filename, content):
    with open(f"training/{filename}", "w") as f:
        f.write(content)

if __name__ == "__main__":
    # Create a "training" folder if it doesn't exist
    if not os.path.exists("training"):
        os.makedirs("training")

    elixir_school_url = "https://elixirschool.com/en/lessons/basics/basics" # Update the URL
    elixir_school_content = scrape_elixir_school(elixir_school_url)
    elixir_school_summary = summarize_text(elixir_school_content)
    save_training_data("elixir_school.txt", elixir_school_summary)