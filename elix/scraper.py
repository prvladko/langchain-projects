import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def init_webdriver():
    # Get the current working directory
    current_directory = os.getcwd()

    # Set the path to the chromedriver in the root folder
    chromedriver_path = os.path.join(current_directory, 'chromedriver')

    # Initialize the webdriver with the chromedriver path
    driver = webdriver.Chrome(chromedriver_path)
    return driver

def scrape_elixir_docs(url, driver):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    content = soup.find('main', {'id': 'main'})
    return content.text

def scrape_elixir_school(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    lessons_sections = soup.find_all('section', {'class': 'lesson'})
    content = []

    for lesson_section in lessons_sections:
        lesson_title = lesson_section.find('h2').text
        lesson_content = lesson_section.find('p').text
        content.append(f"{lesson_title}\n{lesson_content}")

    return "\n".join(content)

# def scrape_medium_articles(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     articles = soup.find_all('h3')
#     content = [article.text for article in articles]
#     return "\n".join(content)

def summarize_text(text):
    summarizer = pipeline("summarization")
    max_length_value = min(150, int(len(text) * 0.5))
    summary = summarizer(text, max_length=max_length_value, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def save_training_data(filename, content):
    with open(f"training/{filename}", "w") as f:
        f.write(content)

if __name__ == "__main__":
    driver = init_webdriver()

    # Create a "training" folder if it doesn't exist
    if not os.path.exists("training"):
        os.makedirs("training")

    # Scrape and save data from each resource
    # elixir_docs_url = "https://hexdocs.pm/elixir/Kernel.html"
    elixir_school_url = "https://elixirschool.com/en/"
    #medium_url = "https://medium.com/tag/elixir"

    # elixir_docs_content = scrape_elixir_docs(elixir_docs_url, driver)
    elixir_school_content = scrape_elixir_school(elixir_school_url)
    #medium_articles_content = scrape_medium_articles(medium_url)

    # Summarize content
    # elixir_docs_summary = summarize_text(elixir_docs_content)
    elixir_school_summary = summarize_text(elixir_school_content)
    #medium_articles_summary = summarize_text(medium_articles_content)

    # Save summarized content as training data
    # save_training_data("elixir_docs.txt", elixir_docs_summary)
    save_training_data("elixir_school.txt", elixir_school_summary)
    #save_training_data("medium_articles.txt", medium_articles_summary)

    driver.quit()