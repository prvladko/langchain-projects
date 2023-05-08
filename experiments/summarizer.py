import langchain
from langchain import chains
chain = chains()

def summarize_text(text):
  """Summarizes the given text.

  Args:
    text: The text to summarize.

  Returns:
    The summary of the text.
  """

  # Create a LangChain chain.
  chain = langchain.chains()

  # Add a summarizer to the chain.
  summarizer = langchain.Summarizer()
  chain.add_component(summarizer)

  # Set the maximum length of the summary.
  summarizer.max_length = 100

  # Run the chain on the text.
  summary = chain.run(text)

  # Return the summary.
  return summary

if __name__ == "__main__":
  # Get the text to summarize.
  text = """
  The quick brown fox jumps over the lazy dog.
  The dog sees the fox and barks.
  The fox runs away.
  """

  # Summarize the text.
  summary = summarize_text(text)

  # Display the summary.
  print(summary)