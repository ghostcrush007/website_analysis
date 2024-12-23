import time
import streamlit as st
from groqTest import Groq
from langchain.document_loaders import UnstructuredURLLoader

# Initialize Groq client
client = Groq(api_key="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")

# Streamlit UI setup
st.title("Query a Web Page Using Groq")

# URL input (you can enter multiple URLs separated by commas)
urls_input = st.text_area("Enter URLs (separate multiple URLs with commas)")
urls = [url.strip() for url in urls_input.split(",")]

# Load documents from the given URLs
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()

# Function to chunk the text into smaller pieces based on token limit
def chunk_text(text, max_tokens=2000):  # Reduced max_tokens to prevent excessive chunking
    """Chunk the text into smaller pieces based on token limit."""
    words = text.split()  # Split the content into words
    chunks = []
    current_chunk = []

    # Accumulate words until the chunk exceeds the token limit
    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_tokens:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]  # Start a new chunk with the current word

    if current_chunk:  # Add the last chunk
        chunks.append(" ".join(current_chunk))

    return chunks

# Combine all document content into a single string
document_content = "\n".join([doc.page_content for doc in documents])

# Chunk the document content to stay within token limits
chunks = chunk_text(document_content)

# Define a function to query the model without rate limiting (for faster execution)
def query_model(chunk, question, max_tokens=512):
    """Query the model with the given chunk and question."""
    # Make the request to the model
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Here is some information: {chunk}\n\nQuestion: {question}"}
        ],
        temperature=1,
        max_tokens=max_tokens,  # Set the max tokens for the response
        top_p=1,
        stream=True,
        stop=None,
    )

    # Accumulate the response to avoid repetition
    full_response = ""

    for chunk_response in completion:
        new_content = chunk_response.choices[0].delta.content or ""
        full_response += new_content  # Add the new content to the accumulated response

    return full_response

# Example question input
question = st.text_input("Enter your question", "can you tell me why honda is increasing its price to 2% from next january?")

# Button to trigger the query process
if st.button("Get Answer"):
    st.write("Processing...")
    all_responses = []
    for chunk in chunks:
        # Query the model for each chunk and store the response
        response = query_model(chunk, question)
        all_responses.append(response)

    # Display all responses in the Streamlit UI
    st.subheader("Answer:")
    st.write("\n".join(all_responses))
