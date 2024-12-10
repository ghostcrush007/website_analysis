import os
import streamlit as st
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import requests

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env (especially Hugging Face API token)

# Hugging Face Hub API token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit app setup
st.title("Ghostcrush's BOT: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# URL management
urls = st.session_state.get("urls", [])
new_url = st.sidebar.text_input("Enter URL:")
if st.sidebar.button("Add URL"):
    if new_url.strip():
        urls.append(new_url.strip())
        st.session_state["urls"] = urls

# Display the list of added URLs
if urls:
    st.sidebar.write("### Added URLs:")
    for i, url in enumerate(urls, 1):
        st.sidebar.write(f"{i}. {url}")

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hugginghubface.pkl"

main_placeholder = st.empty()

# Function to call Hugging Face API for Question Answering
def call_hf_question_answering(context, question):
    url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {"context": context, "question": question}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None
  # Load data from URLs
if process_url_clicked and urls:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Add metadata (source URL) to each document
    for doc, url in zip(data, urls):
        doc.metadata = {"source": url}  # Attach source URL to each document

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save them to FAISS index
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# When a question is asked
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

            # Retrieve relevant documents
            relevant_docs = retriever.get_relevant_documents(query)

            # Combine the contents of the relevant documents into one context
            relevant_context = ""
            sources = set()  # To store unique sources
            for doc in relevant_docs:
                relevant_context += doc.page_content  # Combine all relevant parts
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])  # Collect source metadata

            # Call the Hugging Face QA model with the combined context
            if relevant_context:
                result = call_hf_question_answering(relevant_context, query)
                if result:
                    # Display the answer
                    st.header("Answer")
                    st.write(result.get("answer", "No answer found."))

                    # Display the sources
                    st.header("Sources")
                    if sources:
                        for source in sources:
                            st.write(f"- [Source]({source})")  # Hyperlink the source URL
                    else:
                        st.write("No source information available.")
                else:
                    st.write("No relevant answer found.")
            else:
                st.write("No relevant content found.")


# if process_url_clicked and urls:
#     # Load data from URLs
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()

#     # Split the document into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitting...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)

#     # Create embeddings and save them to FAISS index
#     embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore_openai, f)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#             # Retrieve relevant documents
#             relevant_docs = retriever.get_relevant_documents(query)

#             # Combine the contents of the relevant documents into one context
#             relevant_context = ""
#             for doc in relevant_docs:
#                 relevant_context += doc.page_content  # Combine all relevant parts

#             # Call the Hugging Face QA model with the combined context
#             if relevant_context:
#                 result = call_hf_question_answering(relevant_context, query)
#                 if result:
#                     # Display the answer and sources
#                     st.header("Answer")
#                     st.write(result.get("answer", "No answer found."))

#                     st.header("Source")
#                     st.write(result.get("source_text", "No source information available."))
#                 else:
#                     st.write("No relevant answer found.")
#             else:
#                 st.write("No relevant content found.")
