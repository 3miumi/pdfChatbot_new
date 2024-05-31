import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import time
from collections import defaultdict
import numpy as np



# Set up the OpenAI client with API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Assuming you store your API key in Streamlit's secrets
client = OpenAI(api_key=OPENAI_API_KEY)

def retrieve_text_by_chunk_ids(metadata_store, indices):
    results = []
    for idx in indices:
        metadata = metadata_store[idx]
        results.append({
            "page_content": metadata["text"],  # Ensuring the expected key is present
            "doc_id": metadata["doc_id"],
            "chunk_id": metadata["chunk_id"]
        })
    return results

# Function to generate embeddings for a list of text chunks
def get_embedding(text, model="text-embedding-3-small"):
    # Normalize the text by replacing newlines with spaces
    text = text.replace("\n", " ")
    # Fetch the embedding for the text using the specified model
    response = client.embeddings.create(input=[text], model=model)
    # Extract the embedding from the response
    if isinstance(response, dict):
        embedding = response['data'][0]['embedding']
    else:
        # Assuming the response object has attribute access
        embedding = response.data[0].embedding

    return np.array(embedding, dtype=np.float32).reshape(1, -1)

# Initialize a FAISS index and a list to store text chunks

# Process documents and update index and storage
def process_files(files,  model="text-embedding-3-small"):
    dimension = 1536   # Assuming the embedding dimension based on your model
    base_index = faiss.IndexFlatL2(dimension)  # Simple flat L2 index
    metadata_store = []  # Store metadata for each chunk
    chunk_id = 0  # Global counter for chunk IDs
    index = faiss.IndexIDMap(base_index)
    for doc_id, file in enumerate(files, start=1):  # Start doc_id from 1
        reader = PdfReader(file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
        chunks = splitter.split_text(text)

        for chunk in chunks:
            embedding =get_embedding(chunk, model)
            metadata_store.append({"text": chunk, "doc_id": doc_id, "chunk_id": chunk_id})
            embedding = embedding.astype(np.float32)
            id_array = np.array([chunk_id], dtype=np.int64)

            index.add_with_ids(embedding, id_array )
            chunk_id += 1  # Increment the global chunk ID

    return index, metadata_store
def search_index(index, query_embedding, k=5):
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, k)
    return indices.flatten()

# Assuming the above functions are defined, here’s how you might initialize and process documents:

# Generate a query embedding (assuming the function `generate_embeddings` can handle single chunks)

def show():

    st.title("💬 PDF Comparison Page")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False



    def clear_state():
        st.session_state.messages = []
        st.session_state.uploader_key += 1  # Increment key to reset file uploader

    def stream_response_by_word(response):
        for word in response.split():
            yield word + " "
            time.sleep(0.1)
    st.markdown("""
        <p style='color: gray;'>Here you can reset the conversation by click button below.</p>
        """, unsafe_allow_html=True)
    if st.button('Clear Conversation'):
        clear_state()
        st.session_state.file_processed = False

    st.sidebar.markdown("""
       ## File Uploader
       Below is the file uploader where you can upload your PDF file to analyze. Make sure the file is in PDF format.
       """)
    with st.sidebar.form(key='pdf_form', clear_on_submit=True):

        files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key=f"file_uploader_{st.session_state.uploader_key}")

        submit_button = st.form_submit_button("Submit")


    if submit_button:
        st.session_state.file_processed = True  # Mark that a file has been processed
        if files:
            try:
                st.session_state.vector_store, st.session_state.metadata_store = process_files(files)
                st.success("PDF processed successfully. Ready to answer questions based on the PDF content.")
            except Exception as e:
                st.error(str(e))

        else:
            st.error("Please upload a file before submitting.")

    if not st.session_state.file_processed and submit_button:
        st.error("Please upload a file before submitting.")
        st.session_state.uploader_key += 1  # Reset uploader key here as well

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check if the question can be answered from the PDF
        if 'vector_store' in st.session_state and 'metadata_store' in st.session_state:
            query_embedding = get_embedding(user_input, "text-embedding-3-small")  # Generate the query embedding
            indices = search_index(st.session_state.vector_store, query_embedding, k=5)  # Search the index
            retrieved_texts = retrieve_text_by_chunk_ids(st.session_state.metadata_store, indices)  # Retrieve texts
            with st.chat_message("assistant"):
                if retrieved_texts:
                    # Use the retrieved texts to generate a response
                    documents = defaultdict(list)
                    # for item in retrieved_texts:
                    #     documents[item['doc_id']].append(item['page_content'])

                    # Format input for the chain.run to handle each document
                    input_documents = [{'doc_id': doc_id, 'content': ' '.join(chunks)} for doc_id, chunks in
                                       documents.items()]
                    input_documents = [
                        {'page_content': doc['page_content'], 'doc_id': doc['doc_id'], 'chunk_id': doc['chunk_id']} for doc in
                        retrieved_texts]
                    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0, max_tokens=1000,
                                     model_name="gpt-4")
                    chain = load_qa_chain(llm, chain_type="stuff")  # You might customize 'chain_type'
                    response = chain.run(input_documents=input_documents, question=user_input)
                    response = stream_response_by_word(response)
                    response = st.write_stream(response)

                else:
                    st.write("No matches found or an error occurred.")
        else:
            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True,
                )
                response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    show()