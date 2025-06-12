import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    model_name = os.getenv("MODEL_NAME")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore
