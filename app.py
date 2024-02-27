import os
import pickle
from typing import Any
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
from unstructured.partition.docx import partition_docx
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document as dc
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings

#model preperation
os.environ["GOOGLE_API_KEY"]="AIzaSyBR03q5DwkuBxfeCCja-b-j1hGeI0NRIGE"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__fe037fb4637146b8bdcf8355ea7c7b79"
os.environ["LANGCHAIN_PROJECT"] = "Jade-chatbot"

model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.5, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# The vectorstore to use to index the child chunks
vectorstore1 = Chroma(collection_name="table_summaries", embedding_function=OllamaEmbeddings(model="nomic-embed-text"),persist_directory="./v1")


with open('store1.pkl', 'rb') as f:
    store1 = pickle.load(f)
# The storage layer for the parent documents
id_key1 = "doc_id"

# The retriever (empty to start)
retriever1 = MultiVectorRetriever(
    vectorstore=vectorstore1,
    docstore=store1,
    id_key=id_key1,
)



# Table Retrieval


vectorstore2 = Chroma(collection_name="child_chunks", embedding_function=OllamaEmbeddings(model="nomic-embed-text"),persist_directory="./v2")

with open('store2.pkl', 'rb') as f:
    store2 = pickle.load(f)
id_key2 = "doc_id"
# The retriever (empty to start)
retriever2 = MultiVectorRetriever(
    vectorstore=vectorstore2,
    byte_store=store2,
    id_key=id_key2,
)


from langchain_core.runnables import RunnablePassthrough

# Prompt template
template = """
You are a Private-Public Partnership (PPP) feasibility expert. You are tasked with answering questions the feasibility of a PPP project.\n
Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
from langchain.retrievers import EnsembleRetriever
ensemble=EnsembleRetriever(retrievers=[retriever1,retriever2],weights=[0.5,0.5])
# RAG pipeline
chain = (
    {"context": ensemble, "question": RunnablePassthrough()}
    | prompt
    | model
    |StrOutputParser()
)


while True:
    question = input("Please enter your question (or 'quit' to stop): ")
    if question.lower() == 'quit':
        break

    print(f"Question: {question}")
    result = chain.invoke(question)
    print(f"Answer (ensemble): {ensemble.invoke(question)}")
    print("-------------------------")
    print(f"Answer (retriever): {retriever1.invoke(question)}")
    print("-------------------------")
    print(f"Answer (vectorstore2): {retriever2.invoke(question)}")
    print("-------------------------")
    print(f"Context: {result}")
    print("=========================")