import os
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings



#model preperation
os.environ["GOOGLE_API_KEY"]="AIzaSyBCFZVDjQ0PhVovjrf1XcwRf5nxdOgtb1Q"
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
vectorstore1 = Chroma(collection_name="table_summaries", embedding_function=OllamaEmbeddings(model="nomic-embed-text"),persist_directory=r"C:\Users\Bohmid\Desktop\poc-01\doc1\v1")


with open(r"C:\Users\Bohmid\Desktop\poc-01\doc1\store1.pkl", 'rb') as f:
    store1 = pickle.load(f)
# The storage layer for the parent documents
id_key1 = "doc_id"

# The retriever (empty to start)
retriever1 = vectorstore1.as_retriever()

vectorstore2 = Chroma(collection_name="child_chunks", embedding_function=OllamaEmbeddings(model="nomic-embed-text"),persist_directory=r"C:\Users\Bohmid\Desktop\poc-01\doc1\v2")

with open(r"C:\Users\Bohmid\Desktop\poc-01\doc1\store2.pkl", 'rb') as f:
    store2 = pickle.load(f)
id_key2 = "doc_id"
# The retriever (empty to start)
retriever2 = MultiVectorRetriever(
    vectorstore=vectorstore2,
    byte_store=store2,
    id_key=id_key2,
)

def ensemble_invoke(question):
    from langchain.chains import LLMChain
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from pydantic import BaseModel, Field

    class LineList(BaseModel):
        lines: list[str] = Field(description="Lines of text")


    class LineListOutputParser(PydanticOutputParser):
        def __init__(self) -> None:
            super().__init__(pydantic_object=LineList)

        def parse(self, text: str) -> list[str]:
            lines = text.strip().split("\n")
            return lines


    output_parser = LineListOutputParser()

    qa = ("""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. Only provide the query, no numbering.
        Original question: {question}""",
    )
    llm_chain=(
    {"question": question}
    | qa
    | model
    |output_parser)

    queries = llm_chain.invoke(question)
    queries = queries.get("text")
    ensemble=EnsembleRetriever(retrievers=[retriever1,retriever2],weights=[0.5,0.5])
    docs = [ensemble.get_relevant_documents(q) for q in queries]
    unique_contents = set()
    unique_docs = []
    for sublist in docs:
        for doc in sublist:
            if doc.page_content not in unique_contents:
                unique_docs.append(doc)
                unique_contents.add(doc.page_content)
    unique_contents = list(unique_contents)
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = []
    for doc in unique_contents:
        pairs.append([question, doc])
    scores = cross_encoder.predict(pairs)
    scored_docs = zip(scores, unique_contents)
    sorted_docs = sorted(scored_docs, reverse=True)
    reranked_docs = [doc for _, doc in sorted_docs][0:5]
    from langchain_community.document_transformers import (
    LongContextReorder)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(reranked_docs)
    return reordered_docs

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
    {"context": ensemble_invoke(RunnablePassthrough()), "question": RunnablePassthrough()}
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
    print("-------------------------")
    print(f"Resp: {result}")
    print("=========================")
