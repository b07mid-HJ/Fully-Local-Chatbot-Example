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
os.environ["GOOGLE_API_KEY"]="AIzaSyCxIFTj4hpP2ova-B7j8VGzAb0YPPzlIcY  "

model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.5, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
from xml.etree.ElementTree import Element, SubElement, tostring
from docx import Document

def table_to_xml(table):
    root = Element('table')
    for row in table.rows:
        row_element = SubElement(root, 'row')
        for cell in row.cells:
            cell_element = SubElement(row_element, 'cell')
            cell_element.text = cell.text.strip()  # Use cell.text directly
    return root

def get_paragraphs_before_tables(doc_path):
    doc = Document(doc_path)
    paragraphs_and_tables = []
    last_paragraph = None 
    for element in doc.element.body:
        if element.tag.endswith('p'):
            last_paragraph = element
        elif element.tag.endswith('tbl'):
            # Find the table object corresponding to this element
            for table in doc.tables:
                if table._element == element:
                    if last_paragraph is not None:
                        xml_root = table_to_xml(table)
                        xml_str = tostring(xml_root, encoding='unicode')
                        langchain_document = "Title: "+ last_paragraph.text + "Content: " + xml_str
                        paragraphs_and_tables.append(langchain_document)
                    break

    return paragraphs_and_tables

# Example usage:
docx_file_path = "./extraction_folder/repf.docx"  # Path to your .docx file
table_elements = get_paragraphs_before_tables(docx_file_path)


# Prompt
prompt_text = """You are an assistant tasked with summarizing tables.\ 
Summerize it and keep the most important information.
Also you must put the title at the beginning of the summary. \
If you encounter any table name that has Sc. that means it's a senario \
Give a summary of the table. Table chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

tables = table_elements
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})




with open("./extraction_folder/output.txt",encoding='utf-8') as f:
    state_of_the_union = f.read()

#text spliter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=5000,
    chunk_overlap=0 ,
    separators=["\n\n","\n", " ",""],
)

texts = text_splitter.create_documents([state_of_the_union])

grand_context = table_summaries + texts

print(grand_context[0])
print(grand_context[-1])

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

question_schema = ResponseSchema(
    name="question",
    description="a question about the context."
)

question_response_schemas = [
    question_schema,
]

question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
format_instructions = question_output_parser.get_format_instructions()

bare_prompt_template = "{content}"
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)

from langchain.prompts import ChatPromptTemplate

qa_template = """\
You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=grand_context[0],
    format_instructions=format_instructions
)

question_generation_chain = bare_template | model

response = question_generation_chain.invoke({"content" : messages})
output_dict = question_output_parser.parse(response.content)
