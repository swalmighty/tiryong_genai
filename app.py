# GenAI application

import json
import os
import sys
import boto3
import botocore

# use Titan embedding model to generate embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM

# data ingestion library
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# Vector Embeddings and Vector Store
from langchain_community.vectorstores import FAISS


# LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Client
def setup_bedrock():
    bedrock = boto3.client(service_name="bedrock-runtime",
                           region_name="us-east-1")
    return bedrock

# Embedding function using titan
def embedding_function(bedrock_runtime):
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                           client=bedrock_runtime)
    return bedrock_embeddings

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and Vector Store function
def update_vector_store(docs,bedrock_embeddings):
    vectorstore_faiss = FAISS.from_documents(docs,bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

#initialize LLM
def get_claude_llm(bedrock):
    llm = BedrockLLM(model_id="anthropic.claude-v2:1",
                     client=bedrock,
                     model_kwargs={'max_tokens_to_sample':300}
                     )
    return llm

# Set up bedrock for streamlit
def setup_full_bedrock():
    bedrock_runtime = setup_bedrock()
    bedrock_embeddings = embedding_function(bedrock_runtime)
    return bedrock_embeddings


prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context

Question: {question}
Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
    )

# Get response from LLM
def get_response_llm(llm,faiss_index,query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever=faiss_index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
            ),
        return_source_documents = True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

