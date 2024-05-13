# Streamlit application

import streamlit as st

# Import app.py file
import app

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config("Caht PDF")
st.header("Generative AI with multiple LLM modelds")

# load LLM
def load_llm():
    return app.setup_full_bedrock()

bedrock_embeddings = load_llm()

user_question = st.text_input("Ask a question from the PDF Files")

with st.sidebar:
    st.title("Create or Update Vector Store")

    if st.button("Vector Update"):
        with st.spinner("Processing..."):
            docs = app.data_ingestion()
            app.update_vector_store(docs,bedrock_embeddings)
            st.success("Done!!!")

if st.button("Claude Output"):
    with st.spinner("Processing..."):
        faiss_index = FAISS.load_local("faiss_index",
                                       bedrock_embeddings,
                                       allow_dangerous_deserialization=True)
        bedrock = app.setup_bedrock()
        llm = app.get_claude_llm(bedrock)
        response = app.get_claude_llm(llm,
                                      faiss_index,
                                      user_question)
        st.write(response)
        st.success("Done!!!")