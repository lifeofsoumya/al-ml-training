from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from PyPDF2 import PdfReader
import asyncio
# import pdfplumber
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        try:
            # Create a PdfReader from the uploaded file directly
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle None case
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

# def extract_pdf(pdfs):
#     text = ""
#     for pdf in pdfs:
#         with pdfplumber.open(pdf) as pdf_reader:
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#     return text

def extract_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("embd_lcl")

async def get_conversational_chain():
    prompt_template="""
    You need to answer the question in simple terms, take in account all the details from given context. \n
    If answer is not provided in the context, reply with "Answer could not be provided from the documents".
    Do not provide made up answers. \n
    Context: \n {context} \n
    Question: \n {question} \n

    Answer:

    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(asked_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("embd_lcl", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(asked_question, k=4)

    # Run asynchronous model loading within an event loop
    chain = asyncio.run(get_conversational_chain())

    response = chain(
        {"input_documents": docs, "question": asked_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat with Documents")
    st.header("Chat with Docs")

    user_question = st.text_input("Ask a question from Documents")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your pdf files and submit", accept_multiple_files=True, type="pdf")
        if st.button("Submit to process"):
            with st.spinner("Processing"):
                raw_text = extract_pdf(pdf_docs)
                text_chunks = extract_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
            
if __name__ == "__main__":
    main()
