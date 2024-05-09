from werkzeug.utils import secure_filename
import os
import time
import uuid
import torch
import gradio as gr
import transformers
import chromadb
import pdf2image
import pdfminer
import base64
import chainlit as cl
from io import BytesIO
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import PyPDF2
from gradio_client import Client
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.chains import VectorDBQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import AIMessage, HumanMessage
from qdrant_client.http.models import PointStruct
from langchain.chains import ConversationalRetrievalChain
import qdrant_client
from qdrant_client.http import models as rest
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
# for model's memory on past conversations
from langchain.memory import ConversationBufferMemory
# loader fo files from firectory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.email import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# Loading the Mistral Model
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
model_config = transformers.AutoConfig.from_pretrained(
    model_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

# Building a LLM QNA chain
text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=4096,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
file_id = None
# url = "http://127.0.0.1:6333"
retrieval_chain = None


def add_text(history, text):
    # Adding user query to the chatbot and chain
    # use history with current user question
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history


def image_to_base64(bot_response):
    with open(bot_response, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')


def upload_file(files):
    # Loads files when the file upload button is clicked
    # Displays them on the File window
    # print(type(file))
    return files


def process_file(files):
    global retrieval_chain
    # Loading the splitting the document #
    doc = ""
    for file in files:
        # Convert to lower case to handle mixed case
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == "pdf":
            loader = UnstructuredPDFLoader(file)
            doc = loader.load()
            chunked_documents = split_document_object(doc)
        elif file_extension == "csv":
            loader = CSVLoader(file)
            doc = loader.load()
            chunked_documents = split_document_object(doc)
        elif file_extension == "eml":
            loader = UnstructuredEmailLoader(file, process_attachments=True)
            doc = loader.load()
            chunked_documents = split_document_object(doc)
        elif file_extension == "xlsx":
            loader = UnstructuredExcelLoader(file)
            doc = loader.load()
            chunked_documents = split_document_object(doc)
        elif file_extension == "docx":
            loader = UnstructuredWordDocumentLoader(file)
            doc = loader.load()
            chunked_documents = split_document_object(doc)
        elif file_extension == "md":
            loader = UnstructuredMarkdownLoader(file)
            doc = loader.load()
            chunked_documents = split_document_object(doc)
        elif file_extension == "m4a":
            whisper_client = Client(
                "https://whisper-large-v3-1-1-caas.caas.k-mkaas-dev-1.luxembourg-2.cloud.gcore.dev/")
            doc = whisper_client.predict(file, api_name="/predict")
            chunked_documents = split_document_text(doc)
        else:
            doc = ""
            chunked_documents = split_document_text(doc)

    db = save_doc_to_db(chunked_documents)
    return db

    # Load chunked documents into the Qdrant index


def llm_processing(db):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True)
    retriever = db.as_retriever()
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory)

    return retrieval_chain


def save_doc_to_db(chunked_documents):
    db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"),
        persist_directory="chroma_db")

    return db


def load_from_db():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(embedding_function=embedding, persist_directory="chroma_db")
    return db


def split_document_object(doc):
    """Splits document objects into chunks using a recursive character text splitter.

    Args:
        doc (object): The document object to be split.

    Returns:
        list: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(doc)
    return chunked_documents


def split_document_text(doc):
    """Splits plain text document into chunks using a character text splitter.

    Args:
        doc (str): The document text to be split.

    Returns:
        list: A list of chunked documents.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=200, chunk_overlap=10, length_function=len)
    chunked_documents = text_splitter.create_documents([doc])
    return chunked_documents


def generate_bot_response(history, query, btn):
    """Function takes the query, history and inputs from the qa chain when  
    the submit button is clicked
    to generate a response to the query"""

    if not btn:
        if "@imagine" in query:
            client = Client(
                "https://stable-diffusion-1-1-caas.caas.k-mkaas-dev-1.luxembourg-2.cloud.gcore.dev/")
            query = query.replace("@imagine", "")
            bot_response = client.predict(query, api_name="/predict")
            base64 = image_to_base64(bot_response)
            data_url = f"data:image/png;base64,{base64}"
            history += [(f"![]({data_url})", None)]
            yield history, ''
        else:
            db = load_from_db()
            # run the qa chain with files from upload
            qa_chain = llm_processing(db)
            bot_response = qa_chain({"question": query})

            # simulate streaming
            for char in bot_response['answer']:
                history[-1][-1] += char
                time.sleep(0.005)
                yield history, ''

    if btn:
        db = process_file(btn)  # run the qa chain with files from upload
        qa_chain = llm_processing(db)
        bot_response = qa_chain({"question": query})
        # simulate streaming
        for char in bot_response['answer']:
            history[-1][-1] += char
            time.sleep(0.005)
            yield history, ''


# The GRADIO Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Row(equal_height=True):
            # Chatbot interface
            chatbot = gr.Chatbot(label="Mistral-7B-instruct Gcore bot",
                                 value=[],
                                 elem_id='chatbot')
        with gr.Row(equal_height=True):
            # Uploaded PDFs window
            file_output = gr.File(label="Your Files")

            with gr.Column():
                # PDF upload button
                btn = gr.UploadButton("📁 Upload a File(s): .pdf, .eml, .csv, .xlsx, .docx, .m4a, .md",
                                      file_types=[".pdf", ".eml", ".csv",
                                                  ".xlsx", ".docx", ".m4a", ".md"],
                                      file_count="multiple")

    with gr.Column():
        with gr.Column():
            # Ask question input field
            txt = gr.Text(
                show_label=False, placeholder="Enter question, Generate image via @imagine, Upload file and enter question")

        with gr.Column():
            # button to submit question to the bot
            submit_btn = gr.Button('Ask')

    # Event handler for uploading a File
    btn.upload(fn=upload_file, inputs=[btn], outputs=[file_output])

    # Gradio EVENTS
    # Event handler for submitting text question and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_bot_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=upload_file,
        inputs=[btn],
        outputs=[file_output]
    )


# Launch Gradio app in a separate thread if running with Flask
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5050, auth=(
        "user", "password")).queue().launch(root_path="/")
