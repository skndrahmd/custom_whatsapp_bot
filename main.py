import os
import openai
import sqlite3
from twilio.rest import Client
from flask import Flask, request, render_template, jsonify
import json
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key
twilio_client = Client('AC652aeef1d32f9cf28315b2558c34aa31',
                       'fe39d68cac19aae0b3f438ed22fa6670')


app = Flask(__name__)


def get_pdf_text(folder):
    text = ""
    for pdf in os.listdir(folder):
        pdf_reader = PdfReader(os.path.join(folder, pdf))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separator=" \n"
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorStore):
    llm = ChatOpenAI(temperature=0.3)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversationChain


def handleUserInput():
    prompt = input("User: ")
    response = conversation({'question': prompt})
    response_text = response.answer


text = get_pdf_text('pdfs')
chunks = get_chunks(text)
vectorStore = get_vectorstore(chunks)
conversation = get_conversation_chain(vectorStore)


@app.route('/sms', methods=['POST'])
def sms_reply():
    # Get the message body and sender's phone number
    message_body = request.form['Body']
    sender = request.form['From']

    prompt = message_body
    response = conversation({'question': prompt})
    print(response['answer'])
    response_text = response['answer']


    # Send response back through Twilio
    twilio_client.messages.create(
        body=response_text,
        from_='whatsapp:+14155238886',
        to=sender
    )

    return 'OK', 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
