# app.py

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Función para inicializar el modelo y el retriever
@st.cache_resource
def initialize_model_and_retriever():
    # Cargar documentos
    loader = TextLoader('product_descriptions.txt')
    documents = loader.load()

    # Dividir documentos en chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Inicializar embeddings de Ollama
    embeddings = OllamaEmbeddings()

    # Crear vector store
    db = Chroma.from_documents(texts, embeddings)

    # Crear retriever
    retriever = db.as_retriever()

    # Inicializar modelo de Ollama
    llm = Ollama(model="mistral:latest")

    # Crear cadena de RAG
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    return qa

# Inicializar el modelo y el retriever
qa = initialize_model_and_retriever()

# Configurar la interfaz de Streamlit
st.title("RAG with LangChain and Ollama")

# Campo de entrada para la pregunta
query = st.text_input("Enter your question:")

# Botón para ejecutar la consulta
if st.button("Ask"):
    if query:
        response = qa.invoke(query)
        st.write("Response:", response)
    else:
        st.write("Please enter a question.")