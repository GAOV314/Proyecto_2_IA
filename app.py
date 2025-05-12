import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Inicializar embeddings y base vectorial
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_environment=PINECONE_ENVIRONMENT,
)

# Configurar modelo LLM
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo"  # Puedes cambiarlo si deseas otro modelo
)

# Crear cadena de pregunta-respuesta
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Interfaz con Streamlit
st.set_page_config(page_title="Asistente Técnico")
st.title("🧠 Asistente de Consulta Técnica")
st.write("Haz una pregunta relacionada con los documentos cargados por el equipo.")

# Campo de entrada
query = st.text_input("Escribe tu pregunta:")

# Procesar la pregunta
if query:
    with st.spinner("Buscando la mejor respuesta..."):
        result = qa_chain({"query": query})
        st.subheader("Respuesta:")
        st.write(result["result"])

        st.subheader("📚 Documentos usados como contexto:")
        for doc in result["source_documents"]:
            st.markdown(f"**• Fuente:** `{doc.metadata.get('source', 'Desconocida')}`")
            st.markdown(f"`{doc.page_content[:300]}...`")
