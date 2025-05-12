import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA

# 1. Cargar variables de entorno
auth_loaded = load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME")

# Validación de variables de entorno
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    st.error(
        "Define OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT y PINECONE_INDEX_NAME en .env"
    )
    st.stop()

# 2. Configurar embeddings y vectorstore desde índice existente
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    text_key="page_content"
)
# Limitar el número de documentos de contexto (solo el más relevante)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. Configurar modelo de chat y cadena QA
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 4. Interfaz Streamlit
st.set_page_config(page_title="Asistente Técnico")
st.title("🧠 Asistente de Consulta Técnica")
st.write(
    "Haz una pregunta relacionada con los documentos cargados por el equipo."
)

query = st.text_input("Escribe tu pregunta:")
if st.button("Enviar") and query:
    with st.spinner("Buscando respuesta..."):
        result = qa({"query": query})

    # Mostrar respuesta
    st.subheader("Respuesta:")
    st.write(result["result"])

    # Mostrar contexto
    st.subheader("📚 Documentos usados como contexto:")
    for doc in result["source_documents"]:
        st.markdown(
            f"**• Fuente:** `{doc.metadata.get('source', 'Desconocida')}`"
        )
        st.write(doc.page_content)
