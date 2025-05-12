import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone

# Cargar variables de entorno
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Inicializar Pinecone correctamente (nueva forma)
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Verificar si ya existe el índice
if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # tamaño del embedding de OpenAI
        metric="cosine"
    )

# Crear loader y cargar datos
loader = TextLoader("docs/ejemplo.txt")  # Cambia el nombre del archivo si tienes otro
documents = loader.load()

# Separar en fragmentos
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Crear embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Subir a Pinecone
vectorstore = Pinecone.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_environment=PINECONE_ENVIRONMENT,
)
