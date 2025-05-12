# ingest.py
import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LC_Pinecone
from langchain.schema import Document

# --- 1. Cargar variables de entorno ---
load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME")   

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    raise RuntimeError(
        "Define OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT y PINECONE_INDEX_NAME en tu .env"
    )

# --- 2. Inicializar cliente Pinecone nativo ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- 3. Crear el índice en modo serverless si no existe ---
existing = [idx.name for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing:
    spec = ServerlessSpec(
        cloud="aws",        
        region=PINECONE_ENVIRONMENT
    )
    pc.create_index(
        name = PINECONE_INDEX_NAME,
        dimension = 1536,   
        metric = "cosine",
        spec = spec
    )
    print(f"Índice '{PINECONE_INDEX_NAME}' creado en modo serverless.")
else:
    print(f"Índice '{PINECONE_INDEX_NAME}' ya existe.")

# --- 4. Obtener instancia de índice ---
native_index = pc.Index(PINECONE_INDEX_NAME)

# --- 5. Cargar y fragmentar documentos ---
file_paths = glob.glob("docs/*.txt")
if not file_paths:
    raise RuntimeError("No se encontraron archivos .txt en la carpeta docs/")


splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
all_chunks = []

for path in file_paths:
    filename = os.path.basename(path)
    print(f"Procesando: {filename}")

    if filename == "80 registros.txt":
        
        with open(path, "r", encoding="utf8") as f:
            for i, line in enumerate(f, start=1):
                text = line.strip()
                if text:
                    all_chunks.append(Document(
                        page_content=text,
                        metadata={"source": filename, "line": i}
                    ))
    else:
        
        docs = TextLoader(path, encoding="utf8").load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

print(f"Total de fragmentos a indexar: {len(all_chunks)}")

# --- 6. Generar embeddings y subir documentos ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = LC_Pinecone(
    native_index,
    embeddings,
    text_key="page_content"
)
vectorstore.add_documents(all_chunks)

print(f"Ingestión completada: {len(all_chunks)} fragmentos indexados en '{PINECONE_INDEX_NAME}'.")
