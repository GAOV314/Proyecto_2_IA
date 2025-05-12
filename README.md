# Asistente de Consulta Técnica

Aplicación web con Streamlit + LangChain/OpenAI + Pinecone para responder preguntas sobre documentos cargados por estudiantes.

## Requisitos
- Python ≥ 3.8
- Claves OpenAI y Pinecone en `.env`

## Estructura
```
├── ingest.py         # Indexa documentos en Pinecone
├── app.py            # Interfaz Streamlit
├── .env.example      # .env de ejmplo
├── requirements.txt  # Dependencias Python
├── docs/                
└──  80 registros.txt    # Plantilla de los registros
```

## Configuración
1. python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows Powershell
2. `cp .env.example .env` y completar claves.
3. `pip install -r requirements.txt`

## Flujo de trabajo
1. Indexar:        `python ingest.py`
2. Arrancar UI:    `streamlit run app.py`

## Pruebas rápidas
- **Economía**: “¿Qué estudia la economía?”
- **Energía**: “¿Qué es la energía eólica?”
- **Agronomía**: “¿Qué prácticas de conservación ecológica se usan en cultivos?”

