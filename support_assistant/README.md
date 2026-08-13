# Support Assistant

## Overview
The Support Assistant is a RAG-based Zepto policy assistant. It retrieves relevant policy documents using vector similarity and routes questions through a LangGraph workflow.

## Structure
```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── ingest.py
├── rag.py
├── graph.py
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Technologies
- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- LangGraph
- FastAPI
- Pydantic
- Docker

## RAG Workflow
```text
Policy Documents
       ↓
all-MiniLM-L6-v2
       ↓
Document Embeddings
       ↓
ChromaDB
       ↓
User Query
       ↓
Query Embedding
       ↓
Similarity Retrieval
       ↓
Top-3 Documents
       ↓
LangGraph
       ↓
Structured Response
       ↓
FastAPI
```

## Document Ingestion
`ingest.py` creates document embeddings using `all-MiniLM-L6-v2` and stores them with ChromaDB.

Run:
```bash
python support_assistant/ingest.py
```

## Retrieval
`rag.py` embeds the user's query and retrieves the top 3 relevant policy documents from ChromaDB.

## LangGraph Workflow
Policy questions use retrieval:
```text
Question → Policy Intent → Retrieve Top-3 → Answer
```

General questions use the direct-response path:
```text
Question → General Intent → Direct Response
```

The general response is:
```text
I can only answer questions about Zepto policies right now.
```

## API
The FastAPI application exposes:
```text
POST /ask
```

Example:
```json
{
  "query": "What is the delivery fee?"
}
```

## Structured Response
```json
{
  "answer": "Answer based on the retrieved policy context.",
  "sources": ["doc_01", "doc_03"],
  "confidence": 1.0
}
```

## Running Locally
Create a virtual environment:
```bash
python -m venv .venv
```

Windows activation:
```bash
.venv\Scriptsctivate
```

Install dependencies:
```bash
pip install -r support_assistant/requirements.txt
```

Ingest documents:
```bash
python support_assistant/ingest.py
```

Run the API:
```bash
uvicorn support_assistant.main:app --reload
```

Swagger:
```text
http://127.0.0.1:8000/docs
```

## Docker
Build:
```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
```

Run:
```bash
docker run -p 7860:7860 zepto-support-assistant
```

Swagger:
```text
http://localhost:7860/docs
```

## Design Decisions
`all-MiniLM-L6-v2` provides lightweight local semantic embeddings, while ChromaDB provides vector retrieval. LangGraph separates intent routing from retrieval, and FastAPI provides the HTTP interface. Docker provides a reproducible way to run the service.
