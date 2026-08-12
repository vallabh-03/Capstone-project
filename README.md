# Zepto Capstone Project

A complete data engineering, analytics, and AI-powered support assistant project built as part of the Capstone Project.

## Project Structure

```text
Capstone-project/
│
├── data_pipeline/
│   ├── scraping/
│   ├── cleaning/
│   ├── merging/
│   ├── sql/
│   └── ...
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   └── model_comparison.csv
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── ingest.py
│   ├── rag.py
│   ├── graph.py
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── README.md
```

# Part A — Data Pipeline

The data pipeline module collects, cleans, transforms, enriches, and stores book catalog data.

### Main tasks

- Web scraping using `requests` and `BeautifulSoup`
- Data cleaning and transformation
- Data merging
- Currency conversion/enrichment
- Relational database storage
- SQL-based analysis
- Pandas-based data processing

# Part B — Analytics

The analytics module performs exploratory data analysis and machine-learning-based modeling on the processed dataset.

### Main tasks

- Exploratory Data Analysis
- Data preprocessing
- Feature preparation
- Machine learning model training
- Model comparison
- Performance evaluation

The model comparison results are stored in:

```text
analytics/model_comparison.csv
```

# Part C — Support Assistant

The support assistant is a RAG-based customer support system for answering Zepto policy questions.

## Architecture

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
Cosine Similarity
       ↓
Top-3 Relevant Documents
       ↓
LangGraph
       ↓
Structured Response
       ↓
FastAPI
```

## Technologies Used

- Python
- FastAPI
- Pydantic
- LangGraph
- ChromaDB
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Docker

## RAG Pipeline

The system uses `all-MiniLM-L6-v2` from Sentence Transformers to create embeddings for the policy documents and user queries.

ChromaDB stores the document embeddings and performs similarity-based retrieval.

The top 3 relevant policy documents are passed through the LangGraph workflow.

## Policy Documents

The knowledge base contains eight Zepto policy documents:

```text
doc_01.txt
doc_02.txt
doc_03.txt
doc_04.txt
doc_05.txt
doc_06.txt
doc_07.txt
doc_08.txt
```

These documents cover:

- Delivery
- Returns and refunds
- Membership tiers
- Order tracking
- Order cancellation
- Damaged or missing items
- Gift cards
- Customer support

## LangGraph Workflow

```text
User Query
    ↓
Intent Classification
    ↓
 ┌───────────────┐
 │               │
Policy        General
Question      Question
 │               │
 ↓               ↓
Retrieve       Direct
Top-3          Response
 │
 ↓
Structured Answer
```

Policy questions are answered using retrieved context.

General questions return:

```text
I can only answer questions about Zepto policies right now.
```

## Structured Response

```json
{
  "answer": "Answer based on the retrieved policy context.",
  "sources": [
    "doc_01",
    "doc_03"
  ],
  "confidence": 1.0
}
```

## FastAPI

The application exposes:

```text
POST /ask
```

Example request:

```json
{
  "query": "What is the delivery fee?"
}
```

Example general question:

```json
{
  "query": "What is the capital of India?"
}
```

Swagger documentation is available at:

```text
/docs
```

## Running Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r support_assistant/requirements.txt
```

Run the FastAPI application:

```bash
uvicorn support_assistant.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build the Docker image:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

The application runs on:

```text
http://localhost:7860
```

Swagger documentation:

```text
http://localhost:7860/docs
```

## Testing

### Policy Question

```json
{
  "query": "What is the delivery fee?"
}
```

The response contains an answer based on the retrieved policy context and relevant source document IDs.

### General Question

```json
{
  "query": "What is the capital of India?"
}
```

Expected behavior:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Git Workflow

The repository uses feature branches for development.

The completed work was developed through feature branches and merged into:

```text
main
```

The repository contains the complete implementation for:

- Part A — Data Pipeline
- Part B — Analytics
- Part C — Support Assistant

## Repository

GitHub:

https://github.com/vallabh-03/Capstone-project
