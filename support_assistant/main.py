from fastapi import FastAPI
from pydantic import BaseModel
from graph import build_graph

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto customer support assistant",
    version="1.0.0",
)

graph = build_graph()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant API is running"
    }


@app.post("/ask", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )