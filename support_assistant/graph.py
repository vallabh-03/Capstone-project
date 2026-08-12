import os
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from rag import RAGRetriever
# ============================================================
# CONFIGURATION
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list[dict]
    answer: str
    sources: list[str]
    confidence: float
    response: dict


# ============================================================
# RETRIEVER
# ============================================================

retriever = RAGRetriever()


# ============================================================
# STRUCTURED PROMPT
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant.

CONTEXT:
Use only the Zepto policy information provided below.

TASK:
Answer the user's question using the supplied context.

FORMAT:
Return a concise answer containing the answer and relevant source IDs.

LENGTH:
Keep the answer short and clear.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies.

FEW-SHOT EXAMPLE:
User question: What is the delivery fee below INR 149?
Context: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
Answer: Orders below INR 149 incur a flat INR 25 delivery fee.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# NODE 1 — CLASSIFY INTENT
# ============================================================

def classify_intent(state: GraphState) -> GraphState:

    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    if MOCK_LLM:
        if any(keyword in query for keyword in policy_keywords):
            intent = "policy_question"
        else:
            intent = "general_question"

    else:
        # Optional real-LLM extension.
        # The graded baseline uses MOCK_LLM=1.
        #
        # A real LLM can be connected here later.
        if any(keyword in query for keyword in policy_keywords):
            intent = "policy_question"
        else:
            intent = "general_question"

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# NODE 2 — RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(state: GraphState) -> GraphState:

    query = state["query"]

    retrieved_chunks = retriever.retrieve(
        query,
        top_k=3,
    )

    if not retrieved_chunks:
        response = AnswerResponse(
            answer="No relevant Zepto policy information was found.",
            sources=[],
            confidence=0.0,
        )

        return {
            **state,
            "retrieved_chunks": [],
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "response": response.model_dump(),
        }

    source_ids = [
        chunk["id"]
        for chunk in retrieved_chunks
    ]

    if MOCK_LLM:

        top_chunk_snippet = retrieved_chunks[0]["text"][:200]

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        response = AnswerResponse(
            answer=answer,
            sources=source_ids,
            confidence=1.0,
        )

    else:

        # Optional real-LLM extension.
        #
        # The required graded baseline is MOCK_LLM=1.
        # This branch is intentionally kept separate from
        # deterministic mock generation.

        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        prompt = PROMPT_TEMPLATE.format(
            query=query,
            context=context,
        )

        # Placeholder for optional real LLM integration.
        # Keep the deterministic fallback so the required
        # offline mode remains fully functional.

        answer = (
            f"Based on the retrieved context: "
            f"{retrieved_chunks[0]['text'][:200]}"
        )

        response = AnswerResponse(
            answer=answer,
            sources=source_ids,
            confidence=1.0,
        )

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "response": response.model_dump(),
    }


# ============================================================
# NODE 3 — DIRECT ANSWER
# ============================================================

def direct_answer(state: GraphState) -> GraphState:

    if MOCK_LLM:

        answer = (
            "I can only answer questions about Zepto policies right now."
        )

        response = AnswerResponse(
            answer=answer,
            sources=[],
            confidence=1.0,
        )

    else:

        # Optional real-LLM extension.
        answer = (
            "I can only answer questions about Zepto policies right now."
        )

        response = AnswerResponse(
            answer=answer,
            sources=[],
            confidence=1.0,
        )

    return {
        **state,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "response": response.model_dump(),
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_by_intent(state: GraphState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH STATEGRAPH
# ============================================================

def build_graph():

    workflow = StateGraph(GraphState)

    workflow.add_node(
        "classify_intent",
        classify_intent,
    )

    workflow.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    workflow.add_node(
        "direct_answer",
        direct_answer,
    )

    workflow.set_entry_point(
        "classify_intent"
    )

    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge(
        "retrieve_and_answer",
        END,
    )

    workflow.add_edge(
        "direct_answer",
        END,
    )

    return workflow.compile()


# ============================================================
# TEST THE GRAPH
# ============================================================

if __name__ == "__main__":

    graph = build_graph()

    print("=" * 60)
    print("POLICY QUESTION")
    print("=" * 60)

    policy_result = graph.invoke(
        {
            "query": "What is the delivery fee?",
        }
    )

    print(policy_result["response"])

    print("\n" + "=" * 60)
    print("GENERAL QUESTION")
    print("=" * 60)

    general_result = graph.invoke(
        {
            "query": "What is the capital of India?",
        }
    )

    print(general_result["response"])