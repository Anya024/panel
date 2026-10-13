import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentState(TypedDict):
    error_payload: dict
    catalog_metadata: Optional[str]
    suggested_fix: Optional[str]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def log_parser_node(state: AgentState):
    error = state['error_payload'].get("error_type", "Unknown")
    return {"catalog_metadata": f"Analyzing: {error} | Table: silver_records_final"}

def synthesizer_node(state: AgentState):
    prompt = f"Fix this Databricks error: {state['error_payload']}. Context: {state['catalog_metadata']}. Return ONLY PySpark code."
    res = llm.invoke(prompt)
    return {"suggested_fix": res.content}

workflow = StateGraph(AgentState)
workflow.add_node("parser", log_parser_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.set_entry_point("parser")
workflow.add_edge("parser", "synthesizer")
workflow.add_edge("synthesizer", END)
agent_brain = workflow.compile()

app = FastAPI(title="Agentic Control Plane")

class FailureHook(BaseModel):
    error_type: str
    sample_records: List[dict]

@app.get("/")
def health():
    return {"status": "Agentic Control Plane is Online on Azure"}

@app.post("/triage")
async def triage(payload: FailureHook):
    initial_state = {"error_payload": payload.dict(), "catalog_metadata": None, "suggested_fix": None}
    result = agent_brain.invoke(initial_state)
    return {"analysis": result["catalog_metadata"], "fix": result["suggested_fix"]}
