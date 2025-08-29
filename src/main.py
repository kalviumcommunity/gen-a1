import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from .rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

PROMPT_PATH = os.path.join(os.path.dirname(__file__), '../prompts/system_prompt.txt')
with open(PROMPT_PATH, 'r') as f:
    PROMPT_TEMPLATE = f.read()

class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    stop: list = None

@app.post('/ask')
def ask(request: QueryRequest):
    response = rag.run(
        user_query=request.query,
        prompt_template=PROMPT_TEMPLATE,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop
    )
    return {"response": response}

@app.get('/')
def root():
    return {"message": "AI Sports Coach API is running."}
