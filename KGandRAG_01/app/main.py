from __future__ import annotations
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .engine import ROOT, get_engine
from .evaluation import compare

app=FastAPI(title='Diabetes Knowledge Graph + RAG Demo',version='1.0.0',
    description='Demo nghiên cứu, không phải phần mềm y tế hoặc công cụ chẩn đoán.')

class ChatRequest(BaseModel):
    question: str = Field(min_length=3,max_length=1000)
    mode: str = Field(default='hybrid',pattern='^(hybrid|vector|graph)$')
    top_k: int = Field(default=4,ge=1,le=8)

@app.get('/api/health')
def health():
    return {'status':'ok','service':'KGandRAG_01'}

@app.get('/api/stats')
def stats():
    return get_engine().stats()

@app.post('/api/chat')
def chat(payload: ChatRequest):
    return get_engine().answer(payload.question,payload.mode,payload.top_k)

@app.get('/api/graph')
def graph(q: str = Query(default='',max_length=200), limit: int=Query(default=60,ge=1,le=100)):
    return get_engine().subgraph(q,limit)

@app.get('/api/evaluation')
def evaluation(k:int=Query(default=3,ge=1,le=8)):
    return {'results':compare(get_engine(),k),'dataset':'eval/questions.json',
            'scope':'Chỉ retrieval; chưa có kiểm định lâm sàng hoặc giám định câu trả lời bởi bác sĩ.'}

app.mount('/static', StaticFiles(directory=ROOT/'app'/'static'),name='static')

@app.get('/')
def home():
    return FileResponse(ROOT/'app'/'static'/'index.html')
