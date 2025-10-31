# api/endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

# 핵심 로직 가져오기
from core.llm_rag import app_graph

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    chat_history: List[dict] = []

    

@router.post("/chat")
async def chat_with_agent(chat_message: ChatMessage):
    history = [
        HumanMessage(content=msg['message']) if msg['role'] == 'user' else AIMessage(content=msg['message'])
        for msg in chat_message.chat_history
    ]
    
    inputs = {"input": chat_message.message, "chat_history": history}
    
    # LangGraph 실행
    response = app_graph.invoke(inputs)
    
    # LangGraph가 반환하는 AIMessage 객체에서 content만 추출합니다.
    # isinstance를 사용하여 안전하게 처리합니다.
    final_response = response.get("response")

    if isinstance(final_response, AIMessage):
        response_content = final_response.content
    else:
        # 이미 문자열인 경우 그대로 사용
        response_content = final_response

    return {"response": response_content}