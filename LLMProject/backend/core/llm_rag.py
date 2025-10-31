# core/llm_rag.py
import json
import os
from typing import TypedDict, Annotated, List, Iterator
import operator
from sqlalchemy.orm import Session
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from fastapi import Depends

# ORM 모델 및 DB 세션 가져오기
from core.database import Product, get_db

# LLM 모델 로드
llm_mistral = ChatOllama(model="mistral:latest")
llm_gemma = ChatOllama(model="gemma3:latest")

# RAG DB 초기화
def init_rag_retriever():
    persist_directory = "./chroma_db"
    
    # 벡터 DB가 이미 존재하면 로드합니다.
    if os.path.exists(persist_directory):
        print("INFO: ChromaDB already exists. Loading existing vector store...")
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")
        db = Chroma(persist_directory=persist_directory, embedding_function=ollama_emb)
        return db.as_retriever(search_kwargs={"k": 5})

    # 존재하지 않으면 에러를 발생시킵니다. migrate.py를 먼저 실행해야 합니다.
    raise FileNotFoundError("ChromaDB does not exist. Please run 'python migrate.py rag' first.")

try:
    rag_retriever = init_rag_retriever()
except FileNotFoundError as e:
    print(e)
    exit()

# LangGraph 상태 정의
class AgentState(TypedDict):
    """LangGraph 상태를 나타내는 클래스.
    모든 노드는 이 상태를 입력으로 받고 수정된 상태를 반환합니다.
    """
    input: str
    chat_history: List[dict]
    response: str
    tool_output: str
    
# 라우팅 함수 정의
def route_to_agent(state):
    """
    사용자의 입력에 따라 다음 노드를 선택하는 라우팅 함수입니다.
    사용자 입력이 "제품"과 관련된 경우 "database_agent_node"로 이동합니다.
    그 외의 경우 "rag_agent_node"로 이동합니다.
    """
    print("---라우팅 실행 중---")
    
    if "제품" in state["input"]:
        print("---데이터베이스 에이전트로 이동---")
        return "database_agent_node"
    else:
        print("---RAG 에이전트로 이동---")
        return "rag_agent_node"

def rag_agent_node(state):
    print("---RAG 에이전트 실행 중---")
    
    # 대화 기록을 포함하여 검색 쿼리 생성
    history_aware_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            아래의 대화 내용과 사용자의 마지막 질문을 바탕으로, 검색 엔진에 적합한 독립적인 검색 쿼리를 생성하세요.
            응답은 오직 검색 쿼리만 포함해야 합니다.
            """),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm_gemma, rag_retriever, history_aware_prompt)
    
    retrieved_docs = history_aware_retriever.invoke({"input": state['input'], "chat_history": state['chat_history']})
    
    # RAG 체인 구성
    qa_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            당신은 유용한 AI 비서입니다.
            사용자가 제공한 문서와 추가 정보를 바탕으로 질문에 답하세요.
            만약 문서에 답이 없다면, 사용자에게 질문하여 추가 정보를 얻으십시오.
            답변은 간결하고 명확하게 작성해 주세요.

            문서 내용:
            {context}

            관련 제품 정보:
            {related_products}
            """),
            ("placeholder", "{chat_history}"), # 대화 기록
            ("user", "{input}"), # 시용자 삽입 위치
        ]
    )
    
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_prompt_template | llm_gemma) # 검색 및 생성 단계를 하나로 통합한 체인 생성
    
    # 최종 답변을 위해 모든 정보를 체인에 전달
    response = rag_chain.invoke({
        "input": state['input'],
        "chat_history": state['chat_history'],
        "context": retrieved_docs,
        "related_products": ""
    })
    
    return {"response": response["answer"]} # 응답 및 답변 생성

def database_agent_node(state):
    print("---데이터베이스 에이전트 실행 중---")
    
    search_term_map = {
        "전자제품": "Electronics",
        "전자기기": "Electronics",
        "오디오": "Audio",
        "게임": "Gaming",
        "게이밍": "Gaming",
        "주방": "Kitchen",
        "가전제품": "Home Appliances",
        "가전": "Home Appliances",
        "의류": "Apparel",
        "옷": "Apparel",
        "피트니스": "Fitness",
        "운동": "Fitness",
        "장난감": "Toys",
        "드론": "Drones",
        "보안": "Home Security",
        "스마트홈": "Smart Home",
        "스마트 기기": "Smart Home",
        "도서": "Books & Media",
        "야외용품": "Outdoor",
        "캠핑": "Outdoor",
        "컴퓨터": "Computer Accessories",
        "소프트웨어": "Software",

        "tv": "TV",
        "텔레비전": "TV",
        "노트북": "Laptop",
        "스마트폰": "Smartphone",
        "블렌더": "Blender",
        "믹서": "Blender",
        "커피": "Coffee",
        "웹캠": "Webcam",
        "마우스": "Mouse"
    }

    db_gen = get_db() # DB 세션 확보
    db = next(db_gen)
    
    try:
        query_term = None
        for key, value in search_term_map.items():
            if key in state['input'].lower():
                query_term = value
                break
        
        products = []
        if query_term:
            if query_term in ["TV", "Laptop", "Smartphone", "Blender", "Coffee", "Webcam", "Mouse"]:
                products = db.query(Product).filter(Product.name.like(f"%{query_term}%")).all() # 부분 일치를 사용해 제품 검색
            else:
                products = db.query(Product).filter(Product.category == query_term).all()
        else:
            search_term = state['input'].split()[-1].strip()
            products = db.query(Product).filter(Product.name.like(f"%{search_term}%")).all()

        if products:
            # LLM에게 바로 리스트를 전달하여 사용자 친화적인 답변을 받습니다.
            result_list = [{"id": p.id, "name": p.name, "description": p.description, "category": p.category} for p in products]
            final_response = llm_mistral.invoke(f"데이터베이스에서 다음 정보를 찾았습니다. 사용자에게 친절하게 정리하여 보여주세요:\n{json.dumps(result_list, ensure_ascii=False, indent=2)}")
            return {"response": final_response.content}
        else:
            response_text = "데이터베이스에서 해당 제품 정보를 찾을 수 없습니다. 혹시 필요한 정보가 누락되었는지 확인하거나, 관련 담당자에게 문의해 주세요."
            return {"response": response_text}
            
    finally:
        try:
            db_gen.close()
        except StopIteration:
            pass

# 그래프 구축
workflow = StateGraph(AgentState)

workflow.add_node("rag_agent_node", rag_agent_node)
workflow.add_node("database_agent_node", database_agent_node)
workflow.add_node("start_node", lambda state: state)

workflow.set_entry_point("start_node")

workflow.add_conditional_edges(
    "start_node",
    route_to_agent,
    {
        "rag_agent_node": "rag_agent_node",
        "database_agent_node": "database_agent_node"
    }
)

workflow.add_edge("rag_agent_node", END)
workflow.add_edge("database_agent_node", END)

app_graph = workflow.compile()