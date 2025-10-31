# core/database.py
import os
import csv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text
from typing import Iterator
from fastapi import Depends

# 환경 변수
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "main_db")

# SQLAlchemy 설정
SQLALCHEMY_SERVER_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}"
SQLALCHEMY_DATABASE_URL = f"{SQLALCHEMY_SERVER_URL}/{MYSQL_DATABASE}"

engine = create_engine(SQLALCHEMY_DATABASE_URL) # 데이터베이스 연결 엔진
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) # 세션 생성기
Base = declarative_base() # 모델 클래스

# ORM 모델 접근
class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    description = Column(Text)
    category = Column(String(255))

# 의존성 주입을 위한 DB 세션 함수
def get_db() -> Iterator[SessionLocal]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# DB에 의존성 주입을 하는 이유

# 1. DB 언결 로직 분리 - 코드 재사용 및 테스트 옹이성
# 2. db.close() 를 통해 세션을 무조건 닫도록 보장해 리소스 관리 용이
# 3. 요청이 들어오면 자동으로 DB를 열었다가 요청이 끝나면 DB를 자동으로 닫을 수 있게 하여
#    요청-응답 주기 동안 하나의 DB 세션만 유지시켜 세션 스코프 관리 용이