# migrate.py
import os
import sys
import csv
import json
from dotenv import load_dotenv
from core.database import Base, SessionLocal, Product, engine, SQLALCHEMY_SERVER_URL, MYSQL_DATABASE, create_engine, text
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def migrate_mysql():
    """MySQL DB 스키마 및 데이터 마이그레이션"""
    print("Migrating MySQL database...")
    # 데이터베이스가 없으면 생성
    temp_engine = create_engine(SQLALCHEMY_SERVER_URL)
    with temp_engine.connect() as connection:
        connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}"))
    
    # 이제 데이터베이스가 존재하므로, 테이블을 생성하고 데이터를 삽입할 수 있습니다.
    Base.metadata.create_all(bind=engine)
    
    # 데이터 삽입
    db = SessionLocal()
    
    # 테이블에 데이터가 없는지 확인하여 비어있을 때만 데이터를 삽입합니다.
    if db.query(Product).count() == 0:
        print("  - Inserting data from data.csv...")
        with open("data.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                product = Product(
                    id=row['id'], 
                    name=row['name'], 
                    description=row['description'], 
                    category=row['category']
                )
                db.add(product)
        db.commit()
    else:
        print("  - MySQL database already contains data. Skipping insertion.")
    db.close()
    print("MySQL migration complete.")

def migrate_rag():
    """ChromaDB (RAG DB) 마이그레이션"""
    #검색(Retrieval) 증강(Augmentation) 생성(Generation)
    persist_directory = "./chroma_db"
    
    if os.path.exists(persist_directory):
        print("INFO: ChromaDB already exists. Deleting and creating a new one to sync data...")
        # 기존 디렉터리를 삭제하여 새로 생성할 수 있도록 합니다.
        import shutil
        shutil.rmtree(persist_directory)
        
    print("Migrating ChromaDB...")
    ollama_emb = OllamaEmbeddings(model="nomic-embed-text")
    loader = JSONLoader(
        file_path="rag_data.json",
        jq_schema='.',
        text_content=False
    )
    docs = loader.load()
    
    db = Chroma.from_documents(docs, embedding=ollama_emb, persist_directory=persist_directory)
    print("ChromaDB migration complete.")

if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python migrate.py [db_name]")
        print("  db_name: mysql, rag, or all")
        sys.exit(1)

    command = sys.argv[1]

    if command == "mysql":
        migrate_mysql()
    elif command == "rag":
        migrate_rag()
    elif command == "all":
        migrate_mysql()
        migrate_rag()
    else:
        print(f"ERROR: Unknown command '{command}'.")
        print("Usage: python migrate.py [db_name]")
        print("  db_name: mysql, rag, or all")
        sys.exit(1)

# 두 DB 모두 갱신할 때:
# python migrate.py all


# MySQL DB만 갱신할 때:
# (예: data.csv의 내용을 변경하고 싶을 때)

# python migrate.py mysql

# RAG DB만 갱신할 때:
# (예: rag_data.json의 텍스트를 수정하거나 추가했을 때)

# python migrate.py rag

"""
1. 검색(Retrieval): 사용자의 질문과 관련된 정확한 정보를 외부 지식 베이스(벡터 데이터베이스 등)에서 찾아냅니다. 
2. 증강(Augmentation): 검색된 정보를 LLM이 이해하고 활용할 수 있는 형태로 변환하거나 보강합니다. 
3. 생성(Generation): 검색된 정보와 LLM의 원래 지식을 결합하여 사용자에게 제공할 최종 답변을 생성합니다

검색(Retrieval) 증강(Augmentation) 생성(Generation)
"""
