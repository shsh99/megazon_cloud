# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router
from dotenv import load_dotenv

# 환경 변수 로드  .env파일 호출
load_dotenv()

app = FastAPI()

# CORS 미들웨어 설정 (모든 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)

# 실행할때 콘솔창
# python main.py문서 실행안됨 
# (venv) ~~~ \backend> uvicorn main:app  --reload
