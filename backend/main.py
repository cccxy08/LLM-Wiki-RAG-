"""FastAPI 入口 - Wiki-RAG 双引擎知识问答系统"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from core.config import settings

app = FastAPI(
    title="Wiki-RAG 企业知识问答系统",
    description="基于 Karpathy LLM Wiki 范式 + RAG 双引擎的企业级知识管理方案",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """启动时检查"""
    import sys; sys.stdout.reconfigure(encoding='utf-8')
    print(f"Wiki-RAG 系统启动中...")
    print(f"   LLM Provider: {settings.llm_provider}")
    print(f"   API 文档: http://{settings.host}:{settings.port}/docs")

    # 挂载静态文件（API 路由优先，不会冲突）
    print(f"   🖥️  前端页面: http://{settings.host}:{settings.port}/")
    # 检查 Ollama 连接（如果是 Ollama 模式）
    # if settings.llm_provider == "ollama":
    #     try:
    #         import requests
    #         resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
    #         if resp.status_code == 200:
    #             print(f"   ✅ Ollama 连接正常 ({settings.ollama_base_url})")
    #         else:
    #             print(f"   ⚠️  Ollama 连接异常: HTTP {resp.status_code}")
    #     except Exception as e:
    #         print(f"   ⚠️  Ollama 连接失败: {e}")
    #         print(f"   系统将启动但 LLM 调用可能失败")


# 挂载静态文件（放在路由之后，确保 API 路由优先）
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
