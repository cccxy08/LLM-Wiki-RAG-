"""API 路由"""
import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse

from schemas.schemas import (
    QueryRequest, QueryResponse, IngestResponse,
    IndexResponse, IndexEntry, WikiPageResponse,
    LintRequest, LintResponse, LintIssue,
    HealthResponse, SourceInfo,
)
from core.llm_provider import get_llm

router = APIRouter(prefix="/api")

START_TIME = time.time()

# ===== 延迟加载（避免启动时 OOM）=====

_query_service = None
_ingest_service = None
_wiki_engine = None


def _get_query_service():
    global _query_service
    if _query_service is None:
        from services.query_service import QueryService
        _query_service = QueryService()
    return _query_service


def _get_ingest_service():
    global _ingest_service
    if _ingest_service is None:
        from services.ingest_service import IngestService
        _ingest_service = IngestService()
    return _ingest_service


def _get_wiki_engine():
    global _wiki_engine
    if _wiki_engine is None:
        from core.wiki_engine import WikiEngine
        _wiki_engine = WikiEngine.get_instance()
    return _wiki_engine


# ===== 管理员仪表盘 =====

@router.get("/admin/dashboard")
def admin_dashboard():
    """管理员仪表盘：Wiki + RAG + 缓存 + 最近操作 一站式视图"""
    wiki_engine = _get_wiki_engine()
    query_service = _get_query_service()
    
    # Wiki 统计
    wiki_stats = {
        "total_pages": wiki_engine.page_count(),
        "orphan_pages": 0,
        "stale_pages": 0,
        "last_ingest": None,
    }
    # RAG 统计
    rag_stats = {
        "total_documents": query_service.rag.collection_count(),
        "last_indexed": None,
    }
    # 缓存统计
    cache_stats = query_service.get_cache_stats()
    # 最近操作
    try:
        log_path = wiki_engine.paths["log"]
        if log_path.exists():
            log_lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            recent = [l for l in log_lines[-10:] if l.startswith("## [")]
        else:
            recent = []
    except Exception:
        recent = []
    # Wiki 健康扫描
    try:
        lint_issues = wiki_engine.lint()
        wiki_stats["orphan_pages"] = sum(1 for i in lint_issues if i["type"] == "orphan")
        wiki_stats["stale_pages"] = sum(1 for i in lint_issues if i["type"] in ("stale", "deprecated_stale", "expired"))
    except Exception:
        pass
    # 最近 Ingest 时间
    for line in reversed(recent):
        if "ingest" in line:
            wiki_stats["last_ingest"] = line.split("]")[0].lstrip("## [")
            break
    for line in reversed(recent):
        if "index" in line.lower() or "ingest" in line:
            rag_stats["last_indexed"] = line.split("]")[0].lstrip("## [")
            break
    return {
        "wiki": wiki_stats,
        "rag": rag_stats,
        "cache": cache_stats,
        "recent_operations": [l.lstrip("## ").rstrip() for l in recent[-5:]],
    }


# ===== 健康检查 =====

@router.get("/health", response_model=HealthResponse)
def health():
    wiki_engine = _get_wiki_engine()
    query_service = _get_query_service()
    llm = get_llm()
    return HealthResponse(
        status="ok",
        llm_provider=llm.model_name,
        llm_model=llm.model_name,
        wiki_pages=wiki_engine.page_count(),
        vector_count=query_service.rag.collection_count(),
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ===== 文档摄入 =====

@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """上传文档，触发 Wiki Ingest + RAG 索引"""
    ingest_service = _get_ingest_service()
    try:
        content = await file.read()
        result = ingest_service.ingest_file(content, file.filename)

        return IngestResponse(
            status=result["status"],
            wiki_pages=result.get("wiki_pages", []),
            modified_pages=result.get("modified_pages", []),
            log_entry=f"[{result['status']}] 摄入 {file.filename}",
            error=result.get("wiki_error") or result.get("rag_error"),
        )
    except Exception as e:
        return IngestResponse(
            status="failed",
            error=str(e),
        )


# ===== 知识问答 =====

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """知识问答：Wiki 优先 -> RAG 兜底 + ReAct Agent 自主决策"""
    query_service = _get_query_service()
    mode = req.mode if hasattr(req, "mode") else "pipeline"
    if req.stream:
        raise HTTPException(400, "流式输出请使用 /api/query/stream")

    result = query_service.query_with_mode(req.question, mode, req.top_k)

    source_info = []
    for s in result.get("sources", []):
        source_info.append(SourceInfo(
            file=s.get("file", "unknown"),
            page=s.get("page"),
            chunk_id=s.get("chunk_id"),
        ))

    return QueryResponse(
        answer=result["answer"],
        source=result["source"],
        source_pages=result.get("source_pages", []),
        sources=source_info,
        confidence=result.get("confidence", "medium"),
        cached=result.get("cached", False),
        session_id=req.session_id,
        parsed_question=result.get("parsed_question") or "",
        pages_consulted=result.get("pages_consulted") or [],
    )


@router.post("/query/stream")
def query_stream(req: QueryRequest):
    """流式知识问答"""
    query_service = _get_query_service()
    mode = req.mode if hasattr(req, "mode") else "pipeline"
    result = query_service.query_with_mode(req.question, mode, req.top_k)
    answer = result["answer"]

    async def generate():
        for i in range(0, len(answer), 5):
            yield answer[i:i+5]
            import asyncio
            await asyncio.sleep(0.02)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# ===== Wiki 相关 =====

@router.get("/wiki/index", response_model=IndexResponse)
def wiki_index():
    """获取 Wiki 目录"""
    wiki_engine = _get_wiki_engine()
    index_text = wiki_engine.get_index()
    entries = []
    for line in index_text.split("\n"):
        line = line.strip()
        if line.startswith("- ["):
            # 简单解析
            parts = line.split(" | ")
            title_part = line[3:].split("](")[0] if "](" in line else line[3:].split("]")[0]
            entries.append(IndexEntry(
                title=title_part,
                file="",
                summary=title_part[:60],
                updated="",
            ))

    return IndexResponse(
        categories={"全部": entries},
        total_pages=len(entries),
    )


@router.get("/wiki/page/{title:path}", response_model=WikiPageResponse)
def wiki_page(title: str):
    """获取 Wiki 页面内容"""
    wiki_engine = _get_wiki_engine()
    content = wiki_engine.read_page(title)
    if content is None:
        raise HTTPException(404, f"Wiki 页面不存在: {title}")

    # 提取交叉引用 [[...]]
    import re
    cross_refs = re.findall(r"\[\[(.+?)\]\]", content)

    return WikiPageResponse(
        title=title,
        content=content,
        cross_refs=list(set(cross_refs)),
    )


# ===== Lint 健康检查 =====

@router.post("/wiki/lint", response_model=LintResponse)
def wiki_lint(req: LintRequest = LintRequest()):
    """触发 Wiki 健康检查"""
    wiki_engine = _get_wiki_engine()
    issues_raw = wiki_engine.lint()

    issues = []
    for raw in issues_raw:
        issues.append(LintIssue(
            type=raw["type"],
            pages=raw["pages"],
            description=raw["description"],
            severity=raw.get("severity", "warning"),
        ))

    return LintResponse(
        status="completed",
        issues=issues,
        scanned_pages=wiki_engine.page_count(),
    )
