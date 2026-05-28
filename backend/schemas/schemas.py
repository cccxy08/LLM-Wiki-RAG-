"""Pydantic 数据模型"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


# ===== 请求模型 =====

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID，用于多轮对话")
    top_k: int = Field(5, description="检索返回条数", ge=1, le=20)
    stream: bool = Field(False, description="是否流式输出")
    mode: Literal["auto", "pipeline", "wiki", "rag"] = Field(
        "auto",
        description="查询模式: auto=自动, pipeline=流水线, wiki=Wiki直查, rag=RAG直搜"
    )


class IngestRequest(BaseModel):
    source_name: Optional[str] = Field(None, description="自定义文档名称")
    auto_score: bool = Field(True, description="是否自动评估质量")


class LintRequest(BaseModel):
    full_scan: bool = Field(False, description="是否全量扫描(否则只检最近变更)")


# ===== 响应模型 =====

class SourceInfo(BaseModel):
    file: str = Field(..., description="来源文件名")
    page: Optional[int] = Field(None, description="页码")
    chunk_id: Optional[str] = Field(None, description="切片 ID")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="回答内容")
    source: str = Field(..., description="答案来源: wiki / rag / agent")
    source_pages: list[str] = Field(default_factory=list, description="引用的 Wiki 页面")
    sources: list[SourceInfo] = Field(default_factory=list, description="RAG 来源详情")
    confidence: str = Field("medium", description="置信度: high / medium / low")
    cached: bool = Field(False, description="是否来自缓存")
    ingested_to_wiki: bool = Field(False, description="是否已自动沉淀到 Wiki")
    session_id: Optional[str] = Field(None)
    parsed_question: str = Field("", description="预处理后的查询（清洗/精简）")
    pages_consulted: list[str] = Field(default_factory=list, description="本次查询实际读取的页面名列表")


class IngestResponse(BaseModel):
    status: str = Field(..., description="success / partial / failed")
    wiki_pages: list[str] = Field(default_factory=list, description="??? Wiki ??")
    modified_pages: list[str] = Field(default_factory=list, description="?????? Wiki ??")
    log_entry: str = Field("", description="操作日志条目")
    error: Optional[str] = Field(None, description="错误信息")


class IndexEntry(BaseModel):
    title: str
    file: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    updated: str


class IndexResponse(BaseModel):
    categories: dict[str, list[IndexEntry]] = Field(default_factory=dict)
    total_pages: int = 0


class WikiPageResponse(BaseModel):
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)
    cross_refs: list[str] = Field(default_factory=list)


class LintIssue(BaseModel):
    type: Literal["orphan", "contradiction", "stale", "missing_crossref", "expired", "deprecated_stale", "contradiction_marker"]
    pages: list[str]
    description: str
    severity: Literal["warning", "error"] = "warning"


class LintResponse(BaseModel):
    status: str
    issues: list[LintIssue] = Field(default_factory=list)
    scanned_pages: int = 0


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    wiki_pages: int
    vector_count: int
    uptime_seconds: float
