from pydantic_settings import BaseSettings
from typing import Optional, Literal
from pathlib import Path

class Settings(BaseSettings):
    llm_provider: Literal["ollama", "openai", "zhipu"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    zhipu_api_key: Optional[str] = None
    zhipu_model: str = "glm-4"
    embedding_model: str = "BAAI/bge-small-zh"
    embedding_device: str = "cpu"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "enterprise_knowledge"
    chroma_parent_collection_name: str = "enterprise_knowledge_parents"
    wiki_data_dir: str = "../wiki-data"
    wiki_raw_dir: str = "../wiki-data/raw"
    wiki_pages_dir: str = "../wiki-data/wiki"
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_strategy: dict = {}
    retrieval_top_k: int = 10
    parent_child_enabled: bool = True
    parent_chunk_size: int = 750   # Parent 大切片（上下文完整）
    child_chunk_size: int = 250   # Child 小切片（检索精度高）
    child_chunk_overlap: int = 50
    parent_chunk_overlap: int = 20
    rerank_top_k: int = 3
    similarity_threshold: float = 2.0
    bm25_weight: float = 0.3
    reranker_model_path: str = "./.models/BAAI/bge-reranker-base"
    agent_max_iterations: int = 5
    agent_min_self_score: int = 7
    max_history_rounds: int = 10
    summary_enabled: bool = True
    query_cache_ttl_seconds: int = 3600
    score_cache_ttl_seconds: int = 86400
    llm_max_retries: int = 3
    llm_retry_delay_base: float = 1.0
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_dir: str = "./logs"
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    def get_wiki_paths(self):
        return {
            "data": Path(self.wiki_data_dir),
            "raw": Path(self.wiki_raw_dir),
            "pages": Path(self.wiki_pages_dir),
            "index": Path(self.wiki_pages_dir) / "index.md",
            "log": Path(self.wiki_pages_dir) / "log.md",
        }

settings = Settings()
