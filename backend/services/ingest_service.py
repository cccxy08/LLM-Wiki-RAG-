"""文档摄入服务 - 整合 Wiki + RAG 双引擎"""
import shutil
from pathlib import Path

from core.config import settings
from core.wiki_engine import WikiEngine
from core.rag_engine import RAGEngine


class IngestService:
    """文档摄入服务"""

    def __init__(self):
        self.wiki = WikiEngine.get_instance()
        self.rag = RAGEngine.get_instance()

    def ingest_file(self, file_content: bytes, filename: str) -> dict:
        """
        处理上传的文档：
        1. 保存到 raw/ 目录
        2. 解析文本内容
        3. Wiki: 生成 Wiki 页面 + 更新 index
        4. RAG: 切片 + 向量化 + 存入 Chroma
        """
        # 1. 保存原始文件
        raw_dir = settings.get_wiki_paths()["raw"]
        file_path = raw_dir / filename
        file_path.write_bytes(file_content)

        # 2. 尝试 RAG 索引
        rag_chunks = 0
        rag_error = None
        try:
            rag_chunks = self.rag.index_document(file_path)
        except Exception as e:
            rag_error = str(e)

        # 3. Wiki Ingest（从文件提取文本）
        wiki_pages = []
        modified_pages = []
        wiki_error = None
        try:
            texts = self.rag.load_file(file_path)
            full_text = "\n\n".join(texts)
            if full_text.strip():
                result = self.wiki.ingest(full_text, filename)
                wiki_pages = result.get("wiki_pages", [])
                modified_pages = result.get("modified_pages", [])
                if result.get("error"):
                    wiki_error = result["error"]
        except Exception as e:
            wiki_error = str(e)

        # 4. 汇总结果
        status = "success"
        if wiki_error and rag_error:
            status = "failed"
        elif wiki_error or rag_error:
            status = "partial"

        return {
            "status": status,
            "wiki_pages": wiki_pages,
            "modified_pages": modified_pages,
            "rag_chunks": rag_chunks,
            "wiki_error": wiki_error,
            "rag_error": rag_error,
        }
