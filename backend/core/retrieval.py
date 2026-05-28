"""检索增强：Rerank 重排序"""
from typing import Optional


# NOTE: BM25Retriever 和 merge_results 已删除（死代码）。
# BM25 混合检索逻辑在 rag_engine.py 中用 jieba 分词实现，
# 此处只保留 Reranker。


class Reranker:
    """Rerank 重排序器"""

    def __init__(self, model_name_or_path: str = "./.models/BAAI/bge-reranker-v2-m3"):
        self._reranker = None
        self._model_path = model_name_or_path
        self._init_model()

    def _init_model(self):
        """初始化 Rerank 模型（支持 HF 远程或本地路径）"""
        import os
        from pathlib import Path

        # 先检查本地有没有完整模型，避免 FlagReranker 硬连 HF 卡住
        model_path = self._model_path
        found_locally = Path(model_path).exists()

        if not found_locally:
            # 可能是 HF 模型名，查本地缓存
            cache_dir = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
            required_files = ["config.json", "tokenizer_config.json", "tokenizer.json",
                              "model.safetensors"]
            for d in cache_dir.rglob("snapshots/*/config.json"):
                parent = d.parent
                if all((parent / f).exists() for f in required_files):
                    model_path = str(parent)
                    found_locally = True
                    break

        if not found_locally:
            # 模型文件不全，跳过初始化
            self._reranker = None
            return

        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
        try:
            from FlagEmbedding import FlagReranker
            self._reranker = FlagReranker(
                model_path,
                use_fp16=True, device="cpu",
            )
        except Exception:
            self._reranker = None

    @property
    def available(self) -> bool:
        return self._reranker is not None

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        """
        对检索结果重排序
        Args:
            query: 用户查询
            documents: 粗排结果
            top_k: 返回 Top-K
        Returns: 重新排序后的文档列表
        """
        if not self.available or len(documents) <= 1:
            return documents[:top_k]

        # 构造 (query, doc) 对
        pairs = [[query, doc["content"][:500]] for doc in documents]

        try:
            scores = self._reranker.compute_score(pairs, normalize=True)
            # 单个结果时 scores 是标量
            if not isinstance(scores, list):
                scores = [scores]

            # 附上分数并排序
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

            sorted_docs = sorted(
                documents, key=lambda x: x.get("rerank_score", 0), reverse=True
            )
            return sorted_docs[:top_k]

        except Exception:
            return documents[:top_k]
