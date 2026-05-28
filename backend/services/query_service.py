"""查询服务 - Wiki 优先 + RAG 兜底 + 知识复利"""
import hashlib
import time
from collections import OrderedDict
from typing import Optional

from core.config import settings
from core.wiki_engine import WikiEngine
from core.rag_engine import RAGEngine
from core.llm_provider import get_llm, FALLBACK_MESSAGE
from core.prompts import load

# 缓存容量上限，防止 OOM
_QUERY_CACHE_MAX = 200
_SCORE_CACHE_MAX = 500


class QueryService:
    """知识查询服务"""

    def __init__(self):
        self.wiki = WikiEngine.get_instance()
        self.rag = RAGEngine.get_instance()
        self.llm = get_llm()
        self._query_cache: OrderedDict[str, dict] = OrderedDict()
        self._score_cache: OrderedDict[str, int] = OrderedDict()
        self._session_history: dict[str, list[dict]] = {}  # session_id -> [{question, answer}, ...]
        self._max_history = settings.max_history_rounds
        self._cache_stats = {"query_hits": 0, "query_misses": 0, "score_hits": 0, "score_misses": 0}

    def query(self, question: str, top_k: int = 5, session_id: str = None) -> dict:
        """
        查询流程：
        1. 先查缓存
        2. 查 Wiki
        3. Wiki 命中 → 回答
        4. Wiki 未命中 → RAG 兜底
        5. RAG 回答好 → 自动存档 Wiki（知识复利）
        """
        # 1. 缓存检查
        cache_key = hashlib.md5(question.encode()).hexdigest()
        cached = self._check_cache(cache_key)
        if cached:
            cached["cached"] = True
            return cached

        # 2. Wiki 查询
        if session_id:
            session_context, active_entities = self._get_session_context(session_id)
        else:
            session_context, active_entities = "", []
        try:
            wiki_result = self.wiki.query(question, top_k)
            wiki_hit = wiki_result.get("hit", False)
        except (AttributeError, NotImplementedError):
            wiki_hit = False
            wiki_result = {}

        if wiki_hit:
            answer = wiki_result["answer"]
            result = {
                "answer": answer,
                "source": "wiki",
                "source_pages": wiki_result["sources"],
                "sources": [],
                "confidence": "high",
                "cached": False,
            }
            self._update_cache(cache_key, result)
            if session_id:
                self._save_to_session(session_id, question, result["answer"])
            return result

        # 3. RAG 兜底
        # 3.1 Query 改写
        rewritten_query = self._rewrite_query(question, active_entities)
        # 3.2 检索
        docs = self.rag.retrieve(rewritten_query, top_k)
        # 3.3 回答
        answer = self.rag.answer(question, docs)

        # 3.4 评估答案质量
        score = self._evaluate_answer(question, answer)
        confidence = "high" if score >= 8 else "medium" if score >= 5 else "low"

        # 3.5 知识复利：好答案存档
        if score >= settings.agent_min_self_score:
            try:
                self.wiki.ingest(
                    f"问题：{question}\n\n答案：{answer}",
                    f"qa-{cache_key[:8]}.md"
                )
            except (AttributeError, NotImplementedError):
                pass  # WikiEngine 尚未完整实现，跳过知识复利

        sources = [
            {"file": doc["metadata"].get("source", "unknown"), "score": doc.get("score", 0)}
            for doc in docs[:3]
        ]

        result = {
            "answer": answer,
            "source": "rag",
            "source_pages": [],
            "sources": sources,
            "confidence": confidence,
            "cached": False,
        }

        self._update_cache(cache_key, result)
        if session_id:
            self._save_to_session(session_id, question, result["answer"])
        return result

    def _rewrite_query(self, question: str, entities: list = None) -> str:
        """Query 改写：补充关键词 + 利用活跃实体解决代词指代"""
        prompt = load("rag_rewrite.txt", question=question)
        if entities:
            entity_str = "、".join(entities)
            prompt += f"\n注意：当前已知活跃实体：{entity_str}。如果问题中有代词（如他/她/它/这/那/这个/那个），请结合活跃实体将代词替换为具体名称后再提取关键词。"
        try:
            rewritten = self.llm.chat([
                {"role": "system", "content": "你只输出关键词，不输出其他内容。"},
                {"role": "user", "content": prompt}
            ], label="rewrite")
            # 合并原问题 + 改写关键词
            return f"{question} {rewritten.strip()}"
        except Exception:
            return question

    def _evaluate_answer(self, question: str, answer: str) -> int:
        """答案质量评估"""
        # 先查评分缓存
        cache_key = hashlib.md5((question + answer).encode()).hexdigest()
        if cache_key in self._score_cache:
            self._score_cache.move_to_end(cache_key)
            self._cache_stats["score_hits"] += 1
            return self._score_cache[cache_key]
        self._cache_stats["score_misses"] += 1

        prompt = load("rag_evaluate.txt", question=question, answer=answer)
        try:
            response = self.llm.chat([
                {"role": "system", "content": "你只输出一个 1-10 的数字。"},
                {"role": "user", "content": prompt}
            ], label="evaluate")
            score = int(response.strip())
            score = max(1, min(10, score))
            self._score_cache[cache_key] = score
            # LRU 淘汰
            while len(self._score_cache) > _SCORE_CACHE_MAX:
                self._score_cache.popitem(last=False)
            return score
        except Exception:
            return 5  # 默认中等分

    def _check_cache(self, key: str) -> Optional[dict]:
        """检查查询缓存（LRU）"""
        if key in self._query_cache:
            entry = self._query_cache[key]
            if time.time() - entry["_ts"] < settings.query_cache_ttl_seconds:
                # 命中：移到末尾（最近使用）
                self._query_cache.move_to_end(key)
                self._cache_stats["query_hits"] += 1
                return entry["data"]
            else:
                # 过期：删除
                del self._query_cache[key]
        self._cache_stats["query_misses"] += 1
        return None

    def _update_cache(self, key: str, data: dict):
        """更新查询缓存（LRU 淘汰，失败/降级回复不入缓存）"""
        answer = data.get("answer", "")
        # 降级回复不缓存（避免后续查询命中失败缓存）
        if answer == FALLBACK_MESSAGE:
            return
        # 明确的无结果回复也不缓存
        if answer.strip() in ("未找到相关文档，无法回答此问题。",):
            return
        # 如果已存在则更新并移到末尾
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
        self._query_cache[key] = {
            "data": data,
            "_ts": time.time(),
        }
        # LRU 淘汰：超出上限时删除最旧的
        while len(self._query_cache) > _QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)

    def agent_query(self, question: str) -> dict:
        """
        通过 ReAct Agent 执行查询（自主决策式）
        Agent 会自己决定查 Wiki 还是 RAG、是否自评、是否沉淀。
        """
        from core.agent import ReActAgent
        agent = ReActAgent()
        return agent.run(question)


    def _get_session_context(self, session_id: str) -> tuple:
        """Get recent conversation context and active entities (last 3 rounds deduped)"""
        if not session_id or session_id not in self._session_history:
            return "", []
        history = self._session_history[session_id][-self._max_history:]
        if not history:
            return "", []
        lines = ["Previous conversation:"]
        for h in history:
            lines.append(f"Q: {h['question'][:200]}")
            lines.append(f"A: {h['answer'][:200]}")
        # Active entities: deduped from last 3 rounds
        active_entities = []
        seen = set()
        for h in history[-3:]:
            for e in (h.get("entities") or []):
                if e not in seen:
                    seen.add(e)
                    active_entities.append(e)
        return "\n".join(lines), active_entities

    def _extract_entities(self, question: str, answer: str) -> list:
        """使用 LLM 从问答中提取命名实体（人名、部门名、项目名等）"""
        import json
        prompt = f"""从以下对话中提取所有命名实体（人名、部门名、项目名、产品名、公司名、职位名等）。只返回 JSON 数组，不含其他内容。没有实体则返回 []。

问题：{question[:500]}
回答：{answer[:500]}

输出示例：["张三", "技术部", "项目Alpha"]"""
        try:
            response = self.llm.chat([
                {"role": "system", "content": "你只输出 JSON 数组，不输出其他内容。"},
                {"role": "user", "content": prompt}
            ], label="extract_entities")
            entities = json.loads(response.strip())
            return entities if isinstance(entities, list) else []
        except Exception:
            return []

    def _save_to_session(self, session_id: str, question: str, answer: str):
        """Save Q&A to session history with entity extraction"""
        if not session_id:
            return
        if session_id not in self._session_history:
            self._session_history[session_id] = []
        entities = self._extract_entities(question, answer)
        self._session_history[session_id].append({
            "question": question,
            "answer": answer[:500],
            "entities": entities
        })
        # Trim to max
        if len(self._session_history[session_id]) > self._max_history:
            self._session_history[session_id] = self._session_history[session_id][-self._max_history:]


    def _rag_only_query(self, question: str, top_k: int = 5) -> dict:
        """只查 RAG，不查 Wiki"""
        docs = self.rag.retrieve(question, top_k)
        if not docs:
            return {"answer": "RAG 中未找到相关信息。", "source": "rag", "source_pages": [], "sources": [], "confidence": "low"}
        answer = self.rag.answer(question, docs)
        sources = [{"file": d.get("metadata", {}).get("source", ""), "score": d.get("score", 0)} for d in docs[:3]]
        return {"answer": answer, "source": "rag", "source_pages": [], "sources": sources, "confidence": "medium"}

    # Patterns that indicate a wiki answer is not actually useful
    _NO_ANSWER_PATTERNS = [
        "NO_ANSWER",
        "未找到",
        "未在.*找到",
        "无法确定",
        "无法比较",
        "无法回答",
        "未明确提及",
        "内容中未",
        "没有提供",
        "没有提及",
        "无法提供",
        "无法进行",
        "信息不足",
        "无法得知",
        "无法确认",
    ]

    def _is_low_quality_answer(self, answer: str) -> bool:
        """Check if a wiki answer is a non-answer (acknowledged failure to find info)."""
        if not answer or not answer.strip():
            return True
        stripped = answer.strip()
        # Very short answers are likely non-answers
        if len(stripped) < 10:
            return True
        import re
        for pattern in self._NO_ANSWER_PATTERNS:
            if re.search(pattern, stripped):
                return True
        return False

    def query_with_mode(self, question: str, mode: str = "auto", top_k: int = 5, session_id: str = None) -> dict:
        """
        统一查询入口，支持以下模式：
        - auto: Wiki BFS → 未命中/低质量 → RAG 兜底
        - wiki: 仅 Wiki 检索，未命中返回"未找到"
        - rag: 仅 RAG 语义搜索
        - pipeline: 固定流水线（Wiki→RAG）
        """
        from core.agent import ReActAgent

        if mode == "auto":
            agent = ReActAgent()
            result = agent.run(question)  # run() has built-in wiki→RAG fallback
            # Double-check: even run() may return low-quality wiki answers
            if self._is_low_quality_answer(result.get("answer", "")):
                rag_result = self._rag_only_query(question, top_k)
                if not self._is_low_quality_answer(rag_result.get("answer", "")):
                    # Merge pages_consulted from wiki attempt
                    rag_result["pages_consulted"] = result.get("pages_consulted", [])
                    rag_result["parsed_question"] = result.get("parsed_question", "")
                    return rag_result
                # Both failed — return the more informative one
                if len(rag_result.get("answer", "")) > len(result.get("answer", "")):
                    rag_result["pages_consulted"] = result.get("pages_consulted", [])
                    rag_result["parsed_question"] = result.get("parsed_question", "")
                    return rag_result
            return result

        elif mode == "wiki":
            agent = ReActAgent()
            result = agent.run_wiki_only(question)
            if result.get("answer"):
                return result
            return {
                "answer": "Wiki 中未找到相关信息。",
                "source": "wiki",
                "source_pages": [],
                "sources": [],
                "confidence": "low",
            }

        elif mode == "rag":
            return self._rag_only_query(question, top_k)

        elif mode == "pipeline":
            return self.query(question, top_k, session_id)

        else:
            # Fallback to auto
            return self.query_with_mode(question, "auto", top_k, session_id)

    def get_cache_stats(self) -> dict:
        """获取缓存命中统计"""
        total_q = self._cache_stats["query_hits"] + self._cache_stats["query_misses"]
        total_s = self._cache_stats["score_hits"] + self._cache_stats["score_misses"]
        return {
            "query_cache": {
                "hits": self._cache_stats["query_hits"],
                "misses": self._cache_stats["query_misses"],
                "hit_rate": round(self._cache_stats["query_hits"] / max(total_q, 1), 3),
                "size": len(self._query_cache),
            },
            "score_cache": {
                "hits": self._cache_stats["score_hits"],
                "misses": self._cache_stats["score_misses"],
                "hit_rate": round(self._cache_stats["score_hits"] / max(total_s, 1), 3),
                "size": len(self._score_cache),
            },
        }
