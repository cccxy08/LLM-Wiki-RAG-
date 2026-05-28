"""Agent - Karpathy LLM Wiki iterative lookup pattern. 
Knowledge compound interest via structured wiki navigation.
"""
import re
import json
from typing import Optional

from core.config import settings
from core.llm_provider import get_llm
from core.prompts import load


def _safe_eval(expr: str) -> Optional[float]:
    """安全计算简单算术表达式，只允许数字和+-*/()."""
    if not expr or not re.match(r'^[\d+\-*/().\s%]+$', expr.strip()):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, (int, float)):
            return float(result)
    except Exception:
        pass
    return None


class ReActAgent:
    """
    Karpathy LLM Wiki iterative lookup agent.
    
    Modes:
    - run():          wiki BFS → RAG fallback
    - run_wiki_only(): wiki BFS only, no RAG
    - run_with_tools(): ReAct-style with wiki_search/rag_search/wiki_backlinks tools
    """

    def __init__(self):
        self.llm = get_llm()
        self.max_hops = 3

        # Lazy-loaded engine references
        self._wiki_engine = None
        self._rag_engine = None

    def _get_wiki(self):
        if self._wiki_engine is None:
            from core.wiki_engine import WikiEngine
            self._wiki_engine = WikiEngine.get_instance()
        return self._wiki_engine

    def _get_rag(self):
        if self._rag_engine is None:
            from core.rag_engine import RAGEngine
            self._rag_engine = RAGEngine.get_instance()
        return self._rag_engine

    # ==================== Main run() ====================

    def run(self, question: str) -> dict:
        """
        Execute Karpathy LLM Wiki iterative lookup.

        Returns:
        {
            "answer": str,
            "source": "wiki" | "rag",
            "source_pages": [str],
            "sources": [dict],
            "confidence": "high" | "medium" | "low",
            "parsed_question": str,         # 预处理后的查询
            "pages_consulted": [str],        # 实际读取的页面名列表
        }
        """
        # Step 1: Query preprocessing → (refined_query, parsed_question_display, intent_data, sub_queries)
        refined_query, parsed_q, intent_data, sub_queries = self._preprocess_query(question)

        base = {"parsed_question": parsed_q, "pages_consulted": []}

        # Step 2: If sub-questions exist, decompose and search each separately
        if sub_queries:
            return self._run_decomposed(question, refined_query, parsed_q, intent_data, sub_queries, base)

        # Step 3 (no decomposition): Find initial page(s) from index
        page_titles = self._find_initial_pages(refined_query, intent_data)

        if not page_titles:
            result = self._rag_fallback(refined_query, parsed_q, original_question=question)
            result.update(base)
            return result

        # Step 4: Iterative lookup (BFS, max 3 hops)
        answer, source_pages = self._iterative_lookup(refined_query, page_titles)
        base["pages_consulted"] = source_pages

        if answer and not self._is_no_answer(answer):
            # Extra check: does the wiki answer actually address the original question?
            # If not, fall through to RAG
            if not self._answer_addresses_question(answer, question):
                pass  # fall through to RAG fallback below
            else:
                result = {
                    "answer": answer,
                    "source": "wiki",
                    "source_pages": source_pages,
                    "sources": [],
                    "confidence": "high",
                }
                result.update(base)
                return result

        # Step 5: RAG fallback (wiki empty or NO_ANSWER or off-topic)
        result = self._rag_fallback(refined_query, parsed_q, original_question=question)
        result.update(base)

        # Step 6: Knowledge compound interest — save good RAG answers to wiki
        if result.get("source") == "rag" and result.get("confidence") in ("high", "medium"):
            self._maybe_persist_to_wiki(question, result)

        return result

    def _run_decomposed(self, original_question: str, refined_query: str, parsed_q: str,
                         intent_data: dict, sub_queries: list[str], base: dict) -> dict:
        """Run with query decomposition: search each sub-question separately, then synthesize.
        
        For comparison/cross-doc questions, each sub-question gets its own lookup cycle
        (wiki BFS → RAG fallback). All collected context is then merged for the final answer.
        """
        collected_pages = {}  # title → content
        all_source_pages = []
        collected_rag_context = []  # RAG context for sub-questions that fell back to RAG

        for sub_q in sub_queries:
            # Find pages for this sub-question
            page_titles = self._find_initial_pages(sub_q, intent_data)

            if page_titles:
                # Iterative lookup for this sub-question
                answer, source_pages = self._iterative_lookup(sub_q, page_titles)
                all_source_pages.extend(source_pages)

                # Collect page contents
                wiki = self._get_wiki()
                for title in source_pages:
                    if title not in collected_pages:
                        content = wiki.read_page(title)
                        if content:
                            collected_pages[title] = content
            else:
                # Wiki missed for this sub-question → RAG retrieval for this sub-question
                rag = self._get_rag()
                docs = rag.retrieve(sub_q)
                if docs:
                    rag_context = "\n\n".join(
                        f"[来源: {d['metadata'].get('source', 'unknown')}]\n{d['content'][:1500]}"
                        for d in docs[:3]
                    )
                    collected_rag_context.append({"sub_question": sub_q, "context": rag_context})

        base["pages_consulted"] = list(dict.fromkeys(all_source_pages))  # dedupe, preserve order

        # Build combined context from wiki pages + RAG results
        all_context_parts = []
        if collected_pages:
            for title, content in collected_pages.items():
                all_context_parts.append(f"## {title}\n{content[:2500]}")
        for rag_item in collected_rag_context:
            all_context_parts.append(
                f"## (向量检索: {rag_item['sub_question']})\n{rag_item['context']}"
            )

        if all_context_parts:
            # Use multi-page extractor with combined context
            merged_pages = dict(collected_pages)
            # Add RAG contexts as pseudo-pages for the extractor
            for rag_item in collected_rag_context:
                pseudo_title = f"检索: {rag_item['sub_question']}"
                merged_pages[pseudo_title] = rag_item['context']

            answer = self._extract_answer_from_multi_pages(
                original_question, merged_pages, refined_query
            )
            if answer:
                source_pages = list(collected_pages.keys()) + [
                    f"RAG:{r['sub_question']}" for r in collected_rag_context
                ]
                return {
                    "answer": answer,
                    "source": "wiki+rag" if collected_rag_context else "wiki",
                    "source_pages": source_pages,
                    "sources": [],
                    "confidence": "high",
                    **base,
                }

        # Complete RAG fallback
        result = self._rag_fallback(refined_query, parsed_q, original_question=original_question)
        result.update(base)
        return result

    def _extract_answer_from_multi_pages(self, question: str, pages: dict[str, str],
                                          short_query: str) -> str:
        """Synthesize answer from multiple wiki pages. Handles comparison and cross-doc queries."""
        # Build context with page labels
        context_parts = []
        for title, content in pages.items():
            truncated = content[:2500] + ("\n...(内容过长已截断)" if len(content) > 2500 else "")
            context_parts.append(f"### {title}\n{truncated}")
        merged_context = "\n\n".join(context_parts)

        prompt = f"""基于以下多个 Wiki 页面内容回答用户问题。

规则：
1. 综合所有页面信息，不要只看单一页面
2. 比较类问题必须列出各方的具体数据，再做比较结论
3. 数值计算必须列出算式和计算过程，不要心算
4. 如果涉及多个人的信息，分别列出每个人的相关数据
5. 允许基于页面内容做合理推断
6. 如果内容中找不到相关信息，只回复 NO_ANSWER

页面内容：
{merged_context}

用户问题：{question}

请回答："""

        try:
            result = self.llm.chat([
                {"role": "system", "content": "你是企业知识库助手。综合多页面信息回答比较类和跨文档问题。必须列出数据来源和计算过程。"},
                {"role": "user", "content": prompt}
            ], label="extract")
            result = result.strip()
            if result.upper() == "NO_ANSWER" or "NO_ANSWER" in result[:20]:
                return ""
            return result
        except Exception:
            return ""

    def run_wiki_only(self, question: str) -> dict:
        """Wiki-only BFS lookup, no RAG fallback.

        Returns:
        {
            "answer": str (empty string if not found),
            "source": "wiki",
            "source_pages": [str],
            "sources": [],
            "confidence": "high" | "low",
            "parsed_question": str,
            "pages_consulted": [str],
        }
        """
        refined_query, parsed_q, intent_data, _sub_queries = self._preprocess_query(question)
        page_titles = self._find_initial_pages(refined_query, intent_data)

        base = {"parsed_question": parsed_q, "pages_consulted": []}

        if not page_titles:
            result = {
                "answer": "",
                "source": "wiki",
                "source_pages": [],
                "sources": [],
                "confidence": "low",
            }
            result.update(base)
            return result

        answer, source_pages = self._iterative_lookup(refined_query, page_titles)
        base["pages_consulted"] = source_pages

        result = {
            "answer": answer if answer else "",
            "source": "wiki",
            "source_pages": source_pages,
            "sources": [],
            "confidence": "high" if answer else "low",
        }
        result.update(base)
        return result

    def run_with_tools(self, question: str) -> dict:
        """ReAct-style agent with wiki_search, wiki_backlinks, rag_search tools.

        LLM decides which tools to use and when to output FINAL_ANSWER.
        Returns {
            "answer": str,
            "source": "wiki" | "rag" | "agent",
            "source_pages": [str],
            "sources": [dict],
            "confidence": "high" | "medium" | "low",
        }
        """
        tools_desc = (
            "- wiki_search: 搜索 Wiki 知识库。参数: keyword（关键词）。返回匹配的 Wiki 页面内容。\n"
            "- wiki_backlinks: 查看某 Wiki 页面的引用关系。参数: page_name（页面名称）。返回引用和被引列表。\n"
            "- rag_search: 在原始文档中语义搜索。参数: query（查询语句）。返回最相关的文档摘要。"
        )

        system_msg = load("agent_tools_system.txt", tools_desc=tools_desc)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]

        actions_log = []
        used_rag = False
        source_pages = []

        for iteration in range(10):
            response = self.llm.chat(messages, label="agent_step")
            messages.append({"role": "assistant", "content": response})

            # Check for final answer
            final = self._parse_final_answer(response)
            if final:
                source = "rag" if used_rag else "wiki"
                return {
                    "answer": final,
                    "source": source,
                    "source_pages": source_pages,
                    "sources": [],
                    "confidence": "high" if source_pages else "medium",
                }

            # Parse and execute ACTION
            action_name, action_args = self._parse_action(response)
            if not action_name:
                messages.append({
                    "role": "user",
                    "content": '请使用 ACTION: <工具名>\nARGS: {"args": "参数"} 格式调用工具，或使用 FINAL_ANSWER: <答案> 输出最终答案。'
                })
                continue

            result = self._execute_tool_action(action_name, action_args)
            used_rag = used_rag or (action_name == "rag_search")
            if action_name == "wiki_search" and result and result != "未找到相关信息。":
                source_pages.append(action_args)

            actions_log.append({
                "step": iteration + 1,
                "action": action_name,
                "args": str(action_args)[:200],
            })
            messages.append({
                "role": "user",
                "content": f"工具 [{action_name}] 返回:\\n{result[:3000]}"
            })

        # Max iterations reached
        force_prompt = "已达到最大尝试次数。请基于以上所有检索信息，给出最终答案。"
        messages.append({"role": "user", "content": force_prompt})
        final_response = self.llm.chat(messages, label="answer")
        final = self._parse_final_answer(final_response) or final_response

        source = "rag" if used_rag else ("wiki" if source_pages else "agent")
        return {
            "answer": final,
            "source": source,
            "source_pages": source_pages,
            "sources": [],
            "confidence": "low",
        }

    # ==================== ReAct helpers ====================

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """Parse FINAL_ANSWER from agent response."""
        match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_action(self, text: str) -> tuple[Optional[str], str]:
        """Parse ACTION/ARGS from agent response."""
        action_match = re.search(r"ACTION:\s*(\w+)", text)
        if not action_match:
            return None, ""
        action_name = action_match.group(1)

        args_match = re.search(r"ARGS:\s*(\{[^}]+\})", text, re.DOTALL)
        if args_match:
            try:
                parsed = json.loads(args_match.group(1))
                return action_name, parsed.get("args", args_match.group(1))
            except json.JSONDecodeError:
                return action_name, args_match.group(1)

        args_match = re.search(r"ARGS:\s*(.+?)(?:\n|ACTION:|FINAL_ANSWER:)", text, re.DOTALL)
        if args_match:
            return action_name, args_match.group(1).strip()
        return action_name, ""

    def _execute_tool_action(self, action_name: str, args: str) -> str:
        """Execute a ReAct tool action."""
        try:
            if action_name == "wiki_search":
                wiki = self._get_wiki()
                result = wiki.query(args, top_k=3)
                if result.get("hit"):
                    return result["answer"]
                return "未找到相关信息。"
            elif action_name == "wiki_backlinks":
                wiki = self._get_wiki()
                bl = wiki.get_backlinks(args)
                parts = [f"## {args} 的引用关系"]
                if bl.get("incoming"):
                    parts.append("被以下页面引用：" + ", ".join(bl["incoming"]))
                if bl.get("outgoing"):
                    parts.append("引用了以下页面：" + ", ".join(bl["outgoing"]))
                if not bl.get("incoming") and not bl.get("outgoing"):
                    parts.append("无引用关系")
                return "\n".join(parts)
            elif action_name == "rag_search":
                rag = self._get_rag()
                docs = rag.retrieve(args, top_k=5)
                if not docs:
                    return "未检索到相关文档。"
                return rag.answer(args, docs)
            else:
                return f"未知工具: {action_name}"
        except Exception as e:
            return f"工具执行失败: {e}"

    # ==================== Helpers ====================

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

    def _is_no_answer(self, answer: str) -> bool:
        """Check if the LLM's answer is essentially a 'not found' response."""
        if not answer or not answer.strip():
            return True
        stripped = answer.strip()
        if len(stripped) < 10:
            return True
        for pattern in self._NO_ANSWER_PATTERNS:
            if re.search(pattern, stripped):
                return True
        return False

    def _answer_addresses_question(self, answer: str, question: str) -> bool:
        """Check if the wiki answer actually addresses the original question.
        
        Catches cases like: question='AIGC事业群的负责人是谁' but answer='AIGC是人工智能生成内容...'
        Uses lightweight heuristic: if the question asks 'who/谁' and the answer
        contains no person names, it's likely off-topic.
        """
        import re as _re
        # Detect question type by keywords
        asks_who = bool(_re.search(r'谁|负责人|主管|领导|经理|担任', question))
        asks_when = bool(_re.search(r'什么时候|何时|哪年|日期|时间', question))
        asks_howmuch = bool(_re.search(r'多少|几|金额|数额|费用|预算|经费|收入|营收', question))

        if asks_who:
            # Answer should contain a person name (Chinese: 2-3 char name pattern)
            # Strip HTML tags first
            clean = _re.sub(r'<[^>]+>', '', answer)
            has_name = bool(_re.search(r'[\u4e00-\u9fff]{2,4}(?=（|\(|，|是|为|的|兼|担任|负责)', clean))
            if not has_name:
                return False

        if asks_when:
            clean = _re.sub(r'<[^>]+>', '', answer)
            has_date = bool(_re.search(r'\d{4}年|\d{4}-\d{2}-\d{2}|\d{4}\.\d{1,2}\.\d{1,2}', clean))
            if not has_date:
                return False

        if asks_howmuch:
            clean = _re.sub(r'<[^>]+>', '', answer)
            has_number = bool(_re.search(r'\d+[.\d]*万|\d+[.\d]*亿|\d+[.\d]*元|\d+[.\d]*%', clean))
            if not has_number:
                return False

        return True

    def _preprocess_query(self, question: str) -> tuple[str, str, dict | None, list[str]]:
        """Preprocess query: one LLM call outputs refined query, intent JSON, and sub-questions.
        Returns (refined_query, parsed_question_display, intent_data, sub_queries).
        sub_queries = list of decomposed sub-questions (empty for simple queries).
        """
        clean = question.strip()

        prompt = load("agent_preprocess.txt", question=clean)
        try:
            result = self.llm.chat([
                {"role": "system", "content": "你只按格式返回三部分内容，用 ===INTENT=== 和 ===SUBS=== 分隔，不返回其他内容。\ntarget 字段必须用最简洁的词概括用户想找的信息类型（如\"姓名\"\"日期\"\"金额\"\"人数\"\"职位\"），不要从问题中直接复制词语，也不要推测问题中没有的信息。"},
                {"role": "user", "content": prompt}
            ], label="preprocess")
        except Exception:
            return clean, clean, None, []

        # Split on delimiters
        # Format: part1 ===INTENT=== part2 ===SUBS=== part3
        subs_parts = result.split("===SUBS===", 1)
        intent_and_query = subs_parts[0]
        subs_str = subs_parts[1].strip() if len(subs_parts) > 1 else "[]"

        parts = intent_and_query.split("===INTENT===", 1)

        # Part 1: simplified query (strip prefix labels)
        short_part = parts[0].strip() if parts else ""
        short_query = short_part
        for prefix in ["精简查询：", "精简查询:", "核心查询：", "核心查询:"]:
            if short_query.startswith(prefix):
                short_query = short_query[len(prefix):].strip()
                break
        if not short_query:
            short_query = clean

        # Part 2: intent JSON
        intent_data = None
        if len(parts) > 1:
            json_part = parts[1].strip()
            if json_part.startswith("```"):
                json_part = re.sub(r"^```(?:json)?\s*", "", json_part)
                json_part = re.sub(r"\s*```$", "", json_part)
            try:
                parsed = json.loads(json_part)
                intent_data = parsed
            except (json.JSONDecodeError, Exception):
                parsed = {"intent": "fact_query", "entities": [], "target": ""}
        else:
            parsed = {"intent": "fact_query", "entities": [], "target": ""}

        # Part 3: sub-questions
        sub_queries = []
        try:
            subs_clean = subs_str.strip()
            if subs_clean.startswith("```"):
                subs_clean = re.sub(r"^```(?:json)?\s*", "", subs_clean)
                subs_clean = re.sub(r"\s*```$", "", subs_clean)
            subs_parsed = json.loads(subs_clean)
            if isinstance(subs_parsed, list):
                sub_queries = [s.strip() for s in subs_parsed if isinstance(s, str) and s.strip()]
        except (json.JSONDecodeError, Exception):
            pass

        refined_query = short_query
        parsed_question = self._format_intent(parsed, refined_query)

        return refined_query, parsed_question, intent_data, sub_queries

    def _format_intent(self, parsed: dict, refined_query: str) -> str:
        """Format intent analysis JSON as a readable display string."""
        intent_map = {
            "comparison": "比较",
            "fact_query": "事实查询",
            "fuzzy_search": "模糊搜索",
            "statistics": "统计",
        }
        intent_label = intent_map.get(parsed.get("intent", ""), parsed.get("intent", "事实查询"))
        entities = parsed.get("entities", [])
        target = parsed.get("target", "")

        entities_str = ", ".join(entities) if entities else "无"

        if entities and target:
            if parsed.get("intent") == "comparison" and len(entities) >= 2:
                summary = f"比较{entities[0]}与{entities[1]}的{target}"
            else:
                summary = f"{entities_str}的{target}"
        else:
            summary = refined_query

        return f"意图: {intent_label} | 实体: {entities_str} → {summary}"

    def _find_initial_pages(self, short_query: str, intent_data: dict = None) -> list[str]:
        """Read wiki index and let LLM select 1-3 most relevant page titles.
        If intent_data is provided, appends intent context to selection prompt.
        """
        wiki = self._get_wiki()
        try:
            index_content = wiki.get_index()
        except Exception:
            return []

        if not index_content or index_content.strip() == "":
            return []

        # Build a compact page list with tags
        all_paths = wiki._list_pages()
        if not all_paths:
            return []

        page_lines = []
        for fp in sorted(all_paths, key=lambda f: f.stem):
            name = fp.stem
            tags = wiki.get_page_tags(name)
            tag_str = f" (tags: {', '.join(tags)})" if tags else ""
            page_lines.append(f"- {name}{tag_str}")
        page_list = "\n".join(page_lines)

        if intent_data and intent_data.get('intent') == 'comparison':
            comparison_suffix = "，确保覆盖比较的双方"
            comparison_hint = "如果是比较类问题，请确保同时选到参与比较的两个实体的相关页面。"
            max_pages = "2-4"
        else:
            comparison_suffix = ""
            comparison_hint = ""
            max_pages = "1-3"

        intent_context = ""
        if intent_data:
            intent_label = intent_data.get("intent", "")
            target_info = intent_data.get("target", "")
            intent_context = f"\n意图类型：{intent_label}，目标信息：{target_info}。请优先选择包含这些信息的页面。"

        prompt = load("agent_find_pages.txt",
            max_pages=max_pages,
            comparison_suffix=comparison_suffix,
            intent_context=intent_context,
            page_list=page_list,
            short_query=short_query,
            comparison_hint=comparison_hint,
        )
        try:
            raw = self.llm.chat([
                {"role": "system", "content": "你只返回相关页面名称，每行一个，最多 3 个。不要其他内容。"},
                {"role": "user", "content": prompt}
            ], label="find_pages")
            titles = []
            all_names = {fp.stem for fp in all_paths}
            for line in raw.strip().split("\n"):
                t = line.strip().lstrip("- *0123456789. #（）()")
                if t and t in all_names:
                    titles.append(t)
            return titles[:3]
        except Exception:
            return []

    def _iterative_lookup(self, short_query: str, page_titles: list[str]) -> tuple[str, list[str]]:
        """BFS through wiki pages, expanding via [[links]]. Max 3 hops.
        Returns (answer, source_pages) or ("", []).
        """
        wiki = self._get_wiki()
        visited = set()
        queue = list(page_titles)  # BFS queue
        hops = 0
        source_pages = []

        while queue and hops < self.max_hops:
            page = queue.pop(0)
            if page in visited:
                continue
            visited.add(page)
            hops += 1

            content = wiki.read_page(page)
            if not content:
                continue

            # Ask LLM: does this page contain the answer?
            answer = self._extract_answer_from_page(page, content, short_query)
            if answer:
                source_pages.append(page)
                return answer, source_pages

            source_pages.append(page)

            # If no answer, extract [[links]] for expansion
            links = self._extract_links(content)
            for link in links:
                if link not in visited and link not in queue:
                    queue.append(link)

        return "", source_pages

    def _extract_answer_from_page(self, page_title: str, content: str, short_query: str) -> str:
        """Ask LLM: does this page contain the answer to short_query? Extract if yes."""
        # Truncate long content for LLM context
        if len(content) > 3000:
            content = content[:3000] + "\n...(内容过长已截断)"

        prompt = load("agent_extract.txt", page_title=page_title, content=content, short_query=short_query)
        try:
            result = self.llm.chat([
                {"role": "system", "content": "你是企业知识库助手，只依据提供的 Wiki 内容回答。如果内容中找不到相关信息，只回复 NO_ANSWER。\n重要规则：如果回答涉及数值计算（比较大小、求差值、百分比等），必须列出计算过程和算式，不要心算。"},
                {"role": "user", "content": prompt}
            ], label="extract")
            result = result.strip()
            if result.upper() == "NO_ANSWER" or "NO_ANSWER" in result[:20]:
                return ""
            return result
        except Exception:
            return ""

    def _extract_links(self, content: str) -> list[str]:
        """Extract [[page name]] references from wiki page content."""
        # Strip code blocks first to avoid matching inside them
        clean = re.sub(r"```[\s\S]*?```", "", content)
        clean = re.sub(r"`[^`]+`", "", clean)
        links = re.findall(r"\[\[(.+?)\]\]", clean)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for link in links:
            if link not in seen:
                seen.add(link)
                result.append(link)
        return result

    def _rag_fallback(self, short_query: str, parsed_question: str = None, original_question: str = None) -> dict:
        """RAG fallback when wiki doesn't have the answer.
        
        Uses original_question (if provided) for retrieval and answering,
        because short_query may lose critical info (e.g. '负责人' stripped from 'AIGC事业群负责人').
        """
        # Use original question for retrieval — it preserves all keywords
        retrieve_query = original_question if original_question else short_query
        answer_question = original_question if original_question else short_query

        rag = self._get_rag()
        try:
            docs = rag.retrieve(retrieve_query, top_k=5)
        except Exception:
            docs = []

        display_q = parsed_question if parsed_question else short_query
        base = {"parsed_question": display_q, "pages_consulted": []}

        if not docs:
            result = {
                "answer": "未在 Wiki 和文档库中找到相关信息。",
                "source": "rag",
                "source_pages": [],
                "sources": [],
                "confidence": "low",
            }
            result.update(base)
            return result

        try:
            answer = rag.answer(answer_question, docs)
        except Exception:
            answer = "系统繁忙，请稍后重试。"

        sources = [
            {"file": d.get("metadata", {}).get("source", "unknown"), "score": d.get("score", 0)}
            for d in docs[:3]
        ]

        result = {
            "answer": answer,
            "source": "rag",
            "source_pages": [],
            "sources": sources,
            "confidence": "medium",
        }
        result.update(base)
        return result

    def _maybe_persist_to_wiki(self, question: str, result: dict):
        """Persist high-quality RAG answers to wiki for compound interest.
        
        Only persists when:
        - Answer is from RAG (has real document sources)
        - Confidence is high or medium
        - Answer is not a fallback/low-quality response
        """
        answer = result.get("answer", "")
        if not answer or answer == "未在 Wiki 和文档库中找到相关信息。":
            return
        if self._is_no_answer(answer):
            return

        # Build a simple wiki page from Q&A
        import hashlib
        cache_key = hashlib.md5(question.encode()).hexdigest()[:8]
        source_files = [s.get("file", "unknown") for s in result.get("sources", [])]
        sources_str = ", ".join(source_files) if source_files else "RAG retrieve"

        wiki_content = f"""---
title: 自动沉淀 - {question[:30]}
type: qa_persist
sources: [{sources_str}]
---

# 问题
{question}

# 回答
{answer}
"""
        try:
            wiki = self._get_wiki()
            wiki.ingest(wiki_content, f"qa-persist-{cache_key}.md")
        except Exception:
            # Silently fail — persistence is best-effort
            pass
