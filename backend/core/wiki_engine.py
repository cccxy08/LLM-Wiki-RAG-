"""Wiki engine - structured knowledge management (LLM Wiki pattern)
Based on IMPACT-MAP design spec and Karpathy LLM Wiki pattern.
"""
import os
import re
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from core.prompts import load

from core.config import settings
from core.llm_provider import get_llm, LLMProvider


class WikiEngine:
    """Wiki engine core"""

    _instance: Optional["WikiEngine"] = None

    @classmethod
    def get_instance(cls) -> "WikiEngine":
        """获取单例（避免多次加载模型占满内存）"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.paths = settings.get_wiki_paths()
        self.llm = get_llm()
        self._ensure_dirs()
        self._ensure_base_files()
        self._page_cache = {}

    # ==================== Dirs & Files ====================

    def _ensure_dirs(self):
        for key in ["raw", "pages"]:
            self.paths[key].mkdir(parents=True, exist_ok=True)
        (self.paths["raw"] / "assets").mkdir(exist_ok=True)

    def _ensure_base_files(self):
        if not self.paths["index"].exists():
            self.paths["index"].write_text("# Wiki index\n\n> Auto-maintained\n\n", encoding="utf-8")
        if not self.paths["log"].exists():
            self.paths["log"].write_text("# Operation log\n\n> Chronological\n\n", encoding="utf-8")
        # backlinks.json
        bl_path = self.paths["data"] / "backlinks.json"
        if not bl_path.exists():
            bl_path.write_text("{}", encoding="utf-8")
        # tags.json
        tp = self.paths["data"] / "tags.json"
        if not tp.exists():
            tp.write_text("[]", encoding="utf-8")

    @property
    def _backlinks_path(self) -> Path:
        return self.paths["data"] / "backlinks.json"

    @property
    def _tags_path(self) -> Path:
        return self.paths["data"] / "tags.json"

    def _read_tags(self) -> list[str]:
        """Read tags.json, return list of all known tags."""
        try:
            return json.loads(self._tags_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _extract_tags_from_page(self, content: str) -> list[str]:
        """Extract tags list from page frontmatter."""
        m = re.search(r"^tags:\s*\[(.+?)\]", content, re.MULTILINE)
        if m:
            return [t.strip().strip('"\'') for t in m.group(1).split(",") if t.strip()]
        m = re.search(r"^tags:\s*(.+)$", content, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if val and val != "[]":
                return [t.strip() for t in val.split(",") if t.strip()]
        return []

    def _sync_tags_from_page(self, content: str):
        """Extract tags from page frontmatter and merge into tags.json."""
        page_tags = self._extract_tags_from_page(content)
        if not page_tags:
            return
        existing = self._read_tags()
        merged = list(set(existing + page_tags))
        if sorted(merged) != sorted(existing):
            self._tags_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    def _format_tags_context(self) -> str:
        """Build tags context string for LLM prompts."""
        tags = self._read_tags()
        if not tags:
            return ""
        return f"已有页面使用的 tags：{json.dumps(tags, ensure_ascii=False)}。优先使用已有 tags，含义不匹配时可以创建新 tag。"

    def get_page_tags(self, page_title: str) -> list[str]:
        """Public: get tags for a specific page. Reads directly from the page file."""
        content = self.read_page(page_title)
        if not content:
            return []
        return self._extract_tags_from_page(content)

    def _get_schema(self) -> str:
        sp = self.paths["data"] / "WIKI-SCHEMA.md"
        return sp.read_text(encoding="utf-8") if sp.exists() else ""

    # ==================== Core: Query (LLM-driven page selection) ====================

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Karpathy LLM Wiki query pattern per IMPACT-MAP:
        1. Read index.md to find relevant pages (_find_related_pages via LLM)
        2. Read those pages
        3. LLM synthesizes answer (_answer_from_wiki)

        Returns: {"hit": bool, "answer": str, "sources": [str]}
        """
        all_pages = self._list_pages()
        if not all_pages:
            return {"hit": False, "answer": "", "sources": []}

        # Step 1: LLM selects relevant pages from index
        selected = self._find_related_pages(question, top_k)
        if not selected:
            return {"hit": False, "answer": "", "sources": []}

        # Step 2: Read selected pages
        pages_content = {}
        sources = []
        for title in selected:
            content = self.read_page(title)
            if content:
                pages_content[title] = content
                sources.append(title)

        if not pages_content:
            return {"hit": False, "answer": "", "sources": sources}

        # Step 3: LLM synthesizes answer
        answer = self._answer_from_wiki(question, pages_content)
        return {"hit": True, "answer": answer, "sources": sources}

    def _find_related_pages(self, question: str, top_k: int = 5) -> list[str]:
        """LLM reads the index and selects relevant page titles."""
        index_content = self.get_index()

        # Build a compact page list with first-line summaries
        page_list_lines = []
        for fp in sorted(self._list_pages(), key=lambda f: f.stem):
            try:
                first_line = fp.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ")[:100]
            except Exception:
                first_line = ""
            page_list_lines.append(f"- {fp.stem}: {first_line}")
        page_list = "\n".join(page_list_lines)

        prompt = load("wiki_find_pages.txt", top_k=top_k, page_list=page_list, question=question)

        try:
            raw = self.llm.chat([
                {"role": "system", "content": "你只返回相关页面名称，每行一个，不要其他内容。"},
                {"role": "user", "content": prompt}
            ], label="find_pages")
            titles = []
            for line in raw.strip().split("\n"):
                t = line.strip().lstrip("- *0123456789. #（）()")
                if t:
                    titles.append(t)
            return titles[:top_k]
        except Exception:
            # Fallback: return all page names
            return [fp.stem for fp in self._list_pages()[:top_k]]

    def _answer_from_wiki(self, question: str, pages: dict[str, str]) -> str:
        """LLM synthesizes an answer from wiki page contents."""
        context_parts = []
        for title, content in pages.items():
            if len(content) > 3000:
                content = content[:3000] + "\n...(内容过长已截断)"
            context_parts.append(f"## {title}\n{content}")

        prompt = load("wiki_answer.txt", context="\n\n---\n\n".join(context_parts), question=question)

        try:
            return self.llm.chat([
                {"role": "system", "content": (
                    "你是企业知识库助手，只依据提供的 Wiki 内容回答。不确定就说不知道。\n\n"
                    "语义理解规则：用户可能用不同的词描述同一件事（如「负责人」=「主管」、"
                    "「做支付」=「参与支付系统」）。请结合上下文理解用户意图，不要因为用词"
                    "不完全一致就认为不匹配。"
                )},
                {"role": "user", "content": prompt}
            ], label="answer")
        except Exception:
            return "系统繁忙，请稍后重试。"

    # ==================== Core: Ingest (LLM generates wiki pages) ====================

    def ingest(self, content, source_name):
        """Ingest raw content into wiki. Per IMPACT-MAP:
        1. LLM generates wiki page(s) (_generate_wiki_page or _generate_wiki_page_small)
        2. Clean LLM output (_clean_llm_output)
        3. Auto-link cross-references (_auto_link)
        4. Write page(s)
        5. Update backlinks
        6. Update index
        7. Append log
        """
        from core.llm_provider import detect_model_tier

        schema = self._get_schema()
        tier = detect_model_tier(getattr(self.llm, 'model_name', ''))

        # Step 1: Generate wiki page(s) (small model uses step-by-step fallback)
        if tier == "small":
            wiki_pages = self._generate_wiki_page_small(content, source_name, schema)
        else:
            wiki_pages = self._generate_wiki_page(content, source_name, schema)

        if not wiki_pages:
            return {"wiki_pages": [], "log_entry": "", "error": "Generation failed"}

        all_fns = []
        all_modified = []
        log_entries = []

        for wiki_page in wiki_pages:
            if not wiki_page or not wiki_page.strip():
                continue

            # Step 2: Clean LLM output
            wiki_page = self._clean_llm_output(wiki_page)

            # Step 3: Auto-link cross-references
            wiki_page = self._auto_link(wiki_page)

            title = self._extract_title(wiki_page) or source_name.replace(".md", "")
            fn = self._safe_filename(title) + ".md"
            p = self.paths["pages"] / fn

            # Step 3.5: Dedup detection - check if this page already exists
            matched_title = self._dedup_detect(title)
            if matched_title and matched_title != title:
                existing_fn = self._safe_filename(matched_title) + ".md"
                existing_p = self.paths["pages"] / existing_fn
                if existing_p.exists():
                    old_content = existing_p.read_text(encoding="utf-8")
                    merged = self._merge_pages(matched_title, old_content, title, wiki_page, source_name)
                    if merged:
                        wiki_page = merged
                        title = matched_title
                        fn = existing_fn
                        p = existing_p
                        self._append_log("merge", f"Merged {title} with new content from {source_name}")
                        self._update_backlinks_for_page(title, wiki_page)
                        all_fns.append(fn)
                        log_entries.append(f"{title}(merged)")
                        p.write_text(wiki_page, encoding="utf-8")
                        self._page_cache[title] = wiki_page
                        self._sync_tags_from_page(wiki_page)
                        self._update_index(title, fn, source_name)
                        all_modified.append(fn)
                        continue

            # Step 4: Write page(s)
            p.write_text(wiki_page, encoding="utf-8")
            self._page_cache[title] = wiki_page

            # 同步 tags 到 tags.json
            self._sync_tags_from_page(wiki_page)

            # Update backlinks
            modified = self._update_backlinks_for_page(title, wiki_page)
            all_modified.extend(modified)

            # Update index
            self._update_index(title, fn, source_name)

            all_fns.append(fn)
            log_entries.append(title)

        # 分析影响 → 更新已有页面
        affected = self._analyze_impact(content, source_name)
        if affected:
            updated = self._update_pages(content, source_name, affected)
            if updated:
                all_modified = list(set(all_modified + updated))

        # 自动创建不存在的链接页面
        auto_created = self._autocreate_linked_pages(all_fns)
        if auto_created:
            all_fns.extend(auto_created)
            all_modified = list(set(all_modified + auto_created))

        # Log
        extra = f" (updated {len(all_modified)})" if all_modified else ""
        titles_str = ", ".join(log_entries)
        log = self._append_log("ingest", f"Imported {source_name} -> {titles_str}{extra}")
        return {"wiki_pages": all_fns, "modified_pages": all_modified, "log_entry": log}

    def _generate_wiki_page(self, content: str, source_name: str, schema: str = "") -> list[str]:
        """LLM generates one or more structured wiki pages from raw content.
        Returns a list of page strings (Markdown with frontmatter).
        """
        schema_context = f"""Wiki 格式规范：
{schema[:2000]}
""" if schema else ""

        # 读取现有 index，让 LLM 知道已有哪些页面可交叉引用
        index_context = ""
        try:
            index_text = self.paths["index"].read_text(encoding="utf-8")[:2000]
            if index_text.strip():
                index_context = f"\n已有 Wiki 页面列表（供交叉引用参考）：\n{index_text}\n"
        except:
            pass

        # 读取已有 tags，供 LLM 优先复用
        tags_context = self._format_tags_context()

        prompt = load("wiki_ingest.txt",
            schema_context=schema_context,
            index_context=index_context,
            tags_context=tags_context,
            content=content[:5000],
            source_name=source_name,
        )

        try:
            raw_output = self.llm.chat([
                {"role": "system", "content": "你是 Wiki 编辑助手，输出结构化的 Markdown Wiki 页面。可能输出多个页面，用 ---NEWPAGE--- 分隔。"},
                {"role": "user", "content": prompt}
            ], label="ingest")
            # Split on separator and return list of pages
            if "---NEWPAGE---" in raw_output:
                pages = [p.strip() for p in raw_output.split("---NEWPAGE---")]
            else:
                pages = [raw_output.strip()]
            return [p for p in pages if p]
        except Exception:
            # Fallback: simple markdown conversion
            return [f"""---
title: {source_name.replace('.md', '')}
type: source-summary
created: {datetime.now().strftime('%Y-%m-%d')}
sources: [{source_name}]
---

# {source_name.replace('.md', '')}

{content}
"""]

    def _extract_title(self, wiki_page: str) -> Optional[str]:
        """Extract title from frontmatter or H1 heading."""
        # Try frontmatter
        m = re.search(r"^---\s*\ntitle:\s*(.+?)\s*\n", wiki_page, re.MULTILINE)
        if m:
            return m.group(1).strip()
        # Try H1
        m = re.search(r"^#\s+(.+?)$", wiki_page, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return None

    # ==================== Auto-link & Clean ====================

    def _clean_llm_output(self, raw: str) -> str:
        """Filter LLM verbose prefixes/suffixes, code-block wrappers, and whitespace.

        Called after _generate_wiki_page / _generate_wiki_page_small to ensure
        clean Markdown enters the wiki store.
        """
        if not raw:
            return raw

        # 1. Remove common Chinese LLM prefixes
        prefixes = [
            r'^(好的|没问题|以下是|这是您需要的|根据您的要求|好的，以下是)[^。\n]*[。：:\n]',
            r'^(当然|可以的|明白了)[^。\n]*[。：:\n]',
            r'^(Sure|OK|Here is)[^.]*\.[ \n]',
        ]
        for pat in prefixes:
            raw = re.sub(pat, '', raw, flags=re.IGNORECASE)

        # 2. Remove markdown code block wrappers (```markdown ... ```)
        raw = re.sub(r'^```(?:markdown|md|wiki|yaml)?\s*\n', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\n```\s*$', '', raw)

        # 3. Strip leading/trailing whitespace and blank lines
        raw = raw.strip()
        raw = re.sub(r'\n{3,}', '\n\n', raw)

        return raw

    def _auto_link(self, wiki_page: str) -> str:
        """Auto-add [[...]] cross-references to known page titles in content.

        Per IMPACT-MAP: called after _generate_wiki_page() return to
        automatically complete cross-references.

        Strategy:
        - Get all known page titles, sorted by length desc (longest first)
        - For each title found in body text (outside code blocks / existing links):
          wrap it as [[title]]
        - Skip titles already inside [[...]] or markdown links [text](url)
        - Skip the page's own title (don't self-link)
        """
        known_titles = [fp.stem for fp in self._list_pages()]
        if not known_titles:
            return wiki_page

        # Sort by length descending so longer titles match first
        known_titles.sort(key=len, reverse=True)

        # Extract this page's own title to avoid self-linking
        own_title = self._extract_title(wiki_page)

        # Split into frontmatter + body (only auto-link body)
        parts = wiki_page.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[0] + '---' + parts[1] + '---'
            body = parts[2]
        else:
            frontmatter = ''
            body = wiki_page

        # Find code blocks and existing links — protect them
        protected = []

        def protect(m):
            protected.append(m.group(0))
            return f'\x00PROTECT{len(protected)-1}\x00'

        body = re.sub(r'```[\s\S]*?```', protect, body)           # code blocks
        body = re.sub(r'`[^`]+`', protect, body)                  # inline code
        body = re.sub(r'\[\[.+?\]\]', protect, body)              # existing [[...]]
        body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', protect, body)   # markdown links [text](url)

        for title in known_titles:
            if title == own_title:
                continue
            if len(title) < 2:  # skip single-char titles
                continue
            # 完整词组匹配，避免子串误匹配（如"支付"误匹配到"在线支付服务"）
            # 中文词边界：前后不能是 CJK 字符、字母、数字
            # ASCII 词边界：前后不能是字母、数字、下划线
            escaped = re.escape(title)
            # 判断 title 是否含 CJK 字符，选择对应边界
            has_cjk = bool(re.search(r'[\u4e00-\u9fff]', title))
            if has_cjk:
                # 中文词边界：前后不能是 CJK 统一表意文字、字母、数字
                boundary_l = r'(?<![\u4e00-\u9fffA-Za-z0-9])'
                boundary_r = r'(?![\u4e00-\u9fffA-Za-z0-9])'
            else:
                # ASCII 词边界
                boundary_l = r'(?<![A-Za-z0-9_])'
                boundary_r = r'(?![A-Za-z0-9_])'
            replacement = f'[[{title}]]'
            body = re.sub(
                boundary_l + escaped + boundary_r,
                replacement,
                body
            )

        # Restore protected blocks
        for i, block in enumerate(protected):
            body = body.replace(f'\x00PROTECT{i}\x00', block)

        return frontmatter + body

    def _generate_wiki_page_small(self, content: str, source_name: str, schema: str = "") -> list[str]:
        """Multi-step wiki page generation for small models (fallback).

        When the LLM is too small to generate a full wiki page in one pass,
        this method breaks it into three steps:
        1. Extract title + frontmatter
        2. Generate body sections
        3. Add cross-references + 参见

        Returns a list of page strings (typically one).
        """
        schema_context = f"Wiki 格式规范：\n{schema[:1500]}\n" if schema else ""

        # Step 1: Title + frontmatter
        step1_prompt = load("wiki_small_step1.txt", schema_context=schema_context, content=content[:3000])
        try:
            frontmatter = self.llm.chat([
                {"role": "system", "content": "你只输出 YAML frontmatter，不要其他内容。"},
                {"role": "user", "content": step1_prompt}
            ], label="ingest")
        except Exception:
            frontmatter = f"---\ntitle: {source_name.replace('.md', '')}\ntype: source-summary\ncreated: {datetime.now().strftime('%Y-%m-%d')}\nsources: [{source_name}]\n---"

        frontmatter = self._clean_llm_output(frontmatter)
        if not frontmatter.startswith('---'):
            frontmatter = f"---\ntitle: {source_name.replace('.md', '')}\ntype: source-summary\ncreated: {datetime.now().strftime('%Y-%m-%d')}\nsources: [{source_name}]\n---"

        # Step 2: Body sections
        step2_prompt = load("wiki_small_step2.txt", content=content[:5000])
        try:
            body = self.llm.chat([
                {"role": "system", "content": "你是 Wiki 编辑助手，输出结构化的 Markdown 正文。"},
                {"role": "user", "content": step2_prompt}
            ], label="ingest")
        except Exception:
            body = f"# {source_name.replace('.md', '')}\n\n{content}"

        body = self._clean_llm_output(body)

        # Step 3: Cross-references + 参见
        existing_titles = [fp.stem for fp in self._list_pages()]
        step3_prompt = load("wiki_small_step3.txt",
            existing_titles=', '.join(existing_titles[:30]),
            body_tail=body[-2000:],
        )
        try:
            see_also = self.llm.chat([
                {"role": "system", "content": "你只输出 ## 参见 章节内容，不要其他。"},
                {"role": "user", "content": step3_prompt}
            ], label="ingest")
        except Exception:
            see_also = ""

        see_also = self._clean_llm_output(see_also)
        if see_also and '## 参见' not in body:
            body = body.rstrip() + '\n\n' + see_also

        return [frontmatter + '\n\n' + body]

    def _get_backlinks_data(self) -> dict:
        """Read backlinks.json, return {page: {incoming: [...], outgoing: [...]}}."""
        try:
            return json.loads(self._backlinks_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_backlinks_data(self, data: dict):
        """Write backlinks.json."""
        self._backlinks_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def get_backlinks(self, page_name: str) -> dict:
        """Get incoming/outgoing backlinks for a page.
        Returns: {"incoming": [str], "outgoing": [str]}
        """
        data = self._get_backlinks_data()
        return data.get(page_name, {"incoming": [], "outgoing": []})

    def _update_backlinks_for_page(self, title: str, content: str) -> list[str]:
        """Scan [[...]] refs in content, update backlinks.json.
        Returns: list of page names that were modified.
        """
        # Find all [[...]] refs in content
        refs = re.findall(r"\[\[(.+?)\]\]", content)
        refs = list(set(refs))  # deduplicate

        data = self._get_backlinks_data()
        modified = []

        # Ensure this page entry exists
        if title not in data:
            data[title] = {"incoming": [], "outgoing": []}

        # Update outgoing for this page
        old_outgoing = set(data[title].get("outgoing", []))
        new_outgoing = set(refs)
        if old_outgoing != new_outgoing:
            data[title]["outgoing"] = sorted(new_outgoing)
            modified.append(title)

        # Update incoming for referenced pages
        for ref in refs:
            if ref not in data:
                data[ref] = {"incoming": [], "outgoing": []}
            incoming = set(data[ref].get("incoming", []))
            incoming.add(title)
            data[ref]["incoming"] = sorted(incoming)

        # Remove stale incoming links (pages this page no longer references)
        removed = old_outgoing - new_outgoing
        for ref in removed:
            if ref in data:
                incoming = set(data[ref].get("incoming", []))
                incoming.discard(title)
                data[ref]["incoming"] = sorted(incoming)

        self._save_backlinks_data(data)
        return modified

    # ==================== Read / List ====================

    def _list_pages(self) -> list[Path]:
        """List all wiki page files (excluding index.md and log.md)."""
        return [f for f in self.paths["pages"].glob("*.md")
                if f.name not in ("index.md", "log.md")]

    def page_count(self) -> int:
        return len(self._list_pages())

    def get_index(self) -> str:
        """Return wiki index.md content."""
        p = self.paths["index"]
        if p.exists():
            return p.read_text(encoding="utf-8")
        # Fallback: auto-generate
        pages = sorted(self._list_pages())
        lines = ["# Wiki index\n", "> Auto-generated\n"]
        for fp in pages:
            title = fp.stem
            lines.append(f"- [{title}](./{fp.name}) | {title} | | {title}")
        return "\n".join(lines)

    def read_page(self, title: str) -> Optional[str]:
        """Read wiki page by title, returns Markdown or None."""
        if title in self._page_cache:
            return self._page_cache[title]
        # Exact filename match
        safe = self._safe_filename(title)
        p = self.paths["pages"] / f"{safe}.md"
        if p.exists():
            content = p.read_text(encoding="utf-8")
            self._page_cache[title] = content
            return content
        # Fuzzy match
        for f in self.paths["pages"].glob("*.md"):
            if f.stem == safe or f.stem.lower() == title.lower():
                content = f.read_text(encoding="utf-8")
                self._page_cache[title] = content
                return content
        return None

    # ==================== Lint (Health Check) ====================

    def lint(self) -> list:
        """Wiki health check. Returns [{type, pages, description, severity}]."""
        issues = []
        all_pages = {f.stem: f for f in self._list_pages()}

        # Read index.md references
        index_path = self.paths["index"]
        index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        indexed = set()
        for line in index_text.split("\n"):
            if "](" in line:
                m = re.search(r"\(\.?/?(.+?\.md)\)", line)
                if m:
                    indexed.add(m.group(1).replace(".md", ""))

        # Orphan pages
        for name in all_pages:
            if name not in indexed and len(all_pages) > 1:
                issues.append({
                    "type": "orphan",
                    "pages": [name],
                    "description": f"页面「{name}」未被索引",
                    "severity": "warning"
                })

        # Broken cross-references (only check explicit [[links]], not plain text)
        for name, fp in all_pages.items():
            content = fp.read_text(encoding="utf-8")
            content_clean = re.sub(r"```[\s\S]*?```", "", content)
            content_clean = re.sub(r"`[^`]+`", "", content_clean)
            refs = re.findall(r"\[\[(.+?)\]\]", content_clean)
            for ref in refs:
                if ref not in all_pages:
                    issues.append({
                        "type": "missing_crossref",
                        "pages": [name, ref],
                        "description": f"「{name}」引用了不存在的页面「{ref}」",
                        "severity": "error"
                    })

        # Expired pages (frontmatter valid_until)
        for name, fp in all_pages.items():
            content = fp.read_text(encoding="utf-8")
            m = re.search(r"valid_until:\s*(\d{4}-\d{2}-\d{2})", content)
            if m:
                try:
                    deadline = datetime.strptime(m.group(1), "%Y-%m-%d")
                    if deadline < datetime.now():
                        issues.append({
                            "type": "expired",
                            "pages": [name],
                            "description": f"页面「{name}」有效期已过 ({m.group(1)})",
                            "severity": "warning"
                        })
                except ValueError:
                    pass

        return issues

    # ==================== Helpers ====================

    def _safe_filename(self, name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "_", name).strip()

    def _update_index(self, title, fn, source_name):
        """Update index.md with new page entry."""
        p = self.paths["index"]
        current = p.read_text(encoding="utf-8") if p.exists() else ""
        entry = f"- [{title}](./{fn}) | {title} | | {source_name}"
        if entry not in current:
            p.write_text(current.rstrip() + "\n" + entry + "\n", encoding="utf-8")

    def _append_log(self, action, detail):
        """Append operation to log.md."""
        p = self.paths["log"]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## [{ts}] {action}\n{detail}\n\n"
        current = p.read_text(encoding="utf-8") if p.exists() else ""
        p.write_text(current + entry, encoding="utf-8")
        return f"[{ts}] {action}: {detail}"

    # Backward compat alias
    _log = _append_log

    def _autocreate_linked_pages(self, source_pages: list[str]) -> list[str]:
        """
        扫描指定页面的所有 [[链接]]，对不存在的页面自动创建。
        source_pages: 刚写入的页面文件名列表（如 ['林晓蕾.md']）
        returns: 新创建的页面文件名列表
        """
        new_pages = []
        # Collect all [[links]] from source pages
        links_to_check = []
        for fn in source_pages:
            p = self.paths["pages"] / fn
            if not p.exists():
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except Exception:
                continue
            # Extract [[links]], skipping those inside code blocks
            clean = p.read_text(encoding="utf-8")
            clean = re.sub(r"```[\s\S]*?```", "", clean)
            clean = re.sub(r"`[^`]+`", "", clean)
            found = re.findall(r"\[\[(.+?)\]\]", clean)
            for link in found:
                safe = self._safe_filename(link)
                target = self.paths["pages"] / f"{safe}.md"
                if not target.exists() and link not in links_to_check:
                    links_to_check.append(link)

        if not links_to_check:
            return []

        # For each missing page, generate content using LLM
        for link_title in links_to_check:
            safe = self._safe_filename(link_title)
            target = self.paths["pages"] / f"{safe}.md"

            # Build context from pages that reference this link
            context_parts = []
            for fn in source_pages:
                p = self.paths["pages"] / fn
                if p.exists():
                    try:
                        text = p.read_text(encoding="utf-8")
                        context_parts.append(f"## {fn}\n{text[:2000]}")
                    except Exception:
                        pass

            if not context_parts:
                continue

            context_str = "\n\n---\n\n".join(context_parts)

            prompt = load("wiki_autocreate.txt",
                link_title=link_title,
                created_date=datetime.now().strftime('%Y-%m-%d'),
                context_str=context_str,
            )
            try:
                raw = self.llm.chat([
                    {"role": "system", "content": "你是 Wiki 编辑助手，只基于提供的引用内容生成页面。如果引用中没有实质性信息，回复 EMPTY_PAGE。"},
                    {"role": "user", "content": prompt}
                ], label="ingest")
                raw = raw.strip()
                if raw.upper().startswith("EMPTY_PAGE") or len(raw) < 20:
                    continue

                page_content = self._clean_llm_output(raw)
                target.write_text(page_content, encoding="utf-8")
                self._page_cache[link_title] = page_content

                # Update backlinks
                self._update_backlinks_for_page(link_title, page_content)

                # Update index
                fn = f"{safe}.md"
                self._update_index(link_title, fn, "auto-created")
                new_pages.append(fn)
                self._append_log("autocreate", f"Created page {link_title} from [[link]] in {', '.join(source_pages)}")
            except Exception:
                continue

        return new_pages

    # ==================== Dedup & Merge ====================

    def _dedup_detect(self, title: str) -> Optional[str]:
        """Detect if a new page title duplicates an existing wiki page.

        Uses title normalization + fuzzy matching:
        1. Strip common suffixes
        2. Compare normalized title against all existing page titles
        3. Return matched filename or None
        """
        if not title:
            return None

        # Normalize
        norm = title.lower().strip()
        norm = re.sub(r"[（(].*?[）)]", "", norm)
        norm = norm.replace('有限公司', '').replace('公司', '')
        norm = norm.replace('v', '').replace('version', '')
        norm = norm.replace(' ', '').replace('-', '').replace('_', '')
        norm = norm.replace('[', '').replace(']', '').lstrip('[')

        if not norm:
            return None

        existing_pages = [fp.stem for fp in self._list_pages()]
        best_match = None
        best_score = 0.0

        for stem in existing_pages:
            stem_norm = stem.lower().strip()
            stem_norm = re.sub(r"[（(].*?[）)]", "", stem_norm)
            stem_norm = stem_norm.replace('有限公司', '').replace('公司', '')
            stem_norm = stem_norm.replace('v', '').replace('version', '')
            stem_norm = stem_norm.replace(' ', '').replace('-', '').replace('_', '')
            stem_norm = stem_norm.replace('[', '').replace(']', '').lstrip('[')

            if not stem_norm:
                continue

            # Jaccard similarity on character sets
            set1 = set(norm)
            set2 = set(stem_norm)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            score = intersection / union if union > 0 else 0

            # Substring containment bonus
            if norm in stem_norm or stem_norm in norm:
                score = max(score, 0.8)

            # Key entity overlap bonus
            if len(norm) >= 4 and len(stem_norm) >= 4:
                if norm[:4] in stem_norm or stem_norm[:4] in norm:
                    score = max(score, 0.75)

            if score > best_score:
                best_score = score
                best_match = stem

        if best_score >= 0.6:
            return best_match
        return None

    def _merge_pages(self, old_title, old_content, new_title, new_content, source_name):
        """Merge two wiki pages about the same topic using LLM."""
        from core.prompts import load

        old_sources = source_name
        m = __import__("re").search(r"sources:\s*\[(.+?)\]", old_content)
        if m:
            old_sources = m.group(1)

        prompt = load("wiki_merge.txt",
            old_sources=old_sources,
            old_page=old_content,
            new_source=source_name,
            new_page=new_content,
        )

        try:
            merged = self.llm.chat([
                {"role": "system", "content": "你是 Wiki 编辑助手。合并两个同主题的页面，保留所有信息。"},
                {"role": "user", "content": prompt}
            ], label="ingest")

            if not merged or not merged.strip():
                return None

            merged = self._clean_llm_output(merged)

            if not merged.startswith('---'):
                merged = f"---\ntitle: {old_title}\ntype: merged\nsources: [{old_sources}, {source_name}]\n---\n\n{merged}"

            return merged
        except Exception:
            return None

    # ==================== Impact Analysis ====================

    def _analyze_impact(self, content, source_name):
        """Analyze which existing pages may be affected by new content.

        Scans the new content for entity names that match existing page titles.
        """
        if not content:
            return []

        affected = []
        existing_titles = set()
        for fp in self._list_pages():
            existing_titles.add(fp.stem)

        # Check each existing page title against the new content
        for stem in existing_titles:
            if len(stem) >= 2 and stem in content:
                affected.append(stem)

        return affected

    def _update_pages(self, content, source_name, affected):
        """Update affected existing pages with new information.

        For each affected page, uses LLM to determine if update is needed.
        Returns list of modified page filenames.
        """
        if not affected:
            return []

        modified = []

        for page_title in affected:
            fn = self._safe_filename(page_title) + ".md"
            p = self.paths["pages"] / fn

            if not p.exists():
                continue

            old_content = p.read_text(encoding="utf-8")

            update_prompt = (
                f"\u73b0\u6709 Wiki \u9875\u9762\u6807\u9898\uff1a{page_title}\n\n"
                f"\u73b0\u6709\u9875\u9762\u5185\u5bb9\uff1a\n{old_content[:3000]}\n\n"
                f"\u65b0\u6765\u6e90 ({source_name}) \u7684\u5185\u5bb9\uff1a\n{content[:3000]}\n\n"
                "\u8bf7\u5224\u65ad\u65b0\u5185\u5bb9\u662f\u5426\u5305\u542b\u73b0\u6709\u9875\u9762\u4e2d\u4e0d\u5b58\u5728\u7684\u91cd\u8981\u4fe1\u606f\uff0c\u6216\u4e0e\u73b0\u6709\u9875\u9762\u77db\u76fe\u3002\n"
                "\u5982\u679c\u6709\uff0c\u8bf7\u8f93\u51fa**\u5b8c\u6574\u7684\u66f4\u65b0\u540e\u7684 Wiki \u9875\u9762**\uff08\u5fc5\u987b\u5305\u542b frontmatter\uff0c\u4ee5 --- \u5f00\u5934\uff09\u3002\n"
                "\u5982\u679c\u6ca1\u6709\u65b0\u4fe1\u606f\uff0c\u8bf7\u53ea\u8f93\u51fa\uff1aNO_UPDATE"
            )

            try:
                result = self.llm.chat([
                    {"role": "system", "content": "\u4f60\u662f Wiki \u7f16\u8f91\u52a9\u624b\u3002\u5224\u65ad\u662f\u5426\u9700\u8981\u66f4\u65b0\u73b0\u6709\u9875\u9762\u3002\u5982\u679c\u66f4\u65b0\uff0c\u8f93\u51fa\u5b8c\u6574\u7684 Wiki \u9875\u9762\u5185\u5bb9\uff0c\u4ee5 --- \u5f00\u5934\u3002"},
                    {"role": "user", "content": update_prompt}
                ], label="ingest")

                if result and "NO_UPDATE" not in result:
                    result = self._clean_llm_output(result)

                    # Strip any LLM commentary before the first frontmatter
                    result_lines = result.split("\n")
                    fm_start = -1
                    for j, rl in enumerate(result_lines):
                        if rl.strip() == "---":
                            fm_start = j
                            break

                    if fm_start >= 0:
                        result = "\n".join(result_lines[fm_start:])
                    else:
                        continue  # No frontmatter = garbage, skip

                    # Verify frontmatter has title
                    result = result.strip()
                    title_ok = False
                    for rl in result.split("\n"):
                        if rl.strip().startswith("title:"):
                            title_ok = True
                            break
                    if not title_ok:
                        continue  # Invalid frontmatter, skip

                    p.write_text(result, encoding="utf-8")
                    self._page_cache[page_title] = result
                    self._update_backlinks_for_page(page_title, result)
                    modified.append(fn)
                    self._append_log("update", f"Updated {page_title} from {source_name}")
            except Exception:
                continue

        return modified


    