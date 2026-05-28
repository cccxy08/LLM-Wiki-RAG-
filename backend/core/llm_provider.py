"""LLM Provider - 可插拔的 LLM 调用层（含重试、降级、并发控制）"""
import asyncio
import re
import time
import logging
from abc import ABC, abstractmethod
from threading import Semaphore
from typing import Generator

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

from core.config import settings

logger = logging.getLogger(__name__)

# 全局限流：最多 3 个 LLM 并发请求
_global_semaphore = Semaphore(3)

# 可重试的错误类型（网络瞬断、超时、限流）
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)

# 降级回复
FALLBACK_MESSAGE = "系统繁忙，请稍后重试。"


class LLMError(Exception):
    """LLM 调用异常"""
    def __init__(self, message: str, retryable: bool = True, original: Exception = None):
        super().__init__(message)
        self.retryable = retryable
        self.original = original


class LLMProvider(ABC):
    """LLM 抽象基类 — 内置重试、降级、并发控制"""

    def __init__(self):
        self._max_retries = getattr(settings, 'llm_max_retries', 3)
        self._retry_base_delay = getattr(settings, 'llm_retry_delay_base', 1.0)
        self._retry_backoff = 2.0

    @abstractmethod
    def _raw_chat(self, messages: list[dict], stream: bool = False, label: str = ""):
        """子类实现实际的 API 调用"""
        ...

    @abstractmethod
    def _raw_chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """子类实现实际的流式 API 调用"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    def _classify_error(self, error: Exception) -> LLMError:
        """分类错误：可重试 vs 不可重试"""
        if isinstance(error, RETRYABLE_ERRORS):
            return LLMError(str(error), retryable=True, original=error)
        if isinstance(error, APIError):
            # 4xx 通常是客户端问题不可重试，5xx 可重试
            status = getattr(error, 'status_code', None) or getattr(error, 'http_status', None)
            if status and 400 <= status < 500 and status != 429:
                return LLMError(str(error), retryable=False, original=error)
            return LLMError(str(error), retryable=True, original=error)
        return LLMError(str(error), retryable=False, original=error)

    def chat(self, messages: list[dict], stream: bool = False, label: str = "") -> str:
        """同步对话 — 自动重试 + 降级"""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                with _global_semaphore:
                    return self._raw_chat(messages, stream=stream, label=label)
            except Exception as e:
                classified = self._classify_error(e)
                last_error = classified

                if not classified.retryable:
                    logger.error(f"LLM 不可恢复错误: {classified}")
                    break

                if attempt < self._max_retries - 1:
                    delay = self._retry_base_delay * (self._retry_backoff ** attempt)
                    logger.warning(
                        f"LLM 调用失败 (尝试 {attempt + 1}/{self._max_retries})，"
                        f"{delay:.1f}s 后重试: {str(e)[:200]}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM 调用全部 {self._max_retries} 次重试失败: {str(e)[:200]}")

        # 全部重试失败 → 降级回复
        import sys
        print(f"\n[LLM-FATAL] chat() 全部重试失败", file=sys.stderr)
        if last_error:
            print(f"[LLM-FATAL] 最后错误: {type(last_error).__name__}: {last_error}", file=sys.stderr)
            if last_error.original:
                print(f"[LLM-FATAL] 原始异常: {type(last_error.original).__name__}: {str(last_error.original)[:500]}", file=sys.stderr)
        print("", file=sys.stderr)
        return FALLBACK_MESSAGE

    def chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """流式对话 — 自动重试 + 降级"""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                with _global_semaphore:
                    for chunk in self._raw_chat_stream(messages):
                        yield chunk
                    return
            except Exception as e:
                classified = self._classify_error(e)
                last_error = classified

                if not classified.retryable:
                    logger.error(f"LLM 流式调用不可恢复错误: {classified}")
                    break

                if attempt < self._max_retries - 1:
                    delay = self._retry_base_delay * (self._retry_backoff ** attempt)
                    logger.warning(
                        f"LLM 流式调用失败 (尝试 {attempt + 1}/{self._max_retries})，"
                        f"{delay:.1f}s 后重试: {str(e)[:200]}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM 流式调用全部 {self._max_retries} 次重试失败: {str(e)[:200]}")

        # 全部失败 → 返回降级消息
        yield FALLBACK_MESSAGE


class OllamaProvider(LLMProvider):
    """Ollama 本地模型"""

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama"
        )
        self._model = settings.ollama_model

    @property
    def model_name(self) -> str:
        return self._model

    def _raw_chat(self, messages: list[dict], stream: bool = False, label: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=stream,
            temperature=0.3,
        )
        # 打印 token 使用情况
        if response.usage:
            prefix = f"[LLM] {label}" if label else "[LLM]"
            print(f"{prefix} model={self._model} prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens} total={response.usage.total_tokens}")
        return response.choices[0].message.content or ""

    def _raw_chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            temperature=0.3,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenAIProvider(LLMProvider):
    """OpenAI API"""

    def __init__(self):
        super().__init__()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY 未设置")
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key
        )
        self._model = settings.openai_model

    @property
    def model_name(self) -> str:
        return self._model

    def _raw_chat(self, messages: list[dict], stream: bool = False, label: str = "") -> str:
        if settings.debug:
            import sys
            print(f"[LLM-DEBUG] 调用: base_url={settings.openai_base_url} model={self._model}", file=sys.stderr)
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=stream,
            temperature=0.3,
        )
        # 打印 token 使用情况
        if response.usage:
            prefix = f"[LLM] {label}" if label else "[LLM]"
            print(f"{prefix} model={self._model} prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens} total={response.usage.total_tokens}")
        return response.choices[0].message.content or ""

    def _raw_chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            temperature=0.3,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ZhipuProvider(LLMProvider):
    """智谱 GLM API"""

    def __init__(self):
        super().__init__()
        if not settings.zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY 未设置")
        self.client = OpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=settings.zhipu_api_key
        )
        self._model = settings.zhipu_model

    @property
    def model_name(self) -> str:
        return self._model

    def _raw_chat(self, messages: list[dict], stream: bool = False, label: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=stream,
            temperature=0.3,
        )
        # 打印 token 使用情况
        if response.usage:
            prefix = f"[LLM] {label}" if label else "[LLM]"
            print(f"{prefix} model={self._model} prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens} total={response.usage.total_tokens}")
        return response.choices[0].message.content or ""

    def _raw_chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            temperature=0.3,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content



def detect_model_tier(model_name: str) -> str:
    """检测模型能力层级: small / medium / large

    优先精确匹配（短横/冒号分隔的 token），再降级子串匹配。
    避免子串误判：如 qwen3:32b 被小模型规则 qwen3 命中。
    """
    ml = model_name.lower()

    # 精确 token 集合（按 / - : 分隔后逐 token 匹配）
    tokens = set(re.split(r'[-_:]', ml))
    # 也保留完整名用于子串匹配
    has_token = lambda *ps: any(p in tokens for p in ps)
    contains = lambda *ps: any(p in ml for p in ps)

    # === Large: 旗舰级 ===
    if has_token("gpt-4", "gpt4") or contains("gpt-4", "gpt-4o"):
        return "large"
    if has_token("claude"):
        return "large"
    if has_token("glm-4") or contains("glm-4"):
        return "large"
    if contains("deepseek") and not has_token("7b", "8b", "1.3b", "1.5b"):
        return "large"  # deepseek-chat / deepseek-v3 = large; deepseek-7b ≠ large
    if contains("qwen-max", "qwen2.5-max"):
        return "large"

    # === Medium: 中等规模 ===
    if has_token("14b", "32b", "70b", "72b"):
        return "medium"
    if has_token("qwen2.5-32b") or contains("qwen2.5-32b", "llama3-70b"):
        return "medium"

    # === Small: 小模型 ===
    if has_token("1b", "1.5b", "1.3b", "3b", "7b", "8b"):
        return "small"
    if has_token("phi"):
        return "small"
    if contains("mistral") and not has_token("large", "medium", "nemo"):
        return "small"  # mistral-7b 系列；mistral-large 另算

    # 兜底
    return "medium"

def get_llm() -> LLMProvider:
    """工厂函数：根据配置返回对应的 LLM Provider"""
    provider_map = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "zhipu": ZhipuProvider,
    }
    provider_class = provider_map.get(settings.llm_provider)
    if provider_class is None:
        raise ValueError(f"不支持的 LLM Provider: {settings.llm_provider}")
    return provider_class()
