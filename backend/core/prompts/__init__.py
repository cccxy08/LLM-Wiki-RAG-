"""Prompt 加载器 — 从 prompts/ 目录读取 .txt 文件

使用 string.Template ($var 语法) 替代 str.format()，
避免 prompt 中的 JSON 示例 {} 被 format() 误解析导致 KeyError。
"""
import re
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).parent

# 兼容：自动将 {var} 转为 $var，{{ / }} 转为 { / }
_BRACE_VAR = re.compile(r'\{([A-Za-z_][A-Za-z0-9_]*)\}')
_DOUBLE_BRACE = re.compile(r'\{\{|\}\}')


def _convert_to_template(text: str) -> str:
    """将 {var} 格式的 prompt 转为 $var 格式（兼容旧 prompt 文件）
    - {question} → $question
    - {{ → {, }} → }  (字面大括号)
    """
    # 先替换双花括号（字面大括号）
    text = text.replace('{{', '\x00LBRACE\x00')
    text = text.replace('}}', '\x00RBRACE\x00')
    # 替换单花括号变量
    text = _BRACE_VAR.sub(r'$\1', text)
    # 还原字面大括号
    text = text.replace('\x00LBRACE\x00', '{')
    text = text.replace('\x00RBRACE\x00', '}')
    return text


def load(name: str, **kwargs) -> str:
    """加载 prompt 文本文件。name 为文件名如 'agent_preprocess.txt'。
    如果传入 kwargs，使用 string.Template 替换 $var 变量。
    支持旧格式 {var} 和 {{ / }} 转义，自动转换。
    """
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    if kwargs:
        text = _convert_to_template(text)
        return Template(text).safe_substitute(**kwargs)
    return text
