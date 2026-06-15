import json
from typing import Any

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.memory_graph import EntityNode
from app.domain.services.prompts.memory import (
    PROFILE_SUMMARY_PROMPT,
    PROFILE_SUMMARY_SYSTEM_PROMPT,
)


class MemoryProfileSummarizer:
    """使用 LLM 为长期实体生成画像摘要。"""

    def __init__(self, llm: LLM, json_parser: JSONParser) -> None:
        self._llm = llm
        self._json_parser = json_parser

    async def summarize(
        self, entity: EntityNode, statements: list[str]
    ) -> tuple[list[str], list[str]]:
        """将实体相关陈述压缩为核心事实和特质。"""
        response = await self._llm.invoke(
            messages=[
                {
                    "role": "system",
                    "content": PROFILE_SUMMARY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": PROFILE_SUMMARY_PROMPT.format(
                        entity_name=entity.name,
                        entity_type=entity.type,
                        statements=statements[:50],
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = await self._parse_json(
            response.get("content"),
            default_value={"core_facts": [], "traits": []},
        )
        return (
            _coerce_str_list(parsed.get("core_facts"), limit=8),
            _coerce_str_list(parsed.get("traits"), limit=8),
        )

    async def _parse_json(self, content: Any, default_value: dict[str, Any]) -> dict[str, Any]:
        """解析 LLM 返回内容，异常或非对象结果统一回退到默认结构。"""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        try:
            parsed = await self._json_parser.invoke(content, default_value=default_value)
        except Exception:
            return default_value
        if not isinstance(parsed, dict):
            return default_value
        return parsed


def _coerce_str_list(value, limit: int) -> list[str]:
    """把 LLM 返回值收敛为短字符串列表。"""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:limit]:
        text = str(item).strip()
        if text:
            result.append(text[:200])
    return result
