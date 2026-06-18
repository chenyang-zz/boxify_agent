from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.memory_graph import EntityNode
from app.domain.services.prompts.memory import (
    PROFILE_SUMMARY_PROMPT,
    PROFILE_SUMMARY_SYSTEM_PROMPT,
)
from app.utils.json_utils import parse_json_object


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
        parsed = await parse_json_object(
            self._json_parser,
            response.get("content"),
            default_value={"core_facts": [], "traits": []},
        )
        return (
            _coerce_str_list(parsed.get("core_facts"), limit=8),
            _coerce_str_list(parsed.get("traits"), limit=8),
        )


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
