import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.services.prompts.memory import REFLECT_PROMPT, REFLECT_SYSTEM_PROMPT


class ReflectedInsight(BaseModel):
    """LLM 反思出的高层洞察。"""

    model_config = ConfigDict(extra="ignore")

    theme: str = ""
    content: str = ""
    based_on: list[str] = Field(default_factory=list)
    importance: float = 0.6
    confidence: float = 0.7


class _ReflectedInsights(BaseModel):
    insights: list[ReflectedInsight] = Field(default_factory=list)


class MemoryInsightGenerator:
    """使用 LLM 从记忆清单中生成高层洞察。"""

    def __init__(self, llm: LLM, json_parser: JSONParser) -> None:
        self._llm = llm
        self._json_parser = json_parser

    async def generate(
        self,
        memory_block: str,
        min_insights: int,
        max_insights: int,
    ) -> list[ReflectedInsight]:
        """调用 LLM 并解析洞察列表。"""
        response = await self._llm.invoke(
            messages=[
                {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REFLECT_PROMPT.format(
                        memory_block=memory_block,
                        min_insights=min_insights,
                        max_insights=max_insights,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = await self._parse_json(response.get("content"), {"insights": []})
        return _ReflectedInsights.model_validate(parsed).insights

    async def _parse_json(
        self, content: Any, default_value: dict[str, Any]
    ) -> dict[str, Any]:
        """解析 LLM JSON，失败或非对象结果时回退默认结构。"""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        try:
            parsed = await self._json_parser.invoke(
                content, default_value=default_value
            )
        except Exception:
            return default_value
        return parsed if isinstance(parsed, dict) else default_value
