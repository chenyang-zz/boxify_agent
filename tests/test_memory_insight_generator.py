import json

import pytest

from app.domain.services.memory.insight_generator import MemoryInsightGenerator
from app.domain.services.prompts.memory import REFLECT_PROMPT, REFLECT_SYSTEM_PROMPT


def test_memory_reflect_prompt_defines_insight_schema():
    assert "insights" in REFLECT_PROMPT
    assert "theme" in REFLECT_PROMPT
    assert "based_on" in REFLECT_PROMPT
    assert "高层洞察" in REFLECT_PROMPT
    assert "严格 JSON" in REFLECT_SYSTEM_PROMPT


@pytest.mark.anyio
async def test_memory_insight_generator_parses_reflected_insights():
    llm = FakeReflectLLM(
        {
            "insights": [
                {
                    "theme": "音乐偏好",
                    "content": "用户偏好华语流行音乐。",
                    "based_on": ["周杰伦"],
                    "importance": 0.8,
                    "confidence": 0.9,
                }
            ]
        }
    )
    generator = MemoryInsightGenerator(llm=llm, json_parser=FakeJSONParser())

    insights = await generator.generate(
        memory_block="- 【生命体】周杰伦",
        min_insights=1,
        max_insights=5,
    )

    assert len(insights) == 1
    assert insights[0].theme == "音乐偏好"
    assert insights[0].based_on == ["周杰伦"]
    assert llm.response_format == {"type": "json_object"}
    assert "记忆清单" in llm.messages[-1]["content"]
    assert "归纳 1~5 条高层洞察" in llm.messages[-1]["content"]


@pytest.mark.anyio
async def test_memory_insight_generator_returns_empty_list_for_bad_json():
    generator = MemoryInsightGenerator(
        llm=FakeReflectLLM("not-json"),
        json_parser=FakeJSONParser(),
    )

    insights = await generator.generate(
        memory_block="- 【生命体】周杰伦",
        min_insights=1,
        max_insights=5,
    )

    assert insights == []


@pytest.mark.anyio
async def test_memory_insight_generator_returns_empty_list_for_non_dict_json():
    generator = MemoryInsightGenerator(
        llm=FakeReflectLLM(["not-object"]),
        json_parser=FakeJSONParser(),
    )

    insights = await generator.generate(
        memory_block="- 【生命体】周杰伦",
        min_insights=1,
        max_insights=5,
    )

    assert insights == []


class FakeReflectLLM:
    def __init__(self, content):
        self.content = content
        self.messages = []
        self.response_format = None

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.messages = messages
        self.response_format = response_format
        if isinstance(self.content, str):
            return {"content": self.content}
        return {"content": json.dumps(self.content, ensure_ascii=False)}

    @property
    def model_name(self):
        return "fake-reflect-llm"

    @property
    def temperature(self):
        return 0

    @property
    def max_tokens(self):
        return 0


class FakeJSONParser:
    async def invoke(self, text, default_value=None):
        try:
            parsed = json.loads(text)
        except Exception:
            return default_value
        return parsed if isinstance(parsed, dict) else default_value
