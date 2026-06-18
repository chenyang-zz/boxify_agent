from pydantic import BaseModel, ConfigDict

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.memory_graph import (
    CommunityMemberResult,
    CommunityRelationResult,
    CommunityVoteEntity,
)
from app.domain.services.prompts.memory import (
    COMMUNITY_SUMMARY_PROMPT,
    COMMUNITY_SUMMARY_SYSTEM_PROMPT,
)
from app.utils.json_utils import parse_json_object


class _CommunitySummary(BaseModel):
    """LLM 生成的社区元数据。"""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    summary: str = ""


class MemoryCommunitySummarizer:
    """使用 LLM 为记忆社区生成名称和摘要。"""

    def __init__(self, llm: LLM, json_parser: JSONParser) -> None:
        self._llm = llm
        self._json_parser = json_parser

    async def summarize(
        self,
        members: list[CommunityVoteEntity | CommunityMemberResult],
        relationships: list[CommunityRelationResult],
    ) -> tuple[str, str]:
        """根据社区成员和内部关系生成中文名称、摘要。"""
        response = await self._llm.invoke(
            messages=[
                {"role": "system", "content": COMMUNITY_SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": COMMUNITY_SUMMARY_PROMPT.format(
                        members=_format_members(members),
                        relationships=_format_relationships(relationships),
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = await parse_json_object(self._json_parser, response.get("content"), {})
        result = _CommunitySummary.model_validate(parsed)
        return result.name.strip()[:10], result.summary.strip()[:80]


def _format_members(
    members: list[CommunityVoteEntity | CommunityMemberResult],
) -> str:
    """格式化社区成员供 LLM 阅读。"""
    lines = []
    for member in members:
        name = getattr(member, "name", None) or getattr(member, "entity_name", "")
        entity_type = getattr(member, "type", None) or getattr(member, "entity_type", "")
        description = getattr(member, "description", "")
        suffix = f"：{description}" if description else ""
        lines.append(f"- {name}（{entity_type}）{suffix}")
    return "\n".join(lines)


def _format_relationships(relationships: list[CommunityRelationResult]) -> str:
    """格式化社区内部关系供 LLM 阅读。"""
    if not relationships:
        return "无"
    return "\n".join(
        f"- {relation.source_name} {relation.name} {relation.target_name}"
        for relation in relationships
    )
