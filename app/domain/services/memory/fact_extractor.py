import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.memory_graph import ChunkNode, StatementNode, stable_memory_graph_id
from app.domain.services.prompts.memory import (
    EXTRACT_STATEMENTS_PROMPT,
    EXTRACT_STATEMENTS_SYSTEM_PROMPT,
    EXTRACT_TRIPLETS_PROMPT,
    EXTRACT_TRIPLETS_SYSTEM_PROMPT,
)
from app.utils.datetime import parse_optional_datetime


class ExtractedEntity(BaseModel):
    """LLM 萃取出的实体。"""

    model_config = ConfigDict(extra="ignore")

    entity_idx: int = 0
    name: str
    type: str = "Thing"
    description: str = ""
    importance: float = 0.5
    confidence: float = 0.8


class ExtractedTriplet(BaseModel):
    """LLM 萃取出的实体三元组。"""

    model_config = ConfigDict(extra="ignore")

    subject_id: int = 0
    predicate: str = ""
    object_id: int = 0
    evidence: str
    importance: float = 0.5
    confidence: float = 0.8
    valid_at: str | None = None
    invalid_at: str | None = None


class ExtractedEvent(BaseModel):
    """LLM 萃取出的一次性经历事件。"""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    description: str = ""
    event_time: str | None = None
    participants: list[str] = Field(default_factory=list)


class ExtractedStatement(BaseModel):
    """LLM 萃取出的结构化原子陈述。"""

    model_config = ConfigDict(extra="ignore")

    statement: str = ""
    statement_type: str = "FACT"
    temporal_type: str = "STATIC"
    has_unsolved_reference: bool = False
    importance: float = 0.5
    confidence: float = 0.8
    valid_at: str | None = None
    invalid_at: str | None = None


class ExtractedTriplets(BaseModel):
    """LLM 萃取出的实体和关系集合。"""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    triplets: list[ExtractedTriplet] = Field(default_factory=list)
    events: list[ExtractedEvent] = Field(default_factory=list)


class _ExtractedStatements(BaseModel):
    statements: list[ExtractedStatement] = Field(default_factory=list)


class MemoryFactExtractor:
    """使用 LLM 从记忆文本中抽取原子陈述、实体和三元组。"""

    def __init__(self, llm: LLM, json_parser: JSONParser) -> None:
        self._llm = llm
        self._json_parser = json_parser

    async def extract_statements(
        self, chunks: list[ChunkNode], dialog_at: str | None = None
    ) -> list[StatementNode]:
        """从文本分块中抽取结构化原子陈述。"""
        statements: list[StatementNode] = []
        parsed_dialog_at = parse_optional_datetime(dialog_at)
        for chunk in chunks:
            response = await self._llm.invoke(
                messages=[
                    {
                        "role": "system",
                        "content": EXTRACT_STATEMENTS_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": EXTRACT_STATEMENTS_PROMPT.format(
                            text=chunk.text,
                            dialog_at=dialog_at or "NULL",
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            parsed = await self._parse_json(response.get("content"), {"statements": []})
            extracted = _ExtractedStatements.model_validate(parsed)
            for statement in extracted.statements:
                text = statement.statement.strip()
                if statement.has_unsolved_reference:
                    continue
                if not text:
                    continue
                index = len(statements)
                valid_at = parse_optional_datetime(statement.valid_at)
                invalid_at = parse_optional_datetime(statement.invalid_at)
                if (
                    valid_at is None
                    and statement.temporal_type.upper() == "DYNAMIC"
                ):
                    valid_at = parsed_dialog_at
                statements.append(
                    StatementNode(
                        id=stable_memory_graph_id(
                            chunk.user_id, chunk.id, "statement", str(index), text
                        ),
                        user_id=chunk.user_id,
                        chunk_id=chunk.id,
                        index=index,
                        text=text,
                        statement_type=statement.statement_type,
                        temporal_type=statement.temporal_type,
                        importance=statement.importance,
                        confidence=statement.confidence,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                    )
                )
        return statements

    async def extract_triplets(
        self,
        statements: list[StatementNode],
        dialog_at: str | None = None,
    ) -> ExtractedTriplets:
        """从原子陈述中抽取实体和三元组。"""
        if not statements:
            return ExtractedTriplets()
        response = await self._llm.invoke(
            messages=[
                {
                    "role": "system",
                    "content": EXTRACT_TRIPLETS_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": EXTRACT_TRIPLETS_PROMPT.format(
                        statements=[statement.text for statement in statements],
                        dialog_at=dialog_at or "NULL",
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = await self._parse_json(
            response.get("content"), {"entities": [], "triplets": [], "events": []}
        )
        return ExtractedTriplets.model_validate(parsed)

    async def _parse_json(self, content: Any, default_value: dict[str, Any]) -> Any:
        """解析 LLM 返回内容，异常或非对象结果统一回退到默认结构。"""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        try:
            parsed = await self._json_parser.invoke(
                content, default_value=default_value
            )
        except Exception:
            return default_value
        if not isinstance(parsed, dict):
            return default_value
        return parsed
