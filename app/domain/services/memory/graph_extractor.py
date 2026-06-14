import json
import re
from typing import Any

from pydantic import BaseModel, Field

from app.domain.external.embedding import EmbeddingModel
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.memory_graph import (
    ChunkNode,
    DialogueNode,
    EntityNode,
    MemoryGraph,
    MemoryGraphStats,
    MentionEdge,
    RelationEdge,
    StatementNode,
    stable_memory_graph_id,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.services.prompts.memory import (
    EXTRACT_STATEMENTS_PROMPT,
    EXTRACT_STATEMENTS_SYSTEM_PROMPT,
    EXTRACT_TRIPLETS_PROMPT,
    EXTRACT_TRIPLETS_SYSTEM_PROMPT,
)


class ExtractedEntity(BaseModel):
    """LLM 萃取出的实体。"""

    name: str
    type: str = "Thing"
    description: str = ""


class ExtractedTriplet(BaseModel):
    """LLM 萃取出的实体三元组。"""

    head: ExtractedEntity
    relation: str
    tail: ExtractedEntity
    evidence: str


class _ExtractedStatements(BaseModel):
    statements: list[dict[str, str]] = Field(default_factory=list)


class _ExtractedTriplets(BaseModel):
    triplets: list[ExtractedTriplet] = Field(default_factory=list)


class MemoryGraphExtractor:
    """把一段长期记忆萃取并写入四层溯源图谱。"""

    def __init__(
        self,
        llm: LLM,
        embedding: EmbeddingModel,
        json_parser: JSONParser,
        graph_repository: MemoryGraphRepository,
        chunk_size: int = 1200,
    ) -> None:
        self._llm = llm
        self._embedding = embedding
        self._json_parser = json_parser
        self._graph_repository = graph_repository
        self._chunk_size = chunk_size

    async def extract_memory(
        self, memory_id: str, user_id: str, content: str
    ) -> MemoryGraphStats:
        """萃取一条 PG 记忆并写入 Neo4j 图谱。"""
        dialogue = DialogueNode(
            id=stable_memory_graph_id(user_id, memory_id, "dialogue"),
            user_id=user_id,
            memory_id=memory_id,
            summary=content.strip(),
        )
        chunks = self._chunk_text(user_id, dialogue.id, content)
        statements = await self._extract_statements(user_id, chunks)
        triplets = await self._extract_triplets(statements)
        graph = await self._build_graph(dialogue, chunks, statements, triplets)
        await self._graph_repository.save_graph(graph)
        return graph.stats()

    def _chunk_text(
        self, user_id: str, dialogue_id: str, content: str
    ) -> list[ChunkNode]:
        normalized = content.strip()
        if not normalized:
            return []
        chunks: list[str] = []
        cursor = 0
        while cursor < len(normalized):
            chunks.append(normalized[cursor : cursor + self._chunk_size])
            cursor += self._chunk_size
        return [
            ChunkNode(
                id=stable_memory_graph_id(user_id, dialogue_id, "chunk", str(index)),
                user_id=user_id,
                dialogue_id=dialogue_id,
                index=index,
                text=chunk,
            )
            for index, chunk in enumerate(chunks)
        ]

    async def _extract_statements(
        self, user_id: str, chunks: list[ChunkNode]
    ) -> list[StatementNode]:
        statements: list[StatementNode] = []
        for chunk in chunks:
            response = await self._llm.invoke(
                messages=[
                    {
                        "role": "system",
                        "content": EXTRACT_STATEMENTS_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": EXTRACT_STATEMENTS_PROMPT.format(text=chunk.text),
                    },
                ],
                response_format={"type": "json_object"},
            )
            parsed = await self._parse_json(response.get("content"), {"statements": []})
            extracted = _ExtractedStatements.model_validate(parsed)
            for raw in extracted.statements:
                text = str(raw.get("text", "")).strip()
                if not text:
                    continue
                index = len(statements)
                statements.append(
                    StatementNode(
                        id=stable_memory_graph_id(
                            user_id, chunk.id, "statement", str(index), text
                        ),
                        user_id=user_id,
                        chunk_id=chunk.id,
                        index=index,
                        text=text,
                    )
                )
        return statements

    async def _extract_triplets(
        self, statements: list[StatementNode]
    ) -> list[ExtractedTriplet]:
        if not statements:
            return []
        response = await self._llm.invoke(
            messages=[
                {
                    "role": "system",
                    "content": EXTRACT_TRIPLETS_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": EXTRACT_TRIPLETS_PROMPT.format(
                        statements=[statement.text for statement in statements]
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = await self._parse_json(response.get("content"), {"triplets": []})
        return _ExtractedTriplets.model_validate(parsed).triplets

    async def _build_graph(
        self,
        dialogue: DialogueNode,
        chunks: list[ChunkNode],
        statements: list[StatementNode],
        triplets: list[ExtractedTriplet],
    ) -> MemoryGraph:
        entity_by_key: dict[tuple[str, str], EntityNode] = {}
        relations: list[RelationEdge] = []
        mentions_by_key: dict[tuple[str, str], MentionEdge] = {}
        statement_by_text = {statement.text: statement for statement in statements}
        fallback_statement = statements[0] if statements else None

        for triplet in triplets:
            head = self._get_or_create_entity(
                dialogue.user_id, triplet.head, entity_by_key
            )
            tail = self._get_or_create_entity(
                dialogue.user_id, triplet.tail, entity_by_key
            )
            statement = statement_by_text.get(triplet.evidence, fallback_statement)
            if not statement:
                continue
            for entity in (head, tail):
                mention_key = (statement.id, entity.id)
                mentions_by_key.setdefault(
                    mention_key,
                    MentionEdge(
                        id=stable_memory_graph_id(
                            dialogue.user_id, statement.id, entity.id, "mentions"
                        ),
                        user_id=dialogue.user_id,
                        statement_id=statement.id,
                        entity_id=entity.id,
                    ),
                )
            relations.append(
                RelationEdge(
                    id=stable_memory_graph_id(
                        dialogue.user_id,
                        head.id,
                        triplet.relation,
                        tail.id,
                        statement.id,
                    ),
                    user_id=dialogue.user_id,
                    source_entity_id=head.id,
                    target_entity_id=tail.id,
                    statement_id=statement.id,
                    name=self._normalize_relation(triplet.relation),
                    evidence=triplet.evidence or statement.text,
                )
            )

        entities = list(entity_by_key.values())
        if entities:
            vectors = await self._embedding.embed(
                [
                    f"{entity.name}\n{entity.type}\n{entity.description}"
                    for entity in entities
                ]
            )
            for entity, vector in zip(entities, vectors):
                entity.embedding = vector

        return MemoryGraph(
            dialogue=dialogue,
            chunks=chunks,
            statements=statements,
            entities=entities,
            mentions=list(mentions_by_key.values()),
            relations=relations,
        )

    @classmethod
    def _get_or_create_entity(
        cls,
        user_id: str,
        extracted: ExtractedEntity,
        entity_by_key: dict[tuple[str, str], EntityNode],
    ) -> EntityNode:
        name = extracted.name.strip()
        entity_type = extracted.type.strip() or "Thing"
        key = (name.lower(), entity_type.lower())
        if key not in entity_by_key:
            entity_by_key[key] = EntityNode(
                id=stable_memory_graph_id(
                    user_id, "entity", name.lower(), entity_type.lower()
                ),
                user_id=user_id,
                name=name,
                type=entity_type,
                description=extracted.description.strip(),
            )
        return entity_by_key[key]

    async def _parse_json(self, content: Any, default_value: dict[str, Any]) -> Any:
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

    @staticmethod
    def _normalize_relation(relation: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", relation.strip().upper())
        return normalized.strip("_") or "RELATED_TO"
