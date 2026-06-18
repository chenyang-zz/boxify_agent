from datetime import datetime

from app.domain.external.embedding import EmbeddingModel
from app.domain.models.memory_graph import (
    ChunkNode,
    DialogueNode,
    EntityNode,
    EventNode,
    InvolvesEdge,
    MemoryGraph,
    MemoryGraphStats,
    MentionEdge,
    RelationEdge,
    StatementNode,
    stable_memory_graph_id,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository
from app.domain.services.memory.fact_extractor import (
    ExtractedEntity,
    ExtractedTriplets,
    MemoryFactExtractor,
)
from app.domain.services.memory.entity_deduplicator import MemoryEntityDeduplicator
from app.domain.services.memory.ontology import normalize_entity_type, normalize_predicate
from app.utils.datetime import parse_optional_datetime


class MemoryGraphExtractor:
    """把一段长期记忆萃取并写入四层溯源图谱。"""

    def __init__(
        self,
        fact_extractor: MemoryFactExtractor,
        embedding: EmbeddingModel,
        graph_repository: MemoryGraphRepository,
        deduplicator: MemoryEntityDeduplicator | None = None,
        chunk_size: int = 1200,
    ) -> None:
        self._fact_extractor = fact_extractor
        self._embedding = embedding
        self._graph_repository = graph_repository
        self._deduplicator = deduplicator
        self._chunk_size = chunk_size

    async def extract_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        dialog_at: datetime | None = None,
    ) -> MemoryGraphStats:
        """萃取一条 PG 记忆并写入 Neo4j 图谱。"""
        dialog_at = dialog_at or datetime.now()
        dialogue = DialogueNode(
            id=stable_memory_graph_id(user_id, memory_id, "dialogue"),
            user_id=user_id,
            memory_id=memory_id,
            summary=content.strip(),
            created_at=dialog_at,
        )
        chunks = self._chunk_text(user_id, dialogue.id, content)
        dialog_at_text = dialog_at.isoformat()
        statements = await self._fact_extractor.extract_statements(
            chunks,
            dialog_at=dialog_at_text,
        )
        triplets = await self._fact_extractor.extract_triplets(
            statements,
            dialog_at=dialog_at_text,
        )
        graph = await self._build_graph(dialogue, chunks, statements, triplets)
        await self._graph_repository.save_graph(graph)
        return graph.stats()

    def _chunk_text(
        self, user_id: str, dialogue_id: str, content: str
    ) -> list[ChunkNode]:
        """将记忆正文按固定窗口切成 chunk，保留用户和对话归属。"""
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

    async def _build_graph(
        self,
        dialogue: DialogueNode,
        chunks: list[ChunkNode],
        statements: list[StatementNode],
        triplet_result: ExtractedTriplets,
    ) -> MemoryGraph:
        """根据陈述和三元组组装四层溯源图谱。"""
        entity_by_idx = await self._build_entities(dialogue.user_id, triplet_result.entities)
        relations: list[RelationEdge] = []
        mentions_by_key: dict[tuple[str, str], MentionEdge] = {}
        statement_by_text = {statement.text: statement for statement in statements}
        fallback_statement = statements[0] if statements else None

        for triplet in triplet_result.triplets:
            head = entity_by_idx.get(triplet.subject_id)
            tail = entity_by_idx.get(triplet.object_id)
            if not head or not tail:
                continue
            statement = statement_by_text.get(triplet.evidence, fallback_statement)
            if not statement:
                continue
            valid_at = parse_optional_datetime(triplet.valid_at)
            invalid_at = parse_optional_datetime(triplet.invalid_at)
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
                        triplet.predicate,
                        tail.id,
                        statement.id,
                    ),
                    user_id=dialogue.user_id,
                    source_entity_id=head.id,
                    target_entity_id=tail.id,
                    statement_id=statement.id,
                    name=normalize_predicate(triplet.predicate),
                    evidence=triplet.evidence or statement.text,
                    importance=triplet.importance,
                    confidence=triplet.confidence,
                    valid_at=valid_at or statement.valid_at,
                    invalid_at=invalid_at or statement.invalid_at,
                )
            )

        entities = list({entity.id: entity for entity in entity_by_idx.values()}.values())
        events, involves = self._build_events(
            dialogue=dialogue,
            entities=entities,
            entity_by_idx=entity_by_idx,
            triplet_result=triplet_result,
        )
        entities_needing_embedding = [
            entity for entity in entities if not entity.embedding
        ]
        if entities_needing_embedding:
            vectors = await self._embedding.embed(
                [
                    f"{entity.name}\n{entity.type}\n{entity.description}"
                    for entity in entities_needing_embedding
                ]
            )
            for entity, vector in zip(entities_needing_embedding, vectors):
                entity.embedding = vector

        return MemoryGraph(
            dialogue=dialogue,
            chunks=chunks,
            statements=statements,
            entities=entities,
            mentions=list(mentions_by_key.values()),
            relations=relations,
            events=events,
            involves=involves,
        )

    def _build_events(
        self,
        dialogue: DialogueNode,
        entities: list[EntityNode],
        entity_by_idx: dict[int, EntityNode],
        triplet_result: ExtractedTriplets,
    ) -> tuple[list[EventNode], list[InvolvesEdge]]:
        """把 LLM 事件输出转换为 Event 节点和 INVOLVES 边。"""
        entity_by_name = {
            entity.name.strip(): entity
            for entity in entities
            if entity.name.strip()
        }
        for extracted in triplet_result.entities:
            entity = entity_by_idx.get(extracted.entity_idx)
            name = extracted.name.strip()
            if entity and name:
                entity_by_name[name] = entity
        events: list[EventNode] = []
        involves: list[InvolvesEdge] = []
        for extracted in triplet_result.events:
            title = extracted.title.strip()
            if not title:
                continue
            event_time = parse_optional_datetime(extracted.event_time)
            event = EventNode(
                id=stable_memory_graph_id(
                    dialogue.user_id,
                    dialogue.id,
                    "event",
                    title,
                    event_time.isoformat() if event_time else "",
                    extracted.description.strip(),
                ),
                user_id=dialogue.user_id,
                dialogue_id=dialogue.id,
                title=title,
                description=extracted.description.strip(),
                event_time=event_time,
                created_at=dialogue.created_at,
            )
            linked_entity_ids: set[str] = set()
            for participant in extracted.participants:
                entity = entity_by_name.get(participant.strip())
                if not entity or entity.id in linked_entity_ids:
                    continue
                linked_entity_ids.add(entity.id)
                involves.append(
                    InvolvesEdge(
                        id=stable_memory_graph_id(
                            dialogue.user_id,
                            event.id,
                            entity.id,
                            "involves",
                        ),
                        user_id=dialogue.user_id,
                        event_id=event.id,
                        entity_id=entity.id,
                    )
                )
            events.append(event)
        return events, involves

    async def _build_entities(
        self, user_id: str, extracted_entities: list[ExtractedEntity]
    ) -> dict[int, EntityNode]:
        """归一化并去重本次抽取实体，再与图数据库已有实体融合。"""
        if self._deduplicator:
            entity_by_idx = self._build_raw_entities(user_id, extracted_entities)
            await self._embed_entities(list(entity_by_idx.values()))
            batch_result = await self._deduplicator.dedup_batch(entity_by_idx)
            graph_result = await self._deduplicator.merge_with_graph(
                user_id,
                batch_result.entity_by_idx,
                self._graph_repository,
            )
            return graph_result.entity_by_idx

        entity_by_key: dict[tuple[str, str], EntityNode] = {}
        entity_by_idx: dict[int, EntityNode] = {}
        for extracted in extracted_entities:
            name = extracted.name.strip()
            if not name:
                continue
            entity_type = normalize_entity_type(extracted.type)
            key = (name.lower(), entity_type)
            if key not in entity_by_key:
                entity_by_key[key] = EntityNode(
                    id=stable_memory_graph_id(
                        user_id,
                        "entity",
                        name.lower(),
                        entity_type.lower(),
                    ),
                    user_id=user_id,
                    name=name,
                    type=entity_type,
                    description=extracted.description.strip(),
                    importance=extracted.importance,
                    confidence=extracted.confidence,
                )
            entity_by_idx[extracted.entity_idx] = entity_by_key[key]

        await self._merge_entities_with_graph(user_id, list(entity_by_key.values()))
        return entity_by_idx

    def _build_raw_entities(
        self, user_id: str, extracted_entities: list[ExtractedEntity]
    ) -> dict[int, EntityNode]:
        """构造保留原始 entity_idx 的实体映射，供模糊消歧重定向。"""
        entity_by_idx: dict[int, EntityNode] = {}
        for extracted in extracted_entities:
            name = extracted.name.strip()
            if not name:
                continue
            entity_type = normalize_entity_type(extracted.type)
            entity_by_idx[extracted.entity_idx] = EntityNode(
                id=stable_memory_graph_id(
                    user_id,
                    "entity",
                    name.lower(),
                    entity_type.lower(),
                ),
                user_id=user_id,
                name=name,
                type=entity_type,
                description=extracted.description.strip(),
                importance=extracted.importance,
                confidence=extracted.confidence,
            )
        return entity_by_idx

    async def _embed_entities(self, entities: list[EntityNode]) -> None:
        """为实体生成 embedding；同一 ID 只生成一次，失败语义保持向外抛出。"""
        unique_entities = list({entity.id: entity for entity in entities}.values())
        entities_needing_embedding = [
            entity for entity in unique_entities if not entity.embedding
        ]
        if not entities_needing_embedding:
            return
        vectors = await self._embedding.embed(
            [
                f"{entity.name}\n{entity.type}\n{entity.description}"
                for entity in entities_needing_embedding
            ]
        )
        for entity, vector in zip(entities_needing_embedding, vectors):
            entity.embedding = vector

    async def _merge_entities_with_graph(
        self, user_id: str, entities: list[EntityNode]
    ) -> None:
        """按实体类型查询已有节点，复用同名实体 ID 和更完整描述。"""
        cache: dict[str, list[EntityNode]] = {}
        for entity in entities:
            if entity.type not in cache:
                cache[entity.type] = await self._graph_repository.list_entities_by_type(
                    user_id, entity.type
                )
            norm_name = entity.name.strip().lower()
            existing = next(
                (
                    existing_entity
                    for existing_entity in cache[entity.type]
                    if existing_entity.name.strip().lower() == norm_name
                ),
                None,
            )
            if not existing:
                continue
            entity.id = existing.id
            entity.mention_count = existing.mention_count + 1
            entity.access_count = existing.access_count
            entity.last_access_at = existing.last_access_at
            entity.memory_layer = existing.memory_layer
            entity.core_facts = existing.core_facts
            entity.traits = existing.traits
            entity.importance = max(entity.importance, existing.importance)
            entity.confidence = max(entity.confidence, existing.confidence)
            old_description = existing.description
            if len(old_description) > len(entity.description):
                entity.description = old_description
