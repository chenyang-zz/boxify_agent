from app.domain.external.embedding import EmbeddingModel
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
from app.domain.services.memory.fact_extractor import (
    ExtractedEntity,
    ExtractedTriplets,
    MemoryFactExtractor,
)
from app.domain.services.memory.ontology import normalize_entity_type, normalize_predicate


class MemoryGraphExtractor:
    """把一段长期记忆萃取并写入四层溯源图谱。"""

    def __init__(
        self,
        fact_extractor: MemoryFactExtractor,
        embedding: EmbeddingModel,
        graph_repository: MemoryGraphRepository,
        chunk_size: int = 1200,
    ) -> None:
        self._fact_extractor = fact_extractor
        self._embedding = embedding
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
        statements = await self._fact_extractor.extract_statements(chunks)
        triplets = await self._fact_extractor.extract_triplets(statements)
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
                )
            )

        entities = list({entity.id: entity for entity in entity_by_idx.values()}.values())
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

    async def _build_entities(
        self, user_id: str, extracted_entities: list[ExtractedEntity]
    ) -> dict[int, EntityNode]:
        """归一化并去重本次抽取实体，再与图数据库已有实体融合。"""
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
