import logging
from typing import Any

from app.domain.models.memory_graph import (
    EntityNode,
    GraphRelationFact,
    MemoryGraph,
    MemoryPromotionStats,
    MemoryGraphResult,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository

_VECTOR_WEIGHT = 0.55
_FULLTEXT_WEIGHT = 0.30
_IMPORTANCE_WEIGHT = 0.15
_LONG_TERM_BONUS = 0.05

logger = logging.getLogger(__name__)


class Neo4jMemoryGraphRepository(MemoryGraphRepository):
    """Neo4j 记忆图谱仓储。"""

    def __init__(self, driver: Any, database: str, embedding_dims: int) -> None:
        self._driver = driver
        self._database = database
        self._embedding_dims = embedding_dims

    async def ensure_schema(self) -> None:
        """初始化约束和索引，重复执行保持幂等。"""
        statements = [
            (
                "CREATE CONSTRAINT memory_dialogue_id IF NOT EXISTS "
                "FOR (d:Dialogue) REQUIRE (d.id, d.user_id) IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT memory_chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE (c.id, c.user_id) IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT memory_statement_id IF NOT EXISTS "
                "FOR (s:Statement) REQUIRE (s.id, s.user_id) IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT memory_entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.id, e.user_id) IS UNIQUE"
            ),
            (
                "CREATE INDEX memory_entity_name IF NOT EXISTS "
                "FOR (e:Entity) ON (e.user_id, e.name, e.type)"
            ),
            (
                "CREATE INDEX memory_entity_layer IF NOT EXISTS "
                "FOR (e:Entity) ON (e.user_id, e.memory_layer)"
            ),
            (
                "CREATE FULLTEXT INDEX memory_entity_fulltext IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name, e.description]"
            ),
            (
                "CREATE VECTOR INDEX memory_entity_embedding IF NOT EXISTS "
                "FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: "
                f"{{`vector.dimensions`: {self._embedding_dims}, "
                "`vector.similarity_function`: 'cosine'}}}"
            ),
        ]
        async with self._driver.session(database=self._database) as session:
            for statement in statements:
                await session.run(statement)

    async def save_graph(self, graph: MemoryGraph) -> None:
        """使用 MERGE 幂等写入一条记忆图谱。"""
        params = graph.model_dump(mode="python")
        params["user_id"] = graph.dialogue.user_id
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(self._merge_graph, params)

    async def list_entities_by_type(
        self, user_id: str, entity_type: str
    ) -> list[EntityNode]:
        """按用户和实体类型列出已有实体，供写图前同名融合。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id, type: $entity_type})
        RETURN entity.id AS id,
               entity.name AS name,
               entity.type AS type,
               entity.description AS description,
               coalesce(entity.importance, 0.5) AS importance,
               coalesce(entity.confidence, 0.8) AS confidence,
               coalesce(entity.mention_count, 1) AS mention_count,
               coalesce(entity.access_count, 0) AS access_count,
               entity.last_access_at AS last_access_at,
               coalesce(entity.memory_layer, 'short_term') AS memory_layer,
               coalesce(entity.core_facts, []) AS core_facts,
               coalesce(entity.traits, []) AS traits
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                entity_type=entity_type,
            )
            rows = await result.data()
            return [self._row_to_entity_node(row, user_id) for row in rows]

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryGraphResult]:
        """混合召回实体及一跳关系，Neo4j 不可用时由上层降级。"""
        fulltext_rows = await self._search_fulltext(
            user_id=user_id, query=query, top_k=top_k
        )
        vector_rows: list[dict[str, Any]] = []
        if query_embedding:
            vector_rows = await self._search_vector(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k,
            )
        results = self._fuse_rows(fulltext_rows, vector_rows, top_k)
        if results:
            try:
                await self.bump_entity_access(
                    user_id, [result.entity_id for result in results]
                )
            except Exception as e:
                logger.warning("记忆图谱命中回写失败，忽略: %s", e)
        return results

    async def bump_entity_access(self, user_id: str, entity_ids: list[str]) -> None:
        """记录实体检索命中次数和最后访问时间。"""
        if not entity_ids:
            return
        cypher = """
        MATCH (entity:Entity {user_id: $user_id})
        WHERE entity.id IN $entity_ids
        SET entity.access_count = coalesce(entity.access_count, 0) + 1,
            entity.last_access_at = datetime()
        """
        async with self._driver.session(database=self._database) as session:
            await session.run(cypher, user_id=user_id, entity_ids=entity_ids)

    async def promote_short_to_long(
        self,
        user_id: str,
        min_access: int,
        min_importance: float,
        min_mention: int,
        age_before: str,
    ) -> MemoryPromotionStats:
        """按访问、重要度、提及次数和年龄阈值把短期记忆提升为长期。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id})
        WHERE coalesce(entity.memory_layer, 'short_term') = 'short_term'
          AND (
            coalesce(entity.access_count, 0) >= $min_access
            OR coalesce(entity.importance, 0.5) >= $min_importance
            OR (
              coalesce(entity.mention_count, 1) >= $min_mention
              AND date(entity.created_at) <= date($age_before)
            )
          )
        SET entity.memory_layer = 'long_term'
        WITH DISTINCT entity
        OPTIONAL MATCH (entity)<-[:MENTIONS]-(statement:Statement {user_id: $user_id})
        SET statement.memory_layer = 'long_term'
        WITH collect(DISTINCT entity) AS promoted_entities,
             collect(DISTINCT statement) AS promoted_statements
        RETURN size(promoted_entities) AS entities,
               size([s IN promoted_statements WHERE s IS NOT NULL]) AS statements
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                min_access=min_access,
                min_importance=min_importance,
                min_mention=min_mention,
                age_before=age_before,
            )
            rows = await result.data()
        row = rows[0] if rows else {}
        return MemoryPromotionStats(
            promoted_entities=int(row.get("entities") or 0),
            promoted_statements=int(row.get("statements") or 0),
        )

    async def top_long_term_entities(
        self, user_id: str, top_k: int
    ) -> list[EntityNode]:
        """读取高价值长期实体用于画像增强。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id, memory_layer: 'long_term'})
        RETURN entity.id AS id,
               entity.name AS name,
               entity.type AS type
        ORDER BY coalesce(entity.access_count, 0) DESC,
                 coalesce(entity.mention_count, 0) DESC,
                 coalesce(entity.importance, 0.5) DESC
        LIMIT $top_k
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id, top_k=top_k)
            rows = await result.data()
        return [self._row_to_entity_node(row, user_id) for row in rows]

    async def entity_statements(self, user_id: str, entity_id: str) -> list[str]:
        """读取实体关联陈述供画像增强。"""
        cypher = """
        MATCH (entity:Entity {id: $entity_id, user_id: $user_id})
        MATCH (entity)<-[:MENTIONS]-(statement:Statement {user_id: $user_id})
        RETURN DISTINCT statement.text AS text
        ORDER BY coalesce(statement.importance, 0.5) DESC
        LIMIT 50
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id, entity_id=entity_id)
            rows = await result.data()
        return [str(row.get("text") or "").strip() for row in rows if row.get("text")]

    async def write_entity_profile(
        self,
        user_id: str,
        entity_id: str,
        core_facts: list[str],
        traits: list[str],
    ) -> None:
        """回写长期实体画像。"""
        cypher = """
        MATCH (entity:Entity {id: $entity_id, user_id: $user_id})
        SET entity.core_facts = $core_facts,
            entity.traits = $traits
        """
        async with self._driver.session(database=self._database) as session:
            await session.run(
                cypher,
                user_id=user_id,
                entity_id=entity_id,
                core_facts=core_facts,
                traits=traits,
            )

    async def _search_fulltext(
        self, user_id: str, query: str, top_k: int
    ) -> list[dict[str, Any]]:
        """通过 Neo4j 全文索引召回候选实体及其一跳上下文。"""
        cypher = """
        CALL db.index.fulltext.queryNodes('memory_entity_fulltext', $search_text)
        YIELD node, score
        WHERE node.user_id = $user_id
        WITH node, score
        """ + self._entity_context_return_clause()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                search_text=query,
                top_k=top_k,
            )
            return await result.data()

    async def _search_vector(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """通过 Neo4j 向量索引召回候选实体及其一跳上下文。"""
        cypher = """
        CALL db.index.vector.queryNodes(
            'memory_entity_embedding',
            $vector_top_k,
            $query_embedding
        )
        YIELD node, score
        WHERE node.user_id = $user_id
        WITH node, score
        """ + self._entity_context_return_clause()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                query_embedding=query_embedding,
                vector_top_k=max(top_k, 20),
                top_k=top_k,
            )
            return await result.data()

    @staticmethod
    def _entity_context_return_clause() -> str:
        """返回实体召回后补齐来源记忆和一跳关系的 Cypher 片段。"""
        return """
        ORDER BY score DESC LIMIT $top_k
        OPTIONAL MATCH (source:Entity {user_id: $user_id})-[incoming:RELATION]->(node)
        OPTIONAL MATCH (node)-[outgoing:RELATION]->(target:Entity {user_id: $user_id})
        OPTIONAL MATCH (node)<-[:MENTIONS]-(statement:Statement)
            <-[:HAS_STATEMENT]-(:Chunk)<-[:HAS_CHUNK]-(dialogue:Dialogue)
        WITH node, score, dialogue,
             collect(DISTINCT {
                name: incoming.name,
                direction: 'incoming',
                neighbor_name: source.name,
                neighbor_type: source.type,
                evidence: incoming.evidence
             }) +
             collect(DISTINCT {
                name: outgoing.name,
                direction: 'outgoing',
                neighbor_name: target.name,
                neighbor_type: target.type,
                evidence: outgoing.evidence
             }) AS relations
        RETURN {
            id: node.id,
            name: node.name,
            type: node.type,
            description: node.description,
            importance: coalesce(node.importance, 0.5),
            memory_layer: coalesce(node.memory_layer, 'short_term'),
            core_facts: coalesce(node.core_facts, []),
            traits: coalesce(node.traits, []),
            access_count: coalesce(node.access_count, 0),
            mention_count: coalesce(node.mention_count, 0),
            memory_id: dialogue.memory_id,
            memory_summary: coalesce(dialogue.summary, '')
        } AS entity,
        [relation IN relations WHERE relation.name IS NOT NULL] AS relations,
        score
        """

    def _fuse_rows(
        self,
        fulltext_rows: list[dict[str, Any]],
        vector_rows: list[dict[str, Any]],
        top_k: int,
    ) -> list[MemoryGraphResult]:
        """融合全文和向量召回结果，按归一化加权分数排序。"""
        fulltext_results = [self._row_to_search_result(row) for row in fulltext_rows]
        vector_results = [self._row_to_search_result(row) for row in vector_rows]
        fulltext_scores = {
            result.entity_id: result.score for result in fulltext_results
        }
        vector_scores = {result.entity_id: result.score for result in vector_results}
        fulltext_norm = self._normalize_scores(fulltext_scores)
        vector_norm = self._normalize_scores(vector_scores)
        by_entity: dict[str, MemoryGraphResult] = {}
        for result in [*fulltext_results, *vector_results]:
            existing = by_entity.get(result.entity_id)
            if not existing or result.score > existing.score:
                by_entity[result.entity_id] = result
        for entity_id, result in by_entity.items():
            result.score = (
                _VECTOR_WEIGHT * vector_norm.get(entity_id, 0)
                + _FULLTEXT_WEIGHT * fulltext_norm.get(entity_id, 0)
                + _IMPORTANCE_WEIGHT * result.importance
            )
            if result.memory_layer == "long_term":
                result.score += _LONG_TERM_BONUS
        return sorted(
            by_entity.values(),
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """把一组原始召回分数归一化到 0 到 1 区间。"""
        if not scores:
            return {}
        values = list(scores.values())
        low = min(values)
        high = max(values)
        if high - low < 1e-9:
            return {key: 1.0 for key in scores}
        return {key: (value - low) / (high - low) for key, value in scores.items()}

    @staticmethod
    async def _merge_graph(tx, params: dict[str, Any]) -> None:
        """将完整记忆图谱以 MERGE 方式幂等写入 Neo4j。"""
        await tx.run(
            """
            MERGE (d:Dialogue {id: $dialogue_id, user_id: $user_id})
            SET d.memory_id = $dialogue.memory_id,
                d.summary = $dialogue.summary,
                d.created_at = $dialogue.created_at
            WITH d
            UNWIND $chunks AS row
            MERGE (chunk:Chunk {id: row.id, user_id: row.user_id})
            SET chunk.index = row.index,
                chunk.text = row.text
            MERGE (d)-[:HAS_CHUNK]->(chunk)
            WITH d
            UNWIND $statements AS row
            MATCH (chunk:Chunk {id: row.chunk_id, user_id: row.user_id})
            MERGE (statement:Statement {id: row.id, user_id: row.user_id})
            SET statement.index = row.index,
                statement.text = row.text,
                statement.statement_type = row.statement_type,
                statement.temporal_type = row.temporal_type,
                statement.importance = row.importance,
                statement.confidence = row.confidence,
                statement.access_count = coalesce(statement.access_count, row.access_count),
                statement.last_access_at = coalesce(statement.last_access_at, row.last_access_at),
                statement.memory_layer = CASE
                    WHEN coalesce(statement.memory_layer, row.memory_layer) = 'long_term'
                    THEN 'long_term'
                    ELSE row.memory_layer
                END
            MERGE (chunk)-[:HAS_STATEMENT]->(statement)
            WITH d
            UNWIND $entities AS row
            MERGE (entity:Entity {id: row.id, user_id: row.user_id})
            SET entity.name = row.name,
                entity.type = row.type,
                entity.description = row.description,
                entity.embedding = row.embedding,
                entity.importance = CASE
                    WHEN coalesce(entity.importance, 0.0) > row.importance
                    THEN entity.importance
                    ELSE row.importance
                END,
                entity.confidence = CASE
                    WHEN coalesce(entity.confidence, 0.0) > row.confidence
                    THEN entity.confidence
                    ELSE row.confidence
                END,
                entity.mention_count = CASE
                    WHEN coalesce(entity.mention_count, 0) > row.mention_count
                    THEN entity.mention_count
                    ELSE row.mention_count
                END,
                entity.access_count = coalesce(entity.access_count, row.access_count),
                entity.last_access_at = coalesce(entity.last_access_at, row.last_access_at),
                entity.memory_layer = CASE
                    WHEN coalesce(entity.memory_layer, row.memory_layer) = 'long_term'
                    THEN 'long_term'
                    ELSE row.memory_layer
                END,
                entity.core_facts = coalesce(entity.core_facts, row.core_facts),
                entity.traits = coalesce(entity.traits, row.traits),
                entity.created_at = coalesce(entity.created_at, datetime())
            WITH d
            UNWIND $mentions AS row
            MATCH (statement:Statement {id: row.statement_id, user_id: row.user_id})
            MATCH (entity:Entity {id: row.entity_id, user_id: row.user_id})
            MERGE (statement)-[:MENTIONS {id: row.id, user_id: row.user_id}]->(entity)
            WITH d
            UNWIND $relations AS row
            MATCH (source:Entity {id: row.source_entity_id, user_id: row.user_id})
            MATCH (target:Entity {id: row.target_entity_id, user_id: row.user_id})
            MATCH (statement:Statement {id: row.statement_id, user_id: row.user_id})
            MERGE (source)-[rel:RELATION {id: row.id, user_id: row.user_id}]->(target)
            SET rel.name = row.name,
                rel.evidence = row.evidence,
                rel.statement_id = statement.id,
                rel.importance = row.importance,
                rel.confidence = row.confidence,
                rel.access_count = coalesce(rel.access_count, row.access_count),
                rel.last_access_at = coalesce(rel.last_access_at, row.last_access_at),
                rel.memory_layer = CASE
                    WHEN coalesce(rel.memory_layer, row.memory_layer) = 'long_term'
                    THEN 'long_term'
                    ELSE row.memory_layer
                END
            """,
            dialogue_id=params["dialogue"]["id"],
            user_id=params["user_id"],
            dialogue=params["dialogue"],
            chunks=params["chunks"],
            statements=params["statements"],
            entities=params["entities"],
            mentions=params["mentions"],
            relations=params["relations"],
        )

    @staticmethod
    def _row_to_search_result(row: dict[str, Any]) -> MemoryGraphResult:
        """把 Neo4j 原始查询行转换为领域检索结果模型。"""
        entity = row.get("entity") or {}
        relations = [
            GraphRelationFact.model_validate(relation)
            for relation in row.get("relations", [])
            if relation.get("neighbor_name")
        ]
        return MemoryGraphResult(
            entity_id=entity.get("id", ""),
            entity_name=entity.get("name", ""),
            entity_type=entity.get("type", ""),
            description=entity.get("description") or "",
            importance=float(entity.get("importance") or 0.5),
            memory_layer=entity.get("memory_layer") or "short_term",
            core_facts=entity.get("core_facts") or [],
            traits=entity.get("traits") or [],
            access_count=int(entity.get("access_count") or 0),
            mention_count=int(entity.get("mention_count") or 0),
            source_memory_id=entity.get("memory_id"),
            source_memory_summary=entity.get("memory_summary") or None,
            relations=relations,
            score=float(row.get("score") or 0),
        )

    @staticmethod
    def _row_to_entity_node(row: dict[str, Any], user_id: str) -> EntityNode:
        """把 Neo4j 实体行转换为领域实体节点模型。"""
        return EntityNode(
            id=str(row.get("id") or ""),
            user_id=user_id,
            name=str(row.get("name") or ""),
            type=str(row.get("type") or ""),
            description=str(row.get("description") or ""),
            importance=float(row.get("importance") or 0.5),
            confidence=float(row.get("confidence") or 0.8),
            mention_count=int(row.get("mention_count") or 1),
            access_count=int(row.get("access_count") or 0),
            last_access_at=row.get("last_access_at"),
            memory_layer=str(row.get("memory_layer") or "short_term"),
            core_facts=row.get("core_facts") or [],
            traits=row.get("traits") or [],
        )
