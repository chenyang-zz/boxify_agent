import logging
from typing import Any

from app.domain.models.memory_graph import (
    CommunityMemberResult,
    CommunityRelationResult,
    CommunityResult,
    CommunityVoteEntity,
    CommunityVoteNeighbor,
    EntityNode,
    GraphRelationFact,
    InsightResult,
    MemoryEntitySubgraphResult,
    MemoryGraph,
    MemoryGraphEdgeResult,
    MemoryGraphNodeResult,
    MemoryGraphResult,
    MemoryMergeDuplicatesResult,
    MemoryProfileEntityResult,
    MemoryProfileRelationResult,
    MemoryPromotionStats,
    MemoryQualityGraphCountsResult,
    MemoryQualityIssueListResult,
    MemoryQualityIssueResult,
    MemoryQualityIssueSummaryResult,
    MemoryRelationHistoryResult,
    MemoryTimelineEventResult,
    MemoryTimelineParticipantResult,
    stable_memory_graph_id,
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
                "CREATE CONSTRAINT memory_event_id IF NOT EXISTS "
                "FOR (e:Event) REQUIRE (e.id, e.user_id) IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT memory_insight_id IF NOT EXISTS "
                "FOR (i:Insight) REQUIRE (i.id, i.user_id) IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT memory_community_id IF NOT EXISTS "
                "FOR (c:Community) REQUIRE (c.id, c.user_id) IS UNIQUE"
            ),
            (
                "CREATE INDEX memory_entity_name IF NOT EXISTS "
                "FOR (e:Entity) ON (e.user_id, e.name, e.type)"
            ),
            (
                "CREATE INDEX memory_insight_theme IF NOT EXISTS "
                "FOR (i:Insight) ON (i.user_id, i.theme)"
            ),
            (
                "CREATE INDEX memory_entity_layer IF NOT EXISTS "
                "FOR (e:Entity) ON (e.user_id, e.memory_layer)"
            ),
            (
                "CREATE INDEX memory_event_user_time IF NOT EXISTS "
                "FOR (e:Event) ON (e.user_id, e.event_time)"
            ),
            (
                "CREATE INDEX memory_community_user IF NOT EXISTS "
                "FOR (c:Community) ON (c.user_id, c.id)"
            ),
            (
                "CREATE FULLTEXT INDEX memory_entity_fulltext IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name, e.description]"
            ),
            (
                "CREATE FULLTEXT INDEX memory_insight_fulltext IF NOT EXISTS "
                "FOR (i:Insight) ON EACH [i.theme, i.content]"
            ),
            (
                "CREATE VECTOR INDEX memory_entity_embedding IF NOT EXISTS "
                "FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: "
                f"{{`vector.dimensions`: {self._embedding_dims}, "
                "`vector.similarity_function`: 'cosine'}}"
            ),
            (
                "CREATE VECTOR INDEX memory_insight_embedding IF NOT EXISTS "
                "FOR (i:Insight) ON (i.embedding) OPTIONS {indexConfig: "
                f"{{`vector.dimensions`: {self._embedding_dims}, "
                "`vector.similarity_function`: 'cosine'}}"
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
               coalesce(entity.traits, []) AS traits,
               coalesce(entity.embedding, []) AS embedding
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

    async def reflection_top_entities(
        self, user_id: str, top_k: int
    ) -> list[EntityNode]:
        """读取反思使用的长期实体。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id, memory_layer: 'long_term'})
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
        ORDER BY coalesce(entity.access_count, 0) DESC,
                 coalesce(entity.mention_count, 0) DESC,
                 coalesce(entity.importance, 0.5) DESC
        LIMIT $top_k
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id, top_k=top_k)
            rows = await result.data()
        return [self._row_to_entity_node(row, user_id) for row in rows]

    async def reflection_entity_statements(
        self, user_id: str, entity_id: str, limit: int
    ) -> list[str]:
        """读取反思使用的实体代表性陈述。"""
        cypher = """
        MATCH (entity:Entity {id: $entity_id, user_id: $user_id})
        MATCH (entity)<-[:MENTIONS]-(statement:Statement {user_id: $user_id})
        RETURN DISTINCT statement.text AS text
        ORDER BY coalesce(statement.importance, 0.5) DESC
        LIMIT $limit
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher, user_id=user_id, entity_id=entity_id, limit=limit
            )
            rows = await result.data()
        return [str(row.get("text") or "").strip() for row in rows if row.get("text")]

    async def upsert_insight(
        self,
        user_id: str,
        theme: str,
        content: str,
        embedding: list[float] | None,
        importance: float,
        confidence: float,
        source_count: int,
        entity_ids: list[str],
    ) -> str:
        """按主题 upsert 洞察并重建 DERIVED_FROM 溯源边。"""
        insight_id = stable_memory_graph_id(user_id, "insight", theme.strip().lower())
        cypher = """
        MERGE (insight:Insight {user_id: $user_id, theme: $theme})
        ON CREATE SET insight.id = $insight_id,
                      insight.created_at = datetime()
        SET insight.content = $content,
            insight.embedding = $embedding,
            insight.importance = $importance,
            insight.confidence = $confidence,
            insight.source_count = $source_count,
            insight.updated_at = datetime()
        WITH insight
        OPTIONAL MATCH (insight)-[derived:DERIVED_FROM]->()
        DELETE derived
        WITH insight
        UNWIND $entity_ids AS entity_id
        MATCH (entity:Entity {id: entity_id, user_id: $user_id})
        MERGE (insight)-[:DERIVED_FROM {user_id: $user_id}]->(entity)
        RETURN insight.id AS id
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                theme=theme,
                insight_id=insight_id,
                content=content,
                embedding=embedding or [],
                importance=importance,
                confidence=confidence,
                source_count=source_count,
                entity_ids=entity_ids,
            )
            rows = await result.data()
        return str((rows[0] if rows else {}).get("id") or insight_id)

    async def search_insights_by_vector(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[InsightResult]:
        """按当前用户精确计算向量相似度召回洞察。"""
        cypher = """
        MATCH (insight:Insight {user_id: $user_id})
        WHERE insight.embedding IS NOT NULL
          AND size(insight.embedding) = size($query_embedding)
        WITH insight,
             vector.similarity.cosine(insight.embedding, $query_embedding) AS score
        RETURN insight.id AS id,
               insight.theme AS theme,
               insight.content AS content,
               coalesce(insight.importance, 0.6) AS importance,
               coalesce(insight.confidence, 0.7) AS confidence,
               coalesce(insight.source_count, 0) AS source_count,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            rows = await result.data()
        return [self._row_to_insight_result(row) for row in rows]

    async def list_insights(self, user_id: str) -> list[InsightResult]:
        """列出用户洞察。"""
        cypher = """
        MATCH (insight:Insight {user_id: $user_id})
        RETURN insight.id AS id,
               insight.theme AS theme,
               insight.content AS content,
               coalesce(insight.importance, 0.6) AS importance,
               coalesce(insight.confidence, 0.7) AS confidence,
               coalesce(insight.source_count, 0) AS source_count,
               0.0 AS score
        ORDER BY coalesce(insight.updated_at, insight.created_at) DESC
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return [self._row_to_insight_result(row) for row in rows]

    async def count_insights(self, user_id: str) -> int:
        """统计用户洞察数量。"""
        cypher = """
        MATCH (insight:Insight {user_id: $user_id})
        RETURN count(insight) AS count
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return int((rows[0] if rows else {}).get("count") or 0)

    async def delete_insight(self, user_id: str, insight_id: str) -> bool:
        """删除当前用户单条洞察，返回是否删除成功。"""
        cypher = """
        OPTIONAL MATCH (insight:Insight {id: $insight_id, user_id: $user_id})
        WITH insight, CASE WHEN insight IS NULL THEN 0 ELSE 1 END AS deleted
        FOREACH (_ IN CASE WHEN insight IS NULL THEN [] ELSE [1] END |
            DETACH DELETE insight
        )
        RETURN deleted
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                insight_id=insight_id,
            )
            rows = await result.data()
        return int((rows[0] if rows else {}).get("deleted") or 0) > 0

    async def event_timeline(
        self, user_id: str, limit: int
    ) -> list[MemoryTimelineEventResult]:
        """读取当前用户事件时间线。"""
        cypher = """
        MATCH (event:Event {user_id: $user_id})
        OPTIONAL MATCH (event)-[:INVOLVES {user_id: $user_id}]->
            (entity:Entity {user_id: $user_id})
        WITH event,
             collect({
                 entity_id: entity.id,
                 name: entity.name,
                 type: entity.type
             }) AS raw_participants,
             CASE WHEN event.event_time IS NULL THEN 0 ELSE 1 END AS event_has_time
        RETURN event.id AS id,
               event.title AS title,
               event.description AS description,
               toString(event.event_time) AS event_time,
               toString(event.created_at) AS created_at,
               [p IN raw_participants WHERE p.entity_id IS NOT NULL] AS participants
        ORDER BY event_has_time DESC,
                 coalesce(event.event_time, event.created_at) DESC
        LIMIT $limit
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id, limit=limit)
            rows = await result.data()
        return [self._row_to_timeline_event(row) for row in rows]

    async def has_communities(self, user_id: str) -> bool:
        """判断当前用户是否已有社区。"""
        cypher = """
        MATCH (community:Community {user_id: $user_id})
        RETURN count(community) AS count
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return int((rows[0] if rows else {}).get("count") or 0) > 0

    async def dialogue_entity_ids(self, user_id: str, dialogue_id: str) -> list[str]:
        """读取一次记忆萃取关联到的实体 ID。"""
        cypher = """
        MATCH (dialogue:Dialogue {id: $dialogue_id, user_id: $user_id})
            -[:HAS_CHUNK]->(:Chunk)
            -[:HAS_STATEMENT]->(:Statement)
            -[:MENTIONS]->(entity:Entity {user_id: $user_id})
        RETURN DISTINCT entity.id AS id
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                dialogue_id=dialogue_id,
            )
            rows = await result.data()
        return [str(row.get("id")) for row in rows if row.get("id")]

    async def community_vote_entities(
        self, user_id: str, entity_ids: list[str] | None = None
    ) -> list[CommunityVoteEntity]:
        """读取社区聚类投票所需实体。"""
        where_clause = (
            "WHERE entity.id IN $entity_ids" if entity_ids is not None else ""
        )
        cypher = (
            """
        MATCH (entity:Entity {user_id: $user_id})
        """
            + where_clause
            + """
        RETURN entity.id AS id,
               entity.name AS name,
               entity.type AS type,
               entity.description AS description,
               coalesce(entity.embedding, []) AS embedding,
               entity.community_id AS community_id
        """
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                entity_ids=entity_ids,
            )
            rows = await result.data()
        return [self._row_to_community_vote_entity(row, user_id) for row in rows]

    async def community_vote_neighbors(
        self, user_id: str, entity_ids: list[str]
    ) -> dict[str, list[CommunityVoteNeighbor]]:
        """读取实体一跳邻居及其社区标签。"""
        if not entity_ids:
            return {}
        cypher = """
        MATCH (entity:Entity {user_id: $user_id})
        WHERE entity.id IN $entity_ids
        OPTIONAL MATCH (entity)-[relation:RELATION]-(neighbor:Entity {user_id: $user_id})
        WITH entity, neighbor
        WHERE neighbor IS NOT NULL
          AND (relation.invalid_at IS NULL OR relation.invalid_at > datetime())
        RETURN entity.id AS entity_id,
               neighbor.id AS id,
               neighbor.community_id AS community_id,
               coalesce(neighbor.embedding, []) AS embedding
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                entity_ids=entity_ids,
            )
            rows = await result.data()
        grouped: dict[str, list[CommunityVoteNeighbor]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("entity_id")), []).append(
                self._row_to_community_vote_neighbor(row)
            )
        return grouped

    async def upsert_community(self, user_id: str, community_id: str) -> None:
        """创建或保留社区节点。"""
        cypher = """
        MERGE (community:Community {id: $community_id, user_id: $user_id})
        ON CREATE SET community.name = $community_id,
                      community.summary = '',
                      community.member_count = 0,
                      community.created_at = datetime()
        SET community.updated_at = datetime()
        """
        async with self._driver.session(database=self._database) as session:
            await session.run(cypher, user_id=user_id, community_id=community_id)

    async def assign_entity_community(
        self, user_id: str, entity_id: str, community_id: str
    ) -> None:
        """将实体归入社区，并维护 IN_COMMUNITY 边。"""
        cypher = """
        MATCH (entity:Entity {id: $entity_id, user_id: $user_id})
        MATCH (community:Community {id: $community_id, user_id: $user_id})
        OPTIONAL MATCH (entity)-[old:IN_COMMUNITY {user_id: $user_id}]->(:Community)
        DELETE old
        SET entity.community_id = $community_id
        MERGE (entity)-[:IN_COMMUNITY {user_id: $user_id}]->(community)
        """
        async with self._driver.session(database=self._database) as session:
            await session.run(
                cypher,
                user_id=user_id,
                entity_id=entity_id,
                community_id=community_id,
            )

    async def refresh_community_member_count(
        self, user_id: str, community_id: str
    ) -> int:
        """刷新并返回社区成员数。"""
        cypher = """
        MATCH (community:Community {id: $community_id, user_id: $user_id})
        OPTIONAL MATCH (entity:Entity {user_id: $user_id, community_id: $community_id})
        WITH community, count(entity) AS count
        SET community.member_count = count,
            community.updated_at = datetime()
        RETURN count AS count
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                community_id=community_id,
            )
            rows = await result.data()
        return int((rows[0] if rows else {}).get("count") or 0)

    async def community_members(
        self, user_id: str, community_id: str
    ) -> list[CommunityMemberResult]:
        """读取社区成员实体。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id, community_id: $community_id})
        RETURN entity.id AS entity_id,
               entity.name AS entity_name,
               entity.type AS entity_type,
               entity.description AS description,
               entity.community_id AS community_id,
               coalesce(entity.embedding, []) AS embedding,
               coalesce(entity.importance, 0.5) AS importance,
               coalesce(entity.mention_count, 0) AS mention_count,
               coalesce(entity.access_count, 0) AS access_count
        ORDER BY coalesce(entity.importance, 0.5) DESC,
                 coalesce(entity.mention_count, 0) DESC
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                community_id=community_id,
            )
            rows = await result.data()
        return [self._row_to_community_member(row) for row in rows]

    async def community_relationships(
        self, user_id: str, community_id: str
    ) -> list[CommunityRelationResult]:
        """读取社区内部关系事实。"""
        cypher = """
        MATCH (source:Entity {user_id: $user_id, community_id: $community_id})
            -[relation:RELATION]->
            (target:Entity {user_id: $user_id, community_id: $community_id})
        WHERE relation.invalid_at IS NULL OR relation.invalid_at > datetime()
        RETURN source.id AS source_entity_id,
               source.name AS source_name,
               target.id AS target_entity_id,
               target.name AS target_name,
               relation.name AS name,
               relation.evidence AS evidence,
               toString(relation.valid_at) AS valid_at,
               toString(relation.invalid_at) AS invalid_at,
               relation.invalid_at IS NULL OR relation.invalid_at > datetime()
                   AS is_current
        ORDER BY coalesce(relation.importance, 0.5) DESC
        LIMIT 50
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                community_id=community_id,
            )
            rows = await result.data()
        return [self._row_to_community_relation(row) for row in rows]

    async def update_community_metadata(
        self, user_id: str, community_id: str, name: str, summary: str
    ) -> None:
        """更新社区名称和摘要。"""
        cypher = """
        MATCH (community:Community {id: $community_id, user_id: $user_id})
        SET community.name = $name,
            community.summary = $summary,
            community.updated_at = datetime()
        """
        async with self._driver.session(database=self._database) as session:
            await session.run(
                cypher,
                user_id=user_id,
                community_id=community_id,
                name=name,
                summary=summary,
            )

    async def list_communities(self, user_id: str) -> list[CommunityResult]:
        """列出当前用户社区。"""
        cypher = """
        MATCH (community:Community {user_id: $user_id})
        RETURN community.id AS id,
               community.name AS name,
               community.summary AS summary,
               coalesce(community.member_count, 0) AS member_count
        ORDER BY coalesce(community.member_count, 0) DESC,
                 community.name ASC
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return [self._row_to_community_result(row) for row in rows]

    async def prune_empty_communities(self, user_id: str) -> None:
        """删除当前用户空社区。"""
        cypher = """
        MATCH (community:Community {user_id: $user_id})
        WHERE coalesce(community.member_count, 0) = 0
        DETACH DELETE community
        """
        async with self._driver.session(database=self._database) as session:
            await session.run(cypher, user_id=user_id)

    async def graph_nodes(self, user_id: str) -> list[MemoryGraphNodeResult]:
        """读取当前用户实体关系全图节点。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id})
        RETURN entity.id AS id,
               entity.name AS name,
               entity.type AS type,
               entity.description AS description,
               entity.community_id AS community_id,
               coalesce(entity.importance, 0.5) AS importance,
               coalesce(entity.memory_layer, 'short_term') AS memory_layer,
               coalesce(entity.access_count, 0) AS access_count,
               coalesce(entity.mention_count, 0) AS mention_count,
               coalesce(entity.core_facts, []) AS core_facts,
               coalesce(entity.traits, []) AS traits
        ORDER BY coalesce(entity.importance, 0.5) DESC,
                 entity.name ASC
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return [self._row_to_graph_node(row) for row in rows]

    async def graph_edges(self, user_id: str) -> list[MemoryGraphEdgeResult]:
        """读取当前用户实体关系全图边。"""
        cypher = """
        MATCH (source:Entity {user_id: $user_id})
            -[relation:RELATION {user_id: $user_id}]->
            (target:Entity {user_id: $user_id})
        WHERE relation.invalid_at IS NULL OR relation.invalid_at > datetime()
        RETURN source.id AS source,
               target.id AS target,
               relation.name AS predicate,
               relation.evidence AS evidence,
               toString(relation.valid_at) AS valid_at,
               toString(relation.invalid_at) AS invalid_at,
               relation.invalid_at IS NULL OR relation.invalid_at > datetime()
                   AS is_current
        ORDER BY coalesce(relation.importance, 0.5) DESC
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return [self._row_to_graph_edge(row) for row in rows]

    async def entity_subgraph(
        self, user_id: str, entity_id: str
    ) -> MemoryEntitySubgraphResult:
        """读取当前用户指定实体的一跳子图。"""
        node_cypher = """
        MATCH (center:Entity {id: $entity_id, user_id: $user_id})
        OPTIONAL MATCH (center)-[relation:RELATION]-(neighbor:Entity {user_id: $user_id})
        WHERE relation IS NULL OR relation.invalid_at IS NULL OR relation.invalid_at > datetime()
        WITH collect(DISTINCT center) + collect(DISTINCT neighbor) AS raw_nodes
        UNWIND raw_nodes AS entity
        WITH DISTINCT entity
        WHERE entity IS NOT NULL
        RETURN entity.id AS id,
               entity.name AS name,
               entity.type AS type,
               entity.description AS description,
               entity.community_id AS community_id,
               coalesce(entity.importance, 0.5) AS importance,
               coalesce(entity.memory_layer, 'short_term') AS memory_layer,
               coalesce(entity.access_count, 0) AS access_count,
               coalesce(entity.mention_count, 0) AS mention_count,
               coalesce(entity.core_facts, []) AS core_facts,
               coalesce(entity.traits, []) AS traits
        ORDER BY CASE WHEN entity.id = $entity_id THEN 0 ELSE 1 END,
                 coalesce(entity.importance, 0.5) DESC,
                 entity.name ASC
        """
        edge_cypher = """
        MATCH (center:Entity {id: $entity_id, user_id: $user_id})
        MATCH (center)-[relation:RELATION]-(neighbor:Entity {user_id: $user_id})
        WHERE relation.invalid_at IS NULL OR relation.invalid_at > datetime()
        WITH collect(DISTINCT center.id) + collect(DISTINCT neighbor.id) AS node_ids
        MATCH (source:Entity {user_id: $user_id})
            -[relation:RELATION {user_id: $user_id}]->
            (target:Entity {user_id: $user_id})
        WHERE source.id IN node_ids
          AND target.id IN node_ids
          AND (relation.invalid_at IS NULL OR relation.invalid_at > datetime())
        RETURN source.id AS source,
               target.id AS target,
               relation.name AS predicate,
               relation.evidence AS evidence,
               toString(relation.valid_at) AS valid_at,
               toString(relation.invalid_at) AS invalid_at,
               relation.invalid_at IS NULL OR relation.invalid_at > datetime()
                   AS is_current
        ORDER BY coalesce(relation.importance, 0.5) DESC
        """
        async with self._driver.session(database=self._database) as session:
            node_result = await session.run(
                node_cypher,
                user_id=user_id,
                entity_id=entity_id,
            )
            node_rows = await node_result.data()
            edge_result = await session.run(
                edge_cypher,
                user_id=user_id,
                entity_id=entity_id,
            )
            edge_rows = await edge_result.data()
        return MemoryEntitySubgraphResult(
            center=entity_id,
            nodes=[self._row_to_graph_node(row) for row in node_rows],
            edges=[self._row_to_graph_edge(row) for row in edge_rows],
        )

    async def profile_entities(
        self, user_id: str
    ) -> list[MemoryProfileEntityResult]:
        """读取当前用户画像实体及一跳出边事实。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id})
        OPTIONAL MATCH (entity)-[relation:RELATION {user_id: $user_id}]->
            (target:Entity {user_id: $user_id})
        WITH entity,
             collect(DISTINCT CASE
               WHEN relation IS NULL
                 OR (relation.invalid_at IS NOT NULL AND relation.invalid_at <= datetime())
               THEN null
               ELSE {
                 predicate: relation.name,
                 target_entity_id: target.id,
                 target_name: target.name,
                 target_type: target.type,
                 evidence: relation.evidence,
                 valid_at: toString(relation.valid_at),
                 invalid_at: toString(relation.invalid_at),
                 is_current: relation.invalid_at IS NULL OR relation.invalid_at > datetime()
               }
             END) AS raw_relations
        RETURN entity.id AS id,
               entity.name AS name,
               entity.type AS type,
               entity.description AS description,
               entity.community_id AS community_id,
               coalesce(entity.importance, 0.5) AS importance,
               coalesce(entity.memory_layer, 'short_term') AS memory_layer,
               coalesce(entity.access_count, 0) AS access_count,
               coalesce(entity.mention_count, 0) AS mention_count,
               coalesce(entity.core_facts, []) AS core_facts,
               coalesce(entity.traits, []) AS traits,
               [rel IN raw_relations WHERE rel IS NOT NULL AND rel.predicate IS NOT NULL] AS relations
        ORDER BY entity.type ASC,
                 coalesce(entity.importance, 0.5) DESC,
                 entity.name ASC
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return [self._row_to_profile_entity(row) for row in rows]

    async def entity_relation_history(
        self, user_id: str, entity_id: str, predicate: str | None = None
    ) -> list[MemoryRelationHistoryResult] | None:
        """读取当前用户单实体一跳关系历史。"""
        exists_cypher = """
        MATCH (center:Entity {id: $entity_id, user_id: $user_id})
        RETURN center.id AS id
        """
        history_cypher = """
        MATCH (center:Entity {id: $entity_id, user_id: $user_id})
        MATCH (center)-[relation:RELATION {user_id: $user_id}]-
            (neighbor:Entity {user_id: $user_id})
        WHERE $predicate IS NULL OR relation.name = $predicate
        RETURN relation.id AS relation_id,
               CASE WHEN startNode(relation).id = center.id
                    THEN 'outgoing' ELSE 'incoming' END AS direction,
               neighbor.id AS neighbor_entity_id,
               neighbor.name AS neighbor_name,
               neighbor.type AS neighbor_type,
               relation.name AS predicate,
               relation.evidence AS evidence,
               toString(relation.valid_at) AS valid_at,
               toString(relation.invalid_at) AS invalid_at,
               relation.invalid_at IS NULL OR relation.invalid_at > datetime()
                   AS is_current
        ORDER BY is_current DESC,
                 coalesce(relation.valid_at, datetime('1970-01-01T00:00:00')) DESC,
                 relation.name ASC
        """
        async with self._driver.session(database=self._database) as session:
            exists_result = await session.run(
                exists_cypher,
                user_id=user_id,
                entity_id=entity_id,
            )
            exists_rows = await exists_result.data()
            if not exists_rows:
                return None
            result = await session.run(
                history_cypher,
                user_id=user_id,
                entity_id=entity_id,
                predicate=predicate,
            )
            rows = await result.data()
        return [self._row_to_relation_history(row) for row in rows]

    async def entity_type_counts(self, user_id: str) -> dict[str, int]:
        """统计当前用户各实体类型数量。"""
        cypher = """
        MATCH (entity:Entity {user_id: $user_id})
        RETURN entity.type AS type,
               count(entity) AS count
        ORDER BY count DESC,
                 type ASC
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return {
            str(row.get("type") or "其他"): int(row.get("count") or 0)
            for row in rows
        }

    async def delete_entity(self, user_id: str, entity_id: str) -> bool:
        """删除当前用户单个实体，返回是否删除成功。"""
        cypher = """
        OPTIONAL MATCH (entity:Entity {id: $entity_id, user_id: $user_id})
        WITH entity, CASE WHEN entity IS NULL THEN 0 ELSE 1 END AS deleted
        FOREACH (_ IN CASE WHEN entity IS NULL THEN [] ELSE [1] END |
            DETACH DELETE entity
        )
        RETURN deleted
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                user_id=user_id,
                entity_id=entity_id,
            )
            rows = await result.data()
        return int((rows[0] if rows else {}).get("deleted") or 0) > 0

    async def quality_graph_counts(
        self, user_id: str
    ) -> MemoryQualityGraphCountsResult:
        """统计当前用户图谱节点和关系数量。"""
        cypher = """
        CALL {
          MATCH (dialogue:Dialogue {user_id: $user_id})
          RETURN count(dialogue) AS dialogues
        }
        CALL {
          MATCH (chunk:Chunk {user_id: $user_id})
          RETURN count(chunk) AS chunks
        }
        CALL {
          MATCH (statement:Statement {user_id: $user_id})
          RETURN count(statement) AS statements
        }
        CALL {
          MATCH (entity:Entity {user_id: $user_id})
          RETURN count(entity) AS entities
        }
        CALL {
          MATCH (:Entity {user_id: $user_id})
              -[relation:RELATION {user_id: $user_id}]->
              (:Entity {user_id: $user_id})
          RETURN count(relation) AS relations
        }
        CALL {
          MATCH (event:Event {user_id: $user_id})
          RETURN count(event) AS events
        }
        CALL {
          MATCH (:Event {user_id: $user_id})
              -[involves:INVOLVES {user_id: $user_id}]->
              (:Entity {user_id: $user_id})
          RETURN count(involves) AS involves
        }
        CALL {
          MATCH (community:Community {user_id: $user_id})
          RETURN count(community) AS communities
        }
        CALL {
          MATCH (insight:Insight {user_id: $user_id})
          RETURN count(insight) AS insights
        }
        RETURN dialogues, chunks, statements, entities, relations,
               events, involves, communities, insights
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return MemoryQualityGraphCountsResult.model_validate(rows[0] if rows else {})

    async def quality_issue_summary(
        self, user_id: str
    ) -> MemoryQualityIssueSummaryResult:
        """统计当前用户图谱质量问题摘要。"""
        cypher = """
        CALL {
          MATCH (entity:Entity {user_id: $user_id})
          WITH toLower(entity.name) AS normalized_name,
               entity.type AS entity_type,
               count(entity) AS count
          WHERE normalized_name <> '' AND count > 1
          RETURN count(*) AS duplicate_entities
        }
        CALL {
          MATCH (entity:Entity {user_id: $user_id})
          WHERE entity.embedding IS NULL OR size(entity.embedding) = 0
          RETURN count(entity) AS missing_embeddings
        }
        CALL {
          MATCH (entity:Entity {user_id: $user_id})
          WHERE NOT EXISTS {
            MATCH (:Statement {user_id: $user_id})
                -[:MENTIONS {user_id: $user_id}]->(entity)
          }
          RETURN count(entity) AS orphan_entities
        }
        CALL {
          MATCH (statement:Statement {user_id: $user_id})
          WHERE NOT EXISTS {
            MATCH (statement)-[:MENTIONS {user_id: $user_id}]->
                (:Entity {user_id: $user_id})
          }
          RETURN count(statement) AS orphan_statements
        }
        CALL {
          MATCH (:Entity {user_id: $user_id})
              -[relation:RELATION {user_id: $user_id}]->
              (:Entity {user_id: $user_id})
          WHERE relation.statement_id IS NULL OR NOT EXISTS {
            MATCH (:Statement {id: relation.statement_id, user_id: $user_id})
          }
          RETURN count(relation) AS broken_relations
        }
        CALL {
          MATCH (:Entity {user_id: $user_id})
              -[relation:RELATION {user_id: $user_id}]->
              (:Entity {user_id: $user_id})
          WHERE relation.invalid_at IS NOT NULL
            AND relation.invalid_at <= datetime()
          RETURN count(relation) AS expired_relations
        }
        CALL {
          MATCH (community:Community {user_id: $user_id})
          WHERE NOT EXISTS {
            MATCH (:Entity {user_id: $user_id})
                -[:IN_COMMUNITY {user_id: $user_id}]->(community)
          }
          RETURN count(community) AS empty_communities
        }
        CALL {
          MATCH (insight:Insight {user_id: $user_id})
          WHERE NOT EXISTS {
            MATCH (insight)-[:DERIVED_FROM {user_id: $user_id}]->
                (:Entity {user_id: $user_id})
          }
          RETURN count(insight) AS orphan_insights
        }
        RETURN duplicate_entities, missing_embeddings, orphan_entities,
               orphan_statements, broken_relations, expired_relations,
               empty_communities, orphan_insights
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id)
            rows = await result.data()
        return MemoryQualityIssueSummaryResult.model_validate(rows[0] if rows else {})

    async def quality_issues(
        self, user_id: str, category: str, limit: int
    ) -> MemoryQualityIssueListResult:
        """读取当前用户指定质量问题类别样本。"""
        cypher = self._quality_issue_query(category)
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, user_id=user_id, limit=limit)
            rows = await result.data()
        items = [self._row_to_quality_issue(row) for row in rows]
        return MemoryQualityIssueListResult(
            category=category,
            total=len(items),
            items=items,
        )

    async def merge_duplicate_entities(
        self, user_id: str
    ) -> MemoryMergeDuplicatesResult:
        """合并当前用户历史同名同类型重复实体。"""
        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(self._merge_duplicate_entities, user_id)

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
        OPTIONAL MATCH (source:Entity {user_id: $user_id})
            -[incoming:RELATION {user_id: $user_id}]->(node)
        OPTIONAL MATCH (node)-[outgoing:RELATION {user_id: $user_id}]->
            (target:Entity {user_id: $user_id})
        OPTIONAL MATCH (node)<-[:MENTIONS {user_id: $user_id}]-
            (statement:Statement {user_id: $user_id})
            <-[:HAS_STATEMENT]-(:Chunk {user_id: $user_id})
            <-[:HAS_CHUNK]-(dialogue:Dialogue {user_id: $user_id})
        WITH node, score, dialogue,
             collect(DISTINCT CASE
               WHEN incoming IS NULL
                 OR (incoming.invalid_at IS NOT NULL AND incoming.invalid_at <= datetime())
               THEN null
               ELSE {
                name: incoming.name,
                direction: 'incoming',
                neighbor_name: source.name,
                neighbor_type: source.type,
                evidence: incoming.evidence,
                valid_at: toString(incoming.valid_at),
                invalid_at: toString(incoming.invalid_at),
                is_current: incoming.invalid_at IS NULL OR incoming.invalid_at > datetime()
               }
             END) +
             collect(DISTINCT CASE
               WHEN outgoing IS NULL
                 OR (outgoing.invalid_at IS NOT NULL AND outgoing.invalid_at <= datetime())
               THEN null
               ELSE {
                name: outgoing.name,
                direction: 'outgoing',
                neighbor_name: target.name,
                neighbor_type: target.type,
                evidence: outgoing.evidence,
                valid_at: toString(outgoing.valid_at),
                invalid_at: toString(outgoing.invalid_at),
                is_current: outgoing.invalid_at IS NULL OR outgoing.invalid_at > datetime()
               }
             END) AS relations
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
        [relation IN relations WHERE relation IS NOT NULL AND relation.name IS NOT NULL] AS relations,
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
    async def _merge_duplicate_entities(tx, user_id: str) -> MemoryMergeDuplicatesResult:
        """在单事务中合并同名同类型重复实体。"""
        group_result = await tx.run(
            """
            MATCH (entity:Entity {user_id: $user_id})
            WITH entity,
                 toLower(entity.name) AS normalized_name,
                 entity.type AS entity_type
            ORDER BY
                 CASE WHEN coalesce(entity.memory_layer, 'short_term') = 'long_term'
                      THEN 0 ELSE 1 END,
                 coalesce(entity.created_at, datetime()) ASC,
                 coalesce(entity.mention_count, 0) DESC,
                 coalesce(entity.access_count, 0) DESC
            WITH normalized_name,
                 entity_type,
                 collect(entity) AS entities
            WHERE normalized_name <> '' AND size(entities) > 1
            RETURN [entity IN entities | entity.id] AS ids,
                   [entity IN entities | entity.name] AS names,
                   [entity IN entities | entity.description] AS descriptions,
                   [entity IN entities | coalesce(entity.core_facts, [])] AS core_facts,
                   [entity IN entities | coalesce(entity.traits, [])] AS traits,
                   reduce(total = 0, entity IN entities |
                       total + coalesce(entity.access_count, 0)
                   ) AS access_count,
                   reduce(total = 0, entity IN entities |
                       total + coalesce(entity.mention_count, 0)
                   ) AS mention_count,
                   any(entity IN entities
                       WHERE coalesce(entity.memory_layer, 'short_term') = 'long_term'
                   ) AS has_long_term
            """,
            user_id=user_id,
        )
        groups = await group_result.data()
        removed_entities = 0
        merged_groups = 0
        for group in groups:
            ids = [str(entity_id) for entity_id in group.get("ids") or [] if entity_id]
            if len(ids) < 2:
                continue
            keeper_id = ids[0]
            duplicate_ids = ids[1:]
            description = max(
                [str(item) for item in group.get("descriptions") or [] if item],
                key=len,
                default="",
            )
            core_facts = Neo4jMemoryGraphRepository._flatten_unique(
                group.get("core_facts") or []
            )
            traits = Neo4jMemoryGraphRepository._flatten_unique(
                group.get("traits") or []
            )
            await tx.run(
                """
                MATCH (keeper:Entity {id: $keeper_id, user_id: $user_id})
                MATCH (statement:Statement {user_id: $user_id})
                    -[mention:MENTIONS]->(duplicate:Entity {user_id: $user_id})
                WHERE duplicate.id IN $duplicate_ids
                MERGE (statement)-[:MENTIONS {
                    id: coalesce(mention.id, statement.id + ':' + $keeper_id),
                    user_id: $user_id
                }]->(keeper)
                """,
                user_id=user_id,
                keeper_id=keeper_id,
                duplicate_ids=duplicate_ids,
            )
            await tx.run(
                """
                MATCH (keeper:Entity {id: $keeper_id, user_id: $user_id})
                MATCH (event:Event {user_id: $user_id})
                    -[old_involves:INVOLVES]->(duplicate:Entity {user_id: $user_id})
                WHERE duplicate.id IN $duplicate_ids
                MERGE (event)-[involves:INVOLVES {
                    id: coalesce(old_involves.id, event.id + ':' + $keeper_id),
                    user_id: $user_id
                }]->(keeper)
                SET involves.role = coalesce(old_involves.role, involves.role, '')
                """,
                user_id=user_id,
                keeper_id=keeper_id,
                duplicate_ids=duplicate_ids,
            )
            await tx.run(
                """
                MATCH (keeper:Entity {id: $keeper_id, user_id: $user_id})
                MATCH (duplicate:Entity {user_id: $user_id})
                    -[relation:RELATION {user_id: $user_id}]->
                    (target:Entity {user_id: $user_id})
                WHERE duplicate.id IN $duplicate_ids
                  AND target.id <> $keeper_id
                MERGE (keeper)-[new_relation:RELATION {
                    id: relation.id,
                    user_id: $user_id
                }]->(target)
                SET new_relation.name = relation.name,
                    new_relation.evidence = relation.evidence,
                    new_relation.statement_id = relation.statement_id,
                    new_relation.importance = coalesce(relation.importance, 0.5),
                    new_relation.confidence = coalesce(relation.confidence, 0.8),
                    new_relation.valid_at = relation.valid_at,
                    new_relation.invalid_at = relation.invalid_at,
                    new_relation.access_count = coalesce(relation.access_count, 0),
                    new_relation.last_access_at = relation.last_access_at,
                    new_relation.memory_layer = coalesce(
                        relation.memory_layer, 'short_term'
                    )
                """,
                user_id=user_id,
                keeper_id=keeper_id,
                duplicate_ids=duplicate_ids,
            )
            await tx.run(
                """
                MATCH (keeper:Entity {id: $keeper_id, user_id: $user_id})
                MATCH (source:Entity {user_id: $user_id})
                    -[relation:RELATION {user_id: $user_id}]->
                    (duplicate:Entity {user_id: $user_id})
                WHERE duplicate.id IN $duplicate_ids
                  AND source.id <> $keeper_id
                MERGE (source)-[new_relation:RELATION {
                    id: relation.id,
                    user_id: $user_id
                }]->(keeper)
                SET new_relation.name = relation.name,
                    new_relation.evidence = relation.evidence,
                    new_relation.statement_id = relation.statement_id,
                    new_relation.importance = coalesce(relation.importance, 0.5),
                    new_relation.confidence = coalesce(relation.confidence, 0.8),
                    new_relation.valid_at = relation.valid_at,
                    new_relation.invalid_at = relation.invalid_at,
                    new_relation.access_count = coalesce(relation.access_count, 0),
                    new_relation.last_access_at = relation.last_access_at,
                    new_relation.memory_layer = coalesce(
                        relation.memory_layer, 'short_term'
                    )
                """,
                user_id=user_id,
                keeper_id=keeper_id,
                duplicate_ids=duplicate_ids,
            )
            await tx.run(
                """
                MATCH (keeper:Entity {id: $keeper_id, user_id: $user_id})
                MATCH (duplicate:Entity {user_id: $user_id})
                    -[:IN_COMMUNITY {user_id: $user_id}]->
                    (community:Community {user_id: $user_id})
                WHERE duplicate.id IN $duplicate_ids
                SET keeper.community_id = coalesce(keeper.community_id, community.id)
                MERGE (keeper)-[:IN_COMMUNITY {user_id: $user_id}]->(community)
                """,
                user_id=user_id,
                keeper_id=keeper_id,
                duplicate_ids=duplicate_ids,
            )
            await tx.run(
                """
                MATCH (keeper:Entity {id: $keeper_id, user_id: $user_id})
                SET keeper.description = CASE
                        WHEN size($description) > size(coalesce(keeper.description, ''))
                        THEN $description
                        ELSE keeper.description
                    END,
                    keeper.core_facts = $core_facts,
                    keeper.traits = $traits,
                    keeper.access_count = $access_count,
                    keeper.mention_count = $mention_count,
                    keeper.memory_layer = CASE
                        WHEN $has_long_term THEN 'long_term'
                        ELSE coalesce(keeper.memory_layer, 'short_term')
                    END
                """,
                user_id=user_id,
                keeper_id=keeper_id,
                description=description,
                core_facts=core_facts,
                traits=traits,
                access_count=int(group.get("access_count") or 0),
                mention_count=int(group.get("mention_count") or 0),
                has_long_term=bool(group.get("has_long_term")),
            )
            await tx.run(
                """
                MATCH (duplicate:Entity {user_id: $user_id})
                WHERE duplicate.id IN $duplicate_ids
                DETACH DELETE duplicate
                """,
                user_id=user_id,
                duplicate_ids=duplicate_ids,
            )
            removed_entities += len(duplicate_ids)
            merged_groups += 1
        return MemoryMergeDuplicatesResult(
            removed_entities=removed_entities,
            merged_groups=merged_groups,
        )

    @staticmethod
    def _flatten_unique(values: list) -> list[str]:
        """把 Neo4j 聚合出的 list[list[str]] 去重展平。"""
        unique: list[str] = []
        seen: set[str] = set()
        for item in values:
            nested = item if isinstance(item, list) else [item]
            for value in nested:
                text = str(value or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    unique.append(text)
        return unique

    @staticmethod
    def _quality_issue_query(category: str) -> str:
        """按审计类别返回样本查询。"""
        queries = {
            "duplicate_entities": """
                MATCH (entity:Entity {user_id: $user_id})
                WITH toLower(entity.name) AS normalized_name,
                     entity.type AS entity_type,
                     collect(entity) AS entities,
                     count(entity) AS count
                WHERE normalized_name <> '' AND count > 1
                RETURN 'duplicate_entities' AS category,
                       'info' AS severity,
                       '重复实体' AS title,
                       coalesce(entities[0].name, normalized_name) + '/'
                         + coalesce(entity_type, '其他')
                         + ' 存在 ' + toString(count) + ' 个同名节点' AS detail,
                       [entity IN entities | entity.id] AS entity_ids,
                       [] AS memory_ids,
                       {
                         name: coalesce(entities[0].name, normalized_name),
                         type: coalesce(entity_type, '其他'),
                         count: count
                       } AS metadata
                ORDER BY count DESC
                LIMIT $limit
            """,
            "missing_embeddings": """
                MATCH (entity:Entity {user_id: $user_id})
                WHERE entity.embedding IS NULL OR size(entity.embedding) = 0
                RETURN 'missing_embeddings' AS category,
                       'info' AS severity,
                       '实体缺少 embedding' AS title,
                       entity.name + '/' + coalesce(entity.type, '其他')
                         + ' 缺少向量' AS detail,
                       [entity.id] AS entity_ids,
                       [] AS memory_ids,
                       {name: entity.name, type: entity.type} AS metadata
                ORDER BY coalesce(entity.importance, 0.5) DESC
                LIMIT $limit
            """,
            "orphan_entities": """
                MATCH (entity:Entity {user_id: $user_id})
                WHERE NOT EXISTS {
                  MATCH (:Statement {user_id: $user_id})
                      -[:MENTIONS {user_id: $user_id}]->(entity)
                }
                RETURN 'orphan_entities' AS category,
                       'warning' AS severity,
                       '实体缺少 MENTIONS 溯源' AS title,
                       entity.name + '/' + coalesce(entity.type, '其他')
                         + ' 没有陈述来源' AS detail,
                       [entity.id] AS entity_ids,
                       [] AS memory_ids,
                       {name: entity.name, type: entity.type} AS metadata
                ORDER BY coalesce(entity.importance, 0.5) DESC
                LIMIT $limit
            """,
            "orphan_statements": """
                MATCH (statement:Statement {user_id: $user_id})
                WHERE NOT EXISTS {
                  MATCH (statement)-[:MENTIONS {user_id: $user_id}]->
                      (:Entity {user_id: $user_id})
                }
                OPTIONAL MATCH (statement)<-[:HAS_STATEMENT]-(:Chunk {user_id: $user_id})
                    <-[:HAS_CHUNK]-(dialogue:Dialogue {user_id: $user_id})
                WITH statement, collect(DISTINCT dialogue.memory_id) AS memory_ids
                RETURN 'orphan_statements' AS category,
                       'warning' AS severity,
                       '陈述缺少实体提及' AS title,
                       statement.text AS detail,
                       [] AS entity_ids,
                       [id IN memory_ids WHERE id IS NOT NULL] AS memory_ids,
                       {statement_id: statement.id} AS metadata
                ORDER BY coalesce(statement.importance, 0.5) DESC
                LIMIT $limit
            """,
            "broken_relations": """
                MATCH (source:Entity {user_id: $user_id})
                    -[relation:RELATION {user_id: $user_id}]->
                    (target:Entity {user_id: $user_id})
                WHERE relation.statement_id IS NULL OR NOT EXISTS {
                  MATCH (:Statement {id: relation.statement_id, user_id: $user_id})
                }
                RETURN 'broken_relations' AS category,
                       'warning' AS severity,
                       '断裂关系' AS title,
                       '关系 ' + relation.id + ' 缺少来源陈述' AS detail,
                       [source.id, target.id] AS entity_ids,
                       [] AS memory_ids,
                       {
                         relation_id: relation.id,
                         predicate: relation.name,
                         statement_id: relation.statement_id
                       } AS metadata
                ORDER BY coalesce(relation.importance, 0.5) DESC
                LIMIT $limit
            """,
            "expired_relations": """
                MATCH (source:Entity {user_id: $user_id})
                    -[relation:RELATION {user_id: $user_id}]->
                    (target:Entity {user_id: $user_id})
                WHERE relation.invalid_at IS NOT NULL
                  AND relation.invalid_at <= datetime()
                RETURN 'expired_relations' AS category,
                       'info' AS severity,
                       '已失效关系' AS title,
                       '关系 ' + relation.id + ' 已失效' AS detail,
                       [source.id, target.id] AS entity_ids,
                       [] AS memory_ids,
                       {
                         relation_id: relation.id,
                         predicate: relation.name,
                         valid_at: toString(relation.valid_at),
                         invalid_at: toString(relation.invalid_at)
                       } AS metadata
                ORDER BY relation.invalid_at DESC
                LIMIT $limit
            """,
            "empty_communities": """
                MATCH (community:Community {user_id: $user_id})
                WHERE NOT EXISTS {
                  MATCH (:Entity {user_id: $user_id})
                      -[:IN_COMMUNITY {user_id: $user_id}]->(community)
                }
                RETURN 'empty_communities' AS category,
                       'info' AS severity,
                       '空社区' AS title,
                       coalesce(community.name, community.id) + ' 没有成员实体' AS detail,
                       [] AS entity_ids,
                       [] AS memory_ids,
                       {
                         community_id: community.id,
                         name: community.name,
                         member_count: coalesce(community.member_count, 0)
                       } AS metadata
                ORDER BY coalesce(community.updated_at, community.created_at) DESC
                LIMIT $limit
            """,
            "orphan_insights": """
                MATCH (insight:Insight {user_id: $user_id})
                WHERE NOT EXISTS {
                  MATCH (insight)-[:DERIVED_FROM {user_id: $user_id}]->
                      (:Entity {user_id: $user_id})
                }
                RETURN 'orphan_insights' AS category,
                       'warning' AS severity,
                       '洞察缺少 DERIVED_FROM 溯源' AS title,
                       insight.theme AS detail,
                       [] AS entity_ids,
                       [] AS memory_ids,
                       {
                         insight_id: insight.id,
                         theme: insight.theme,
                         source_count: coalesce(insight.source_count, 0)
                       } AS metadata
                ORDER BY coalesce(insight.importance, 0.6) DESC
                LIMIT $limit
            """,
        }
        if category not in queries:
            raise ValueError(f"unknown memory quality issue category: {category}")
        return queries[category]

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
                statement.valid_at = row.valid_at,
                statement.invalid_at = row.invalid_at,
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
                rel.valid_at = row.valid_at,
                rel.invalid_at = row.invalid_at,
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
        if params["events"]:
            await tx.run(
                """
                UNWIND $events AS row
                MERGE (event:Event {id: row.id, user_id: row.user_id})
                SET event.title = row.title,
                    event.description = row.description,
                    event.event_time = row.event_time,
                    event.created_at = coalesce(event.created_at, row.created_at)
                """,
                events=params["events"],
                user_id=params["user_id"],
            )
        if params["involves"]:
            await tx.run(
                """
                UNWIND $involves AS row
                MATCH (event:Event {id: row.event_id, user_id: row.user_id})
                MATCH (entity:Entity {id: row.entity_id, user_id: row.user_id})
                MERGE (event)-[involves:INVOLVES {id: row.id, user_id: row.user_id}]->(entity)
                SET involves.role = row.role
                """,
                involves=params["involves"],
                user_id=params["user_id"],
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
    def _row_to_community_vote_entity(
        row: dict[str, Any],
        user_id: str,
    ) -> CommunityVoteEntity:
        """把 Neo4j 行转换为社区聚类投票实体。"""
        return CommunityVoteEntity(
            id=str(row.get("id") or ""),
            user_id=user_id,
            name=str(row.get("name") or ""),
            type=str(row.get("type") or ""),
            description=str(row.get("description") or ""),
            embedding=row.get("embedding") or [],
            community_id=row.get("community_id"),
        )

    @staticmethod
    def _row_to_community_vote_neighbor(row: dict[str, Any]) -> CommunityVoteNeighbor:
        """把 Neo4j 行转换为社区聚类投票邻居。"""
        return CommunityVoteNeighbor(
            id=str(row.get("id") or ""),
            community_id=row.get("community_id"),
            embedding=row.get("embedding") or [],
        )

    @staticmethod
    def _row_to_community_result(row: dict[str, Any]) -> CommunityResult:
        """把 Neo4j 行转换为社区列表项。"""
        return CommunityResult(
            id=str(row.get("id") or ""),
            name=str(row.get("name") or ""),
            summary=str(row.get("summary") or ""),
            member_count=int(row.get("member_count") or 0),
        )

    @staticmethod
    def _row_to_community_member(row: dict[str, Any]) -> CommunityMemberResult:
        """把 Neo4j 行转换为社区成员。"""
        return CommunityMemberResult(
            entity_id=str(row.get("entity_id") or ""),
            entity_name=str(row.get("entity_name") or ""),
            entity_type=str(row.get("entity_type") or ""),
            description=str(row.get("description") or ""),
            community_id=str(row.get("community_id") or ""),
            embedding=[float(value) for value in row.get("embedding") or []],
            importance=float(row.get("importance") or 0.5),
            mention_count=int(row.get("mention_count") or 0),
            access_count=int(row.get("access_count") or 0),
        )

    @staticmethod
    def _row_to_community_relation(row: dict[str, Any]) -> CommunityRelationResult:
        """把 Neo4j 行转换为社区内部关系。"""
        return CommunityRelationResult(
            source_entity_id=str(row.get("source_entity_id") or ""),
            source_name=str(row.get("source_name") or ""),
            target_entity_id=str(row.get("target_entity_id") or ""),
            target_name=str(row.get("target_name") or ""),
            name=str(row.get("name") or ""),
            evidence=str(row.get("evidence") or ""),
            valid_at=row.get("valid_at"),
            invalid_at=row.get("invalid_at"),
            is_current=bool(row.get("is_current", True)),
        )

    @staticmethod
    def _row_to_graph_node(row: dict[str, Any]) -> MemoryGraphNodeResult:
        """把 Neo4j 行转换为图谱展示节点。"""
        return MemoryGraphNodeResult(
            id=str(row.get("id") or ""),
            name=str(row.get("name") or ""),
            type=str(row.get("type") or ""),
            description=str(row.get("description") or ""),
            community_id=row.get("community_id"),
            importance=float(row.get("importance") or 0.5),
            memory_layer=str(row.get("memory_layer") or "short_term"),
            access_count=int(row.get("access_count") or 0),
            mention_count=int(row.get("mention_count") or 0),
            core_facts=row.get("core_facts") or [],
            traits=row.get("traits") or [],
        )

    @staticmethod
    def _row_to_graph_edge(row: dict[str, Any]) -> MemoryGraphEdgeResult:
        """把 Neo4j 行转换为图谱展示边。"""
        return MemoryGraphEdgeResult(
            source=str(row.get("source") or ""),
            target=str(row.get("target") or ""),
            predicate=str(row.get("predicate") or ""),
            evidence=str(row.get("evidence") or ""),
            valid_at=row.get("valid_at"),
            invalid_at=row.get("invalid_at"),
            is_current=bool(row.get("is_current", True)),
        )

    @staticmethod
    def _row_to_profile_entity(row: dict[str, Any]) -> MemoryProfileEntityResult:
        """把 Neo4j 行转换为画像实体。"""
        return MemoryProfileEntityResult(
            id=str(row.get("id") or ""),
            name=str(row.get("name") or ""),
            type=str(row.get("type") or ""),
            description=str(row.get("description") or ""),
            community_id=row.get("community_id"),
            importance=float(row.get("importance") or 0.5),
            memory_layer=str(row.get("memory_layer") or "short_term"),
            access_count=int(row.get("access_count") or 0),
            mention_count=int(row.get("mention_count") or 0),
            core_facts=row.get("core_facts") or [],
            traits=row.get("traits") or [],
            relations=[
                MemoryProfileRelationResult(
                    predicate=str(relation.get("predicate") or ""),
                    target_entity_id=relation.get("target_entity_id"),
                    target_name=relation.get("target_name"),
                    target_type=relation.get("target_type"),
                    evidence=str(relation.get("evidence") or ""),
                    valid_at=relation.get("valid_at"),
                    invalid_at=relation.get("invalid_at"),
                    is_current=bool(relation.get("is_current", True)),
                )
                for relation in row.get("relations") or []
                if relation.get("predicate")
            ],
        )

    @staticmethod
    def _row_to_relation_history(row: dict[str, Any]) -> MemoryRelationHistoryResult:
        """把 Neo4j 行转换为关系历史项。"""
        return MemoryRelationHistoryResult(
            relation_id=str(row.get("relation_id") or ""),
            direction=str(row.get("direction") or ""),
            neighbor_entity_id=str(row.get("neighbor_entity_id") or ""),
            neighbor_name=str(row.get("neighbor_name") or ""),
            neighbor_type=str(row.get("neighbor_type") or ""),
            predicate=str(row.get("predicate") or ""),
            evidence=str(row.get("evidence") or ""),
            valid_at=row.get("valid_at"),
            invalid_at=row.get("invalid_at"),
            is_current=bool(row.get("is_current", True)),
        )

    @staticmethod
    def _row_to_quality_issue(row: dict[str, Any]) -> MemoryQualityIssueResult:
        """把 Neo4j 行转换为质量审计问题项。"""
        return MemoryQualityIssueResult(
            category=str(row.get("category") or ""),
            severity=str(row.get("severity") or "info"),
            title=str(row.get("title") or ""),
            detail=str(row.get("detail") or ""),
            entity_ids=[
                str(entity_id)
                for entity_id in row.get("entity_ids") or []
                if entity_id
            ],
            memory_ids=[
                str(memory_id)
                for memory_id in row.get("memory_ids") or []
                if memory_id
            ],
            metadata=row.get("metadata") or {},
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
            embedding=[
                float(value)
                for value in row.get("embedding") or []
            ],
        )

    @staticmethod
    def _row_to_insight_result(row: dict[str, Any]) -> InsightResult:
        """把 Neo4j 洞察行转换为领域结果。"""
        return InsightResult(
            id=str(row.get("id") or ""),
            theme=str(row.get("theme") or ""),
            content=str(row.get("content") or ""),
            importance=float(row.get("importance") or 0.6),
            confidence=float(row.get("confidence") or 0.7),
            source_count=int(row.get("source_count") or 0),
            score=float(row.get("score") or 0),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    @staticmethod
    def _row_to_timeline_event(row: dict[str, Any]) -> MemoryTimelineEventResult:
        """把 Neo4j 事件行转换为时间线结果。"""
        return MemoryTimelineEventResult(
            id=str(row.get("id") or ""),
            title=str(row.get("title") or ""),
            description=str(row.get("description") or ""),
            event_time=row.get("event_time"),
            created_at=row.get("created_at"),
            participants=[
                MemoryTimelineParticipantResult(
                    entity_id=str(participant.get("entity_id") or ""),
                    name=str(participant.get("name") or ""),
                    type=str(participant.get("type") or ""),
                )
                for participant in row.get("participants") or []
                if participant.get("entity_id")
            ],
        )
