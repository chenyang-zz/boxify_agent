from typing import Any

from app.domain.models.memory_graph import (
    GraphRelationFact,
    MemoryGraph,
    MemoryGraphResult,
)
from app.domain.repositories.memory_graph_repository import MemoryGraphRepository


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

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryGraphResult]:
        """混合召回实体及一跳关系，Neo4j 不可用时由上层降级。"""
        rows = await self._search_fulltext(user_id=user_id, query=query, top_k=top_k)
        if query_embedding:
            rows.extend(
                await self._search_vector(
                    user_id=user_id,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )
            )
        return self._dedupe_rows(rows, top_k)

    async def _search_fulltext(
        self, user_id: str, query: str, top_k: int
    ) -> list[dict[str, Any]]:
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
        return """
        ORDER BY score DESC
        LIMIT $top_k
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
            memory_id: dialogue.memory_id,
            memory_summary: coalesce(dialogue.summary, '')
        } AS entity,
        [relation IN relations WHERE relation.name IS NOT NULL] AS relations,
        score
        """

    def _dedupe_rows(
        self, rows: list[dict[str, Any]], top_k: int
    ) -> list[MemoryGraphResult]:
        by_entity: dict[str, MemoryGraphResult] = {}
        for row in rows:
            result = self._row_to_search_result(row)
            existing = by_entity.get(result.entity_id)
            if not existing or result.score > existing.score:
                by_entity[result.entity_id] = result
        return sorted(
            by_entity.values(),
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

    @staticmethod
    async def _merge_graph(tx, params: dict[str, Any]) -> None:
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
                statement.text = row.text
            MERGE (chunk)-[:HAS_STATEMENT]->(statement)
            WITH d
            UNWIND $entities AS row
            MERGE (entity:Entity {id: row.id, user_id: row.user_id})
            SET entity.name = row.name,
                entity.type = row.type,
                entity.description = row.description,
                entity.embedding = row.embedding
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
                rel.statement_id = statement.id
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
            source_memory_id=entity.get("memory_id"),
            source_memory_summary=entity.get("memory_summary") or None,
            relations=relations,
            score=float(row.get("score") or 0),
        )
