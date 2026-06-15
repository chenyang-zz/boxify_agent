import pytest

from app.domain.models.memory_graph import (
    ChunkNode,
    DialogueNode,
    EntityNode,
    InsightResult,
    MemoryGraph,
    MemoryPromotionStats,
    RelationEdge,
    StatementNode,
)
from app.infrastructure.repositories.neo4j_memory_graph_repository import (
    Neo4jMemoryGraphRepository,
)


@pytest.mark.anyio
async def test_neo4j_repository_initializes_schema_and_merges_graph_by_user():
    driver = FakeNeo4jDriver()
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )
    graph = MemoryGraph(
        dialogue=DialogueNode(id="dialogue-1", user_id="user-a", memory_id="mem-1"),
        chunks=[
            ChunkNode(
                id="chunk-1",
                user_id="user-a",
                dialogue_id="dialogue-1",
                index=0,
                text="用户喜欢周杰伦。",
            )
        ],
        statements=[
            StatementNode(
                id="statement-1",
                user_id="user-a",
                chunk_id="chunk-1",
                index=0,
                text="用户喜欢周杰伦。",
            )
        ],
        entities=[
            EntityNode(
                id="entity-1",
                user_id="user-a",
                name="周杰伦",
                type="Person",
                description="歌手",
                embedding=[0.1, 0.2],
            )
        ],
        relations=[
            RelationEdge(
                id="rel-1",
                user_id="user-a",
                source_entity_id="entity-user",
                target_entity_id="entity-1",
                statement_id="statement-1",
                name="LIKES",
                evidence="用户喜欢周杰伦。",
            )
        ],
    )

    await repository.ensure_schema()
    await repository.save_graph(graph)

    executed_queries = "\n".join(query for query, _ in driver.executed)
    assert "CREATE CONSTRAINT memory_dialogue_id" in executed_queries
    assert "CREATE VECTOR INDEX memory_entity_embedding" in executed_queries
    assert "MERGE (d:Dialogue {id: $dialogue_id, user_id: $user_id})" in executed_queries
    assert "MERGE (entity:Entity {id: row.id, user_id: row.user_id})" in executed_queries
    assert "MERGE (source)-[rel:RELATION {id: row.id, user_id: row.user_id}]->(target)" in executed_queries
    save_params = driver.executed[-1][1]
    assert save_params["user_id"] == "user-a"
    assert save_params["entities"][0]["user_id"] == "user-a"
    assert save_params["relations"][0]["user_id"] == "user-a"


def test_memory_graph_nodes_have_dynamic_defaults():
    entity = EntityNode(
        id="entity-1",
        user_id="user-a",
        name="用户",
        type="生命体",
    )
    statement = StatementNode(
        id="statement-1",
        user_id="user-a",
        chunk_id="chunk-1",
        index=0,
        text="用户喜欢周杰伦。",
    )
    relation = RelationEdge(
        id="rel-1",
        user_id="user-a",
        source_entity_id="entity-user",
        target_entity_id="entity-1",
        statement_id="statement-1",
        name="偏好",
        evidence="用户喜欢周杰伦。",
    )

    assert entity.memory_layer == "short_term"
    assert entity.access_count == 0
    assert entity.mention_count == 1
    assert entity.last_access_at is None
    assert entity.core_facts == []
    assert entity.traits == []
    assert statement.memory_layer == "short_term"
    assert relation.memory_layer == "short_term"


@pytest.mark.anyio
async def test_neo4j_repository_returns_one_hop_relationships_with_source_memory():
    driver = FakeNeo4jDriver(
        result_rows=[
            {
                "entity": {
                    "id": "entity-1",
                    "name": "周杰伦",
                    "type": "Person",
                    "description": "歌手",
                    "memory_id": "mem-1",
                    "memory_summary": "用户喜欢周杰伦。",
                },
                "relations": [
                    {
                        "name": "LIKES",
                        "direction": "incoming",
                        "neighbor_name": "用户",
                        "neighbor_type": "Person",
                        "evidence": "用户喜欢周杰伦。",
                    }
                ],
                "score": 0.8,
            }
        ]
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    results = await repository.search(user_id="user-a", query="喜欢的歌手", top_k=5)

    assert results[0].entity_name == "周杰伦"
    assert results[0].entity_type == "Person"
    assert results[0].source_memory_id == "mem-1"
    assert results[0].relations[0].evidence == "用户喜欢周杰伦。"


@pytest.mark.anyio
async def test_neo4j_repository_search_orders_before_fetching_neighbors_and_fuses_hits():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "fulltext.queryNodes": [
                {
                    "entity": {
                        "id": "entity-1",
                        "name": "周杰伦",
                        "type": "生命体",
                        "description": "歌手",
                    },
                    "relations": [],
                    "score": 0.4,
                }
            ],
            "vector.queryNodes": [
                {
                    "entity": {
                        "id": "entity-1",
                        "name": "周杰伦",
                        "type": "生命体",
                        "description": "歌手",
                    },
                    "relations": [],
                    "score": 0.9,
                },
                {
                    "entity": {
                        "id": "entity-2",
                        "name": "林俊杰",
                        "type": "生命体",
                        "description": "歌手",
                    },
                    "relations": [],
                    "score": 0.7,
                },
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    results = await repository.search(
        user_id="user-a",
        query="喜欢的歌手",
        top_k=2,
        query_embedding=[0.1, 0.2],
    )

    assert [result.entity_id for result in results] == ["entity-1", "entity-2"]
    assert results[0].score == pytest.approx(0.925)
    assert results[1].score == pytest.approx(0.075)
    search_queries = [query for query, _ in driver.executed if "queryNodes" in query]
    assert len(search_queries) == 2
    for query in search_queries:
        assert query.index("ORDER BY score DESC") < query.index("OPTIONAL MATCH")


@pytest.mark.anyio
async def test_neo4j_repository_scores_importance_layer_and_bumps_access():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "fulltext.queryNodes": [
                _entity_row("stable", score=0.8, importance=0.9, layer="long_term"),
                _entity_row("semantic", score=0.7, importance=0.1),
                _entity_row("weak", score=0.1, importance=0.5),
            ],
            "vector.queryNodes": [
                _entity_row("stable", score=0.89, importance=0.9, layer="long_term"),
                _entity_row("semantic", score=0.9, importance=0.1),
                _entity_row("weak", score=0.5, importance=0.5),
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    results = await repository.search(
        user_id="user-a",
        query="偏好的歌手",
        top_k=2,
        query_embedding=[0.1, 0.2],
    )

    assert [result.entity_id for result in results] == ["stable", "semantic"]
    assert results[0].score == pytest.approx(1.02125)
    assert results[0].importance == pytest.approx(0.9)
    assert results[0].memory_layer == "long_term"
    bump_query, bump_params = driver.executed[-1]
    assert "SET entity.access_count = coalesce(entity.access_count, 0) + 1" in bump_query
    assert bump_params == {"user_id": "user-a", "entity_ids": ["stable", "semantic"]}


@pytest.mark.anyio
async def test_neo4j_repository_promotes_short_term_and_writes_profiles_by_user():
    driver = FakeNeo4jDriver(result_rows=[{"entities": 2, "statements": 3}])
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    promoted = await repository.promote_short_to_long(
        user_id="user-a",
        min_access=3,
        min_importance=0.8,
        min_mention=3,
        age_before="2026-06-15T00:00:00",
    )
    await repository.write_entity_profile(
        user_id="user-a",
        entity_id="entity-1",
        core_facts=["用户长期喜欢周杰伦"],
        traits=["偏好华语流行"],
    )

    assert promoted == MemoryPromotionStats(
        promoted_entities=2,
        promoted_statements=3,
    )
    promote_query, promote_params = driver.executed[0]
    assert "MATCH (entity:Entity {user_id: $user_id})" in promote_query
    assert "entity.memory_layer = 'long_term'" in promote_query
    assert promote_params["user_id"] == "user-a"
    profile_query, profile_params = driver.executed[1]
    assert "MATCH (entity:Entity {id: $entity_id, user_id: $user_id})" in profile_query
    assert "SET entity.core_facts = $core_facts" in profile_query
    assert profile_params["core_facts"] == ["用户长期喜欢周杰伦"]


@pytest.mark.anyio
async def test_neo4j_repository_lists_existing_entities_by_type_for_dedup():
    driver = FakeNeo4jDriver(
        result_rows=[
            {
                "id": "entity-1",
                "name": "用户",
                "type": "生命体",
                "description": "当前用户",
            }
        ]
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    rows = await repository.list_entities_by_type("user-a", "生命体")

    assert rows == [
        EntityNode(
            id="entity-1",
            user_id="user-a",
            name="用户",
            type="生命体",
            description="当前用户",
        )
    ]
    query, params = driver.executed[-1]
    assert "MATCH (entity:Entity {user_id: $user_id, type: $entity_type})" in query
    assert params == {"user_id": "user-a", "entity_type": "生命体"}


@pytest.mark.anyio
async def test_neo4j_repository_manages_insight_schema_upsert_and_vector_search():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "MATCH (entity:Entity {user_id: $user_id, memory_layer: 'long_term'})": [
                {"id": "entity-1", "name": "周杰伦", "type": "生命体"}
            ],
            "MATCH (entity:Entity {id: $entity_id, user_id: $user_id})": [
                {"text": "用户喜欢周杰伦。"}
            ],
            "vector.similarity.cosine": [
                {
                    "id": "insight-1",
                    "theme": "音乐偏好",
                    "content": "用户偏好华语流行音乐。",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "source_count": 2,
                    "score": 0.92,
                }
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    await repository.ensure_schema()
    top_entities = await repository.reflection_top_entities("user-a", top_k=30)
    statements = await repository.reflection_entity_statements(
        "user-a", "entity-1", limit=5
    )
    await repository.upsert_insight(
        user_id="user-a",
        theme="音乐偏好",
        content="用户偏好华语流行音乐。",
        embedding=[0.1, 0.2],
        importance=0.8,
        confidence=0.9,
        source_count=2,
        entity_ids=["entity-1", "entity-2"],
    )
    insights = await repository.search_insights_by_vector(
        "user-a", query_embedding=[0.1, 0.2], top_k=3
    )

    executed_queries = "\n".join(query for query, _ in driver.executed)
    assert "CREATE CONSTRAINT memory_insight_id" in executed_queries
    assert "CREATE VECTOR INDEX memory_insight_embedding" in executed_queries
    assert top_entities[0].name == "周杰伦"
    assert statements == ["用户喜欢周杰伦。"]
    upsert_query, upsert_params = next(
        (query, params) for query, params in driver.executed if "MERGE (insight:Insight" in query
    )
    assert "MERGE (insight:Insight {user_id: $user_id, theme: $theme})" in upsert_query
    assert "DELETE derived" in upsert_query
    assert upsert_params["entity_ids"] == ["entity-1", "entity-2"]
    insight_query, insight_params = driver.executed[-1]
    assert "MATCH (insight:Insight {user_id: $user_id})" in insight_query
    assert "vector.similarity.cosine(insight.embedding, $query_embedding)" in insight_query
    assert "LIMIT $top_k" in insight_query
    assert "memory_insight_embedding" not in insight_query
    assert "db.index.vector.queryNodes" not in insight_query
    assert insight_params == {
        "user_id": "user-a",
        "query_embedding": [0.1, 0.2],
        "top_k": 3,
    }
    assert insights == [
        InsightResult(
            id="insight-1",
            theme="音乐偏好",
            content="用户偏好华语流行音乐。",
            importance=0.8,
            confidence=0.9,
            source_count=2,
            score=0.92,
        )
    ]


class FakeNeo4jDriver:
    def __init__(self, result_rows=None, result_rows_by_keyword=None):
        self.executed = []
        self.result_rows = result_rows or []
        self.result_rows_by_keyword = result_rows_by_keyword or {}

    def session(self, database=None):
        return FakeNeo4jSession(self, database)


class FakeNeo4jSession:
    def __init__(self, driver, database):
        self.driver = driver
        self.database = database

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def run(self, query, **params):
        self.driver.executed.append((query, params))
        return FakeNeo4jResult(self._rows_for_query(query))

    async def execute_write(self, callback, *args):
        tx = FakeNeo4jTransaction(self.driver)
        return await callback(tx, *args)

    def _rows_for_query(self, query):
        for keyword, rows in self.driver.result_rows_by_keyword.items():
            if keyword in query:
                return rows
        return self.driver.result_rows


class FakeNeo4jTransaction:
    def __init__(self, driver):
        self.driver = driver

    async def run(self, query, **params):
        self.driver.executed.append((query, params))
        return FakeNeo4jResult(self.driver.result_rows)


class FakeNeo4jResult:
    def __init__(self, rows):
        self.rows = rows

    async def data(self):
        return self.rows


def _entity_row(
    entity_id: str,
    *,
    score: float,
    importance: float,
    layer: str = "short_term",
):
    return {
        "entity": {
            "id": entity_id,
            "name": entity_id,
            "type": "生命体",
            "description": "",
            "importance": importance,
            "memory_layer": layer,
            "core_facts": ["核心事实"] if layer == "long_term" else [],
            "traits": ["稳定偏好"] if layer == "long_term" else [],
            "access_count": 0,
            "mention_count": 2,
        },
        "relations": [],
        "score": score,
    }
