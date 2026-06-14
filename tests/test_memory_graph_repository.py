import pytest

from app.domain.models.memory_graph import (
    ChunkNode,
    DialogueNode,
    EntityNode,
    MemoryGraph,
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


class FakeNeo4jDriver:
    def __init__(self, result_rows=None):
        self.executed = []
        self.result_rows = result_rows or []

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
        return FakeNeo4jResult(self.driver.result_rows)

    async def execute_write(self, callback, *args):
        tx = FakeNeo4jTransaction(self.driver)
        return await callback(tx, *args)


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
