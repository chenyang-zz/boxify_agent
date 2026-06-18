import pytest

from app.domain.models.memory_graph import (
    ChunkNode,
    CommunityMemberResult,
    CommunityRelationResult,
    CommunityResult,
    CommunityVoteNeighbor,
    DialogueNode,
    EntityNode,
    EventNode,
    InsightResult,
    InvolvesEdge,
    MemoryEntitySubgraphResult,
    MemoryGraph,
    MemoryGraphEdgeResult,
    MemoryGraphNodeResult,
    MemoryMergeDuplicatesResult,
    MemoryProfileEntityResult,
    MemoryProfileRelationResult,
    MemoryPromotionStats,
    MemoryQualityGraphCountsResult,
    MemoryQualityIssueListResult,
    MemoryQualityIssueResult,
    MemoryQualityIssueSummaryResult,
    MemoryTraceChunkResult,
    MemoryTraceDialogueResult,
    MemoryTraceEntityResult,
    MemoryTraceEventResult,
    MemoryTraceMentionResult,
    MemoryTraceRelationResult,
    MemoryTraceResult,
    MemoryTraceStatementResult,
    MemoryRelationHistoryResult,
    MemoryTimelineEventResult,
    MemoryTimelineParticipantResult,
    MemoryActiveRecallCommunityResult,
    MemoryActiveRecallEventResult,
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
                valid_at="2026-06-16T09:00:00",
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
                valid_at="2026-06-16T09:00:00",
                invalid_at="2026-07-01T00:00:00",
            )
        ],
        events=[
            EventNode(
                id="event-1",
                user_id="user-a",
                title="参加周杰伦演唱会",
                description="用户参加了周杰伦演唱会",
            )
        ],
        involves=[
            InvolvesEdge(
                id="involves-1",
                user_id="user-a",
                event_id="event-1",
                entity_id="entity-1",
            )
        ],
    )

    await repository.ensure_schema()
    await repository.save_graph(graph)

    executed_queries = "\n".join(query for query, _ in driver.executed)
    assert "CREATE CONSTRAINT memory_dialogue_id" in executed_queries
    assert "CREATE CONSTRAINT memory_event_id" in executed_queries
    assert "CREATE INDEX memory_event_user_time" in executed_queries
    assert "CREATE VECTOR INDEX memory_entity_embedding" in executed_queries
    assert "MERGE (d:Dialogue {id: $dialogue_id, user_id: $user_id})" in executed_queries
    assert "MERGE (entity:Entity {id: row.id, user_id: row.user_id})" in executed_queries
    assert "MERGE (source)-[rel:RELATION {id: row.id, user_id: row.user_id}]->(target)" in executed_queries
    assert "MERGE (event:Event {id: row.id, user_id: row.user_id})" in executed_queries
    assert "event.dialogue_id = row.dialogue_id" in executed_queries
    assert "MERGE (event)-[involves:INVOLVES {id: row.id, user_id: row.user_id}]->(entity)" in executed_queries
    save_params = next(params for _, params in driver.executed if "entities" in params)
    event_params = next(params for _, params in driver.executed if "events" in params)
    involves_params = next(
        params for _, params in driver.executed if "involves" in params
    )
    assert save_params["user_id"] == "user-a"
    assert save_params["entities"][0]["user_id"] == "user-a"
    assert save_params["statements"][0]["valid_at"].isoformat() == "2026-06-16T09:00:00"
    assert save_params["relations"][0]["user_id"] == "user-a"
    assert save_params["relations"][0]["valid_at"].isoformat() == "2026-06-16T09:00:00"
    assert save_params["relations"][0]["invalid_at"].isoformat() == "2026-07-01T00:00:00"
    assert event_params["events"][0]["user_id"] == "user-a"
    assert event_params["events"][0]["dialogue_id"] == "dialogue-1"
    assert involves_params["involves"][0]["user_id"] == "user-a"


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
    assert statement.valid_at is None
    assert statement.invalid_at is None
    assert relation.memory_layer == "short_term"
    assert relation.valid_at is None
    assert relation.invalid_at is None


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
                        "valid_at": "2026-06-16T09:00:00",
                        "invalid_at": None,
                        "is_current": True,
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
    assert results[0].relations[0].valid_at.isoformat() == "2026-06-16T09:00:00"
    assert results[0].relations[0].is_current is True


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
                "embedding": [0.1, 0.2],
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
            embedding=[0.1, 0.2],
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


@pytest.mark.anyio
async def test_neo4j_repository_manages_community_schema_assignment_and_queries():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "RETURN count(community) AS count": [{"count": 1}],
            "dialogue:Dialogue {id: $dialogue_id, user_id: $user_id}": [
                {"id": "entity-1"},
                {"id": "entity-2"},
            ],
            "WHERE entity.id IN $entity_ids\n        RETURN entity.id AS id": [
                {
                    "id": "entity-1",
                    "name": "周杰伦",
                    "type": "生命体",
                    "description": "歌手",
                    "embedding": [0.1, 0.2],
                    "community_id": "community-music",
                }
            ],
            "neighbor.id AS id": [
                {
                    "entity_id": "entity-1",
                    "id": "entity-2",
                    "community_id": "community-music",
                    "embedding": [0.2, 0.3],
                }
            ],
            "RETURN count AS count": [{"count": 2}],
            "RETURN community.id AS id": [
                {
                    "id": "community-music",
                    "name": "音乐偏好",
                    "summary": "用户的音乐相关实体",
                    "member_count": 2,
                }
            ],
            "RETURN entity.id AS entity_id": [
                {
                    "entity_id": "entity-1",
                    "entity_name": "周杰伦",
                    "entity_type": "生命体",
                    "description": "歌手",
                    "community_id": "community-music",
                    "embedding": [0.1, 0.2],
                    "importance": 0.8,
                    "mention_count": 2,
                    "access_count": 1,
                }
            ],
            "RETURN source.id AS source_entity_id": [
                {
                    "source_entity_id": "entity-user",
                    "source_name": "用户",
                    "target_entity_id": "entity-1",
                    "target_name": "周杰伦",
                    "name": "偏好",
                    "evidence": "用户喜欢周杰伦。",
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
    assert await repository.has_communities("user-a") is True
    assert await repository.dialogue_entity_ids("user-a", "dialogue-1") == [
        "entity-1",
        "entity-2",
    ]
    vote_entities = await repository.community_vote_entities(
        "user-a", ["entity-1"]
    )
    neighbors = await repository.community_vote_neighbors("user-a", ["entity-1"])
    await repository.upsert_community("user-a", "community-music")
    await repository.assign_entity_community("user-a", "entity-1", "community-music")
    assert await repository.refresh_community_member_count(
        "user-a", "community-music"
    ) == 2
    communities = await repository.list_communities("user-a")
    members = await repository.community_members("user-a", "community-music")
    relationships = await repository.community_relationships(
        "user-a", "community-music"
    )
    await repository.update_community_metadata(
        "user-a", "community-music", "音乐偏好", "用户的音乐相关实体"
    )
    await repository.prune_empty_communities("user-a")

    executed_queries = "\n".join(query for query, _ in driver.executed)
    assert "CREATE CONSTRAINT memory_community_id" in executed_queries
    assert "CREATE INDEX memory_community_user" in executed_queries
    assert "MERGE (community:Community {id: $community_id, user_id: $user_id})" in executed_queries
    assert "MERGE (entity)-[:IN_COMMUNITY {user_id: $user_id}]->(community)" in executed_queries
    assert vote_entities[0].community_id == "community-music"
    assert neighbors == {
        "entity-1": [
            CommunityVoteNeighbor(
                id="entity-2",
                community_id="community-music",
                embedding=[0.2, 0.3],
            )
        ]
    }
    assert communities == [
        CommunityResult(
            id="community-music",
            name="音乐偏好",
            summary="用户的音乐相关实体",
            member_count=2,
        )
    ]
    assert members == [
        CommunityMemberResult(
            entity_id="entity-1",
            entity_name="周杰伦",
            entity_type="生命体",
            description="歌手",
            community_id="community-music",
            embedding=[0.1, 0.2],
            importance=0.8,
            mention_count=2,
            access_count=1,
        )
    ]
    assert relationships == [
        CommunityRelationResult(
            source_entity_id="entity-user",
            source_name="用户",
            target_entity_id="entity-1",
            target_name="周杰伦",
            name="偏好",
            evidence="用户喜欢周杰伦。",
        )
    ]
    for _, params in driver.executed:
        if params:
            assert params.get("user_id") == "user-a"


@pytest.mark.anyio
async def test_neo4j_repository_returns_event_timeline_by_user():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "MATCH (event:Event {user_id: $user_id})": [
                {
                    "id": "event-1",
                    "title": "参加周杰伦演唱会",
                    "description": "用户参加了周杰伦演唱会",
                    "event_time": "2026-06-15T20:00:00",
                    "created_at": "2026-06-16T09:00:00",
                    "participants": [
                        {
                            "entity_id": "entity-user",
                            "name": "用户",
                            "type": "生命体",
                        },
                        {
                            "entity_id": "entity-1",
                            "name": "周杰伦",
                            "type": "生命体",
                        },
                    ],
                }
            ]
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    events = await repository.event_timeline("user-a", limit=50)

    assert events == [
        MemoryTimelineEventResult(
            id="event-1",
            title="参加周杰伦演唱会",
            description="用户参加了周杰伦演唱会",
            event_time="2026-06-15T20:00:00",
            created_at="2026-06-16T09:00:00",
            participants=[
                MemoryTimelineParticipantResult(
                    entity_id="entity-user",
                    name="用户",
                    type="生命体",
                ),
                MemoryTimelineParticipantResult(
                    entity_id="entity-1",
                    name="周杰伦",
                    type="生命体",
                ),
            ],
        )
    ]
    query, params = driver.executed[-1]
    assert "MATCH (event:Event {user_id: $user_id})" in query
    assert "ORDER BY event_has_time DESC" in query
    assert params == {"user_id": "user-a", "limit": 50}


@pytest.mark.anyio
async def test_neo4j_repository_searches_communities_by_member_embeddings():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "RETURN community.id AS id": [
                {
                    "id": "community-music",
                    "name": "音乐偏好",
                    "summary": "用户的音乐相关实体",
                    "member_count": 2,
                }
            ],
            "RETURN entity.id AS id": [
                {
                    "id": "entity-1",
                    "name": "周杰伦",
                    "type": "生命体",
                    "description": "歌手",
                    "embedding": [1.0, 0.0],
                    "community_id": "community-music",
                }
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=2,
    )

    results = await repository.search_communities_by_vector(
        "user-a", [1.0, 0.0], top_k=2
    )

    assert results == [
        MemoryActiveRecallCommunityResult(
            id="community-music",
            name="音乐偏好",
            summary="用户的音乐相关实体",
            member_count=2,
            score=1.0,
        )
    ]
    assert all(params.get("user_id") == "user-a" for _, params in driver.executed)


@pytest.mark.anyio
async def test_neo4j_repository_searches_events_by_text_or_participant_vectors():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "MATCH (event:Event {user_id: $user_id})": [
                {
                    "id": "event-1",
                    "title": "参加周杰伦演唱会",
                    "description": "用户参加了周杰伦演唱会",
                    "event_time": "2026-06-15T20:00:00",
                    "created_at": "2026-06-16T09:00:00",
                    "participants": [
                        {
                            "entity_id": "entity-1",
                            "name": "周杰伦",
                            "type": "生命体",
                            "embedding": [1.0, 0.0],
                        }
                    ],
                }
            ]
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=2,
    )

    results = await repository.search_events_by_vector_or_text(
        "user-a", "周杰伦演唱会", [1.0, 0.0], top_k=2
    )

    assert results == [
        MemoryActiveRecallEventResult(
            id="event-1",
            title="参加周杰伦演唱会",
            description="用户参加了周杰伦演唱会",
            event_time="2026-06-15T20:00:00",
            created_at="2026-06-16T09:00:00",
            participants=[
                MemoryTimelineParticipantResult(
                    entity_id="entity-1",
                    name="周杰伦",
                    type="生命体",
                )
            ],
            score=1.0,
        )
    ]
    query, params = driver.executed[-1]
    assert "MATCH (event:Event {user_id: $user_id})" in query
    assert params["user_id"] == "user-a"


@pytest.mark.anyio
async def test_neo4j_repository_returns_graph_view_nodes_and_edges_by_user():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "MATCH (entity:Entity {user_id: $user_id})": [
                _graph_node_row("entity-1")
            ],
            "relation:RELATION {user_id: $user_id}": [
                _graph_edge_row()
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    nodes = await repository.graph_nodes("user-a")
    edges = await repository.graph_edges("user-a")

    assert nodes == [
        MemoryGraphNodeResult(
            id="entity-1",
            name="周杰伦",
            type="生命体",
            description="歌手",
            community_id="community-music",
            importance=0.8,
            memory_layer="long_term",
            access_count=2,
            mention_count=3,
            core_facts=["用户长期喜欢周杰伦"],
            traits=["偏好华语流行"],
        )
    ]
    assert edges == [
        MemoryGraphEdgeResult(
            source="entity-user",
            target="entity-1",
            predicate="偏好",
            evidence="用户喜欢周杰伦。",
            valid_at="2026-06-16T09:00:00",
            invalid_at=None,
            is_current=True,
        )
    ]
    node_query, node_params = driver.executed[0]
    edge_query, edge_params = driver.executed[1]
    assert "MATCH (entity:Entity {user_id: $user_id})" in node_query
    assert "MATCH (source:Entity {user_id: $user_id})" in edge_query
    assert "Event" not in edge_query
    assert "INVOLVES" not in edge_query
    assert "relation.invalid_at IS NULL OR relation.invalid_at > datetime()" in edge_query
    assert node_params == {"user_id": "user-a"}
    assert edge_params == {"user_id": "user-a"}


@pytest.mark.anyio
async def test_neo4j_repository_returns_entity_subgraph_by_user():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "OPTIONAL MATCH (center)-[relation:RELATION]-(neighbor:Entity {user_id: $user_id})": [
                _graph_node_row("entity-1"),
                _graph_node_row("entity-user", name="用户", description="当前用户"),
            ],
            "MATCH (center)-[relation:RELATION]-(neighbor:Entity {user_id: $user_id})": [
                _graph_edge_row()
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    subgraph = await repository.entity_subgraph("user-a", "entity-1")

    assert subgraph == MemoryEntitySubgraphResult(
        center="entity-1",
        nodes=[
            MemoryGraphNodeResult(
                id="entity-1",
                name="周杰伦",
                type="生命体",
                description="歌手",
                community_id="community-music",
                importance=0.8,
                memory_layer="long_term",
                access_count=2,
                mention_count=3,
                core_facts=["用户长期喜欢周杰伦"],
                traits=["偏好华语流行"],
            ),
            MemoryGraphNodeResult(
                id="entity-user",
                name="用户",
                type="生命体",
                description="当前用户",
                community_id="community-music",
                importance=0.8,
                memory_layer="long_term",
                access_count=2,
                mention_count=3,
                core_facts=["用户长期喜欢周杰伦"],
                traits=["偏好华语流行"],
            ),
        ],
        edges=[
            MemoryGraphEdgeResult(
                source="entity-user",
                target="entity-1",
                predicate="偏好",
                evidence="用户喜欢周杰伦。",
                valid_at="2026-06-16T09:00:00",
                invalid_at=None,
                is_current=True,
            )
        ],
    )
    node_query, node_params = driver.executed[0]
    edge_query, edge_params = driver.executed[1]
    assert "MATCH (center:Entity {id: $entity_id, user_id: $user_id})" in node_query
    assert "MATCH (center:Entity {id: $entity_id, user_id: $user_id})" in edge_query
    assert "Event" not in edge_query
    assert "INVOLVES" not in edge_query
    assert "relation.invalid_at IS NULL OR relation.invalid_at > datetime()" in edge_query
    assert node_params == {"user_id": "user-a", "entity_id": "entity-1"}
    assert edge_params == {"user_id": "user-a", "entity_id": "entity-1"}


@pytest.mark.anyio
async def test_neo4j_repository_returns_profile_and_deletes_entities_insights_by_user():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "OPTIONAL MATCH (entity)-[relation:RELATION": [
                {
                    "id": "entity-1",
                    "name": "周杰伦",
                    "type": "生命体",
                    "description": "歌手",
                    "community_id": "community-music",
                    "importance": 0.8,
                    "memory_layer": "long_term",
                    "access_count": 2,
                    "mention_count": 3,
                    "core_facts": ["用户长期喜欢周杰伦"],
                    "traits": ["偏好华语流行"],
                    "relations": [
                        {
                            "predicate": "偏好",
                            "target_entity_id": "entity-1",
                            "target_name": "周杰伦",
                            "target_type": "生命体",
                            "evidence": "用户喜欢周杰伦。",
                            "valid_at": "2026-06-16T09:00:00",
                            "invalid_at": None,
                            "is_current": True,
                        }
                    ],
                }
            ],
            "RETURN entity.type AS type": [{"type": "生命体", "count": 1}],
            "DETACH DELETE entity": [{"deleted": 1}],
            "DETACH DELETE insight": [{"deleted": 1}],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    profile_entities = await repository.profile_entities("user-a")
    type_counts = await repository.entity_type_counts("user-a")
    entity_deleted = await repository.delete_entity("user-a", "entity-1")
    insight_deleted = await repository.delete_insight("user-a", "insight-1")

    assert profile_entities == [
        MemoryProfileEntityResult(
            id="entity-1",
            name="周杰伦",
            type="生命体",
            description="歌手",
            community_id="community-music",
            importance=0.8,
            memory_layer="long_term",
            access_count=2,
            mention_count=3,
            core_facts=["用户长期喜欢周杰伦"],
            traits=["偏好华语流行"],
            relations=[
                MemoryProfileRelationResult(
                    predicate="偏好",
                    target_entity_id="entity-1",
                    target_name="周杰伦",
                    target_type="生命体",
                    evidence="用户喜欢周杰伦。",
                    valid_at="2026-06-16T09:00:00",
                    invalid_at=None,
                    is_current=True,
                )
            ],
        )
    ]
    assert type_counts == {"生命体": 1}
    assert entity_deleted is True
    assert insight_deleted is True
    for query, params in driver.executed:
        assert params["user_id"] == "user-a"
        if "DETACH DELETE entity" in query:
            assert "MATCH (entity:Entity {id: $entity_id, user_id: $user_id})" in query
        if "DETACH DELETE insight" in query:
            assert "MATCH (insight:Insight {id: $insight_id, user_id: $user_id})" in query


@pytest.mark.anyio
async def test_neo4j_repository_returns_relation_history_by_user_and_predicate():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "RETURN center.id AS id": [{"id": "entity-1"}],
            "RETURN relation.id AS relation_id": [
                {
                    "relation_id": "rel-current",
                    "direction": "outgoing",
                    "neighbor_entity_id": "entity-company",
                    "neighbor_name": "腾讯",
                    "neighbor_type": "组织机构",
                    "predicate": "就职于",
                    "evidence": "用户现在在腾讯工作。",
                    "valid_at": "2026-06-16T09:00:00",
                    "invalid_at": None,
                    "is_current": True,
                },
                {
                    "relation_id": "rel-history",
                    "direction": "outgoing",
                    "neighbor_entity_id": "entity-old-company",
                    "neighbor_name": "字节",
                    "neighbor_type": "组织机构",
                    "predicate": "就职于",
                    "evidence": "用户曾经在字节工作。",
                    "valid_at": "2024-01-01T00:00:00",
                    "invalid_at": "2025-01-01T00:00:00",
                    "is_current": False,
                },
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    relations = await repository.entity_relation_history(
        "user-a",
        "entity-1",
        predicate="就职于",
    )

    assert relations == [
        MemoryRelationHistoryResult(
            relation_id="rel-current",
            direction="outgoing",
            neighbor_entity_id="entity-company",
            neighbor_name="腾讯",
            neighbor_type="组织机构",
            predicate="就职于",
            evidence="用户现在在腾讯工作。",
            valid_at="2026-06-16T09:00:00",
            invalid_at=None,
            is_current=True,
        ),
        MemoryRelationHistoryResult(
            relation_id="rel-history",
            direction="outgoing",
            neighbor_entity_id="entity-old-company",
            neighbor_name="字节",
            neighbor_type="组织机构",
            predicate="就职于",
            evidence="用户曾经在字节工作。",
            valid_at="2024-01-01T00:00:00",
            invalid_at="2025-01-01T00:00:00",
            is_current=False,
        ),
    ]
    history_query, params = driver.executed[-1]
    assert "MATCH (center:Entity {id: $entity_id, user_id: $user_id})" in history_query
    assert "relation.name = $predicate" in history_query
    assert "relation.invalid_at IS NULL OR relation.invalid_at > datetime()" in history_query
    assert params == {
        "user_id": "user-a",
        "entity_id": "entity-1",
        "predicate": "就职于",
    }


@pytest.mark.anyio
async def test_neo4j_repository_merges_duplicate_entities_in_single_transaction():
    driver = FakeNeo4jDriver(
        result_rows=[
            {
                "ids": ["entity-1", "entity-dup"],
                "names": ["用户", "用户"],
                "descriptions": ["当前用户", "当前用户重复节点"],
                "core_facts": [["用户喜欢周杰伦"], ["用户喜欢音乐"]],
                "traits": [["偏好华语流行"], ["偏好演唱会"]],
                "access_count": 5,
                "mention_count": 4,
            }
        ]
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    stats = await repository.merge_duplicate_entities("user-a")

    assert stats == MemoryMergeDuplicatesResult(
        removed_entities=1,
        merged_groups=1,
    )
    executed_queries = "\n".join(query for query, _ in driver.executed)
    assert "toLower(entity.name) AS normalized_name" in executed_queries
    assert "MERGE (statement)-[:MENTIONS" in executed_queries
    assert "MERGE (event)-[involves:INVOLVES" in executed_queries
    assert "MERGE (keeper)-[new_relation:RELATION" in executed_queries
    assert "MERGE (source)-[new_relation:RELATION" in executed_queries
    assert "new_relation.valid_at = relation.valid_at" in executed_queries
    assert "new_relation.invalid_at = relation.invalid_at" in executed_queries
    assert "target.id <> $keeper_id" in executed_queries
    assert "source.id <> $keeper_id" in executed_queries
    assert "DETACH DELETE duplicate" in executed_queries
    for _, params in driver.executed:
        assert params["user_id"] == "user-a"


@pytest.mark.anyio
async def test_neo4j_repository_returns_quality_counts_summary_and_issue_samples():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "RETURN dialogues, chunks, statements, entities": [
                {
                    "dialogues": 1,
                    "chunks": 2,
                    "statements": 3,
                    "entities": 4,
                    "relations": 5,
                    "events": 6,
                    "involves": 7,
                    "communities": 8,
                    "insights": 9,
                }
            ],
            "RETURN duplicate_entities, missing_embeddings": [
                {
                    "duplicate_entities": 2,
                    "missing_embeddings": 1,
                    "orphan_entities": 3,
                    "orphan_statements": 4,
                    "broken_relations": 5,
                    "expired_relations": 6,
                    "empty_communities": 7,
                    "orphan_insights": 8,
                }
            ],
            "duplicate_entities' AS category": [
                {
                    "category": "duplicate_entities",
                    "severity": "info",
                    "title": "重复实体",
                    "detail": "用户/生命体 存在 2 个同名节点",
                    "entity_ids": ["entity-1", "entity-dup"],
                    "memory_ids": [],
                    "metadata": {"name": "用户", "type": "生命体", "count": 2},
                }
            ],
            "broken_relations' AS category": [
                {
                    "category": "broken_relations",
                    "severity": "warning",
                    "title": "断裂关系",
                    "detail": "关系 rel-1 缺少来源陈述",
                    "entity_ids": ["entity-user", "entity-company"],
                    "memory_ids": [],
                    "metadata": {"relation_id": "rel-1", "predicate": "就职于"},
                }
            ],
            "expired_relations' AS category": [
                {
                    "category": "expired_relations",
                    "severity": "info",
                    "title": "已失效关系",
                    "detail": "关系 rel-old 已失效",
                    "entity_ids": ["entity-user", "entity-old-company"],
                    "memory_ids": [],
                    "metadata": {
                        "relation_id": "rel-old",
                        "predicate": "就职于",
                        "invalid_at": "2025-01-01T00:00:00",
                    },
                }
            ],
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    counts = await repository.quality_graph_counts("user-a")
    summary = await repository.quality_issue_summary("user-a")
    duplicate_issues = await repository.quality_issues(
        "user-a", "duplicate_entities", limit=50
    )
    broken_issues = await repository.quality_issues(
        "user-a", "broken_relations", limit=50
    )
    expired_issues = await repository.quality_issues(
        "user-a", "expired_relations", limit=50
    )

    assert counts == MemoryQualityGraphCountsResult(
        dialogues=1,
        chunks=2,
        statements=3,
        entities=4,
        relations=5,
        events=6,
        involves=7,
        communities=8,
        insights=9,
    )
    assert summary == MemoryQualityIssueSummaryResult(
        duplicate_entities=2,
        missing_embeddings=1,
        orphan_entities=3,
        orphan_statements=4,
        broken_relations=5,
        expired_relations=6,
        empty_communities=7,
        orphan_insights=8,
    )
    assert duplicate_issues == MemoryQualityIssueListResult(
        category="duplicate_entities",
        total=1,
        items=[
            MemoryQualityIssueResult(
                category="duplicate_entities",
                severity="info",
                title="重复实体",
                detail="用户/生命体 存在 2 个同名节点",
                entity_ids=["entity-1", "entity-dup"],
                memory_ids=[],
                metadata={"name": "用户", "type": "生命体", "count": 2},
            )
        ],
    )
    assert broken_issues.items[0].severity == "warning"
    assert expired_issues.items[0].metadata["invalid_at"] == "2025-01-01T00:00:00"
    executed_queries = "\n".join(query for query, _ in driver.executed)
    assert "MATCH (dialogue:Dialogue {user_id: $user_id})" in executed_queries
    assert "toLower(entity.name) AS normalized_name" in executed_queries
    assert "entity.type AS entity_type" in executed_queries
    assert "relation.statement_id IS NULL" in executed_queries
    assert "Statement {id: relation.statement_id, user_id: $user_id}" in executed_queries
    assert "relation.invalid_at <= datetime()" in executed_queries
    for _, params in driver.executed:
        assert params["user_id"] == "user-a"


@pytest.mark.anyio
async def test_neo4j_repository_returns_memory_trace_by_user_and_memory_id():
    driver = FakeNeo4jDriver(
        result_rows_by_keyword={
            "MATCH (dialogue:Dialogue {user_id: $user_id, memory_id: $memory_id})": [
                {
                    "dialogue": {
                        "id": "dialogue-1",
                        "memory_id": "mem-1",
                        "summary": "用户喜欢周杰伦。",
                        "created_at": "2026-06-16T09:00:00",
                    },
                    "chunks": [
                        {
                            "id": "chunk-1",
                            "index": 0,
                            "text": "用户喜欢周杰伦。",
                        }
                    ],
                    "statements": [
                        {
                            "id": "statement-1",
                            "chunk_id": "chunk-1",
                            "index": 0,
                            "text": "用户喜欢周杰伦。",
                            "statement_type": "FACT",
                            "temporal_type": "STATIC",
                            "importance": 0.8,
                            "confidence": 0.9,
                            "valid_at": "2026-06-16T09:00:00",
                            "invalid_at": None,
                            "memory_layer": "short_term",
                        }
                    ],
                    "entities": [
                        {
                            "id": "entity-user",
                            "name": "用户",
                            "type": "生命体",
                            "description": "当前用户",
                        },
                        {
                            "id": "entity-jay",
                            "name": "周杰伦",
                            "type": "生命体",
                            "description": "歌手",
                        },
                    ],
                    "mentions": [
                        {
                            "id": "mention-1",
                            "statement_id": "statement-1",
                            "entity_id": "entity-jay",
                        }
                    ],
                    "relations": [
                        {
                            "id": "rel-1",
                            "source_entity_id": "entity-user",
                            "source_name": "用户",
                            "target_entity_id": "entity-jay",
                            "target_name": "周杰伦",
                            "name": "偏好",
                            "evidence": "用户喜欢周杰伦。",
                            "statement_id": "statement-1",
                            "valid_at": "2026-06-16T09:00:00",
                            "invalid_at": None,
                            "is_current": True,
                        }
                    ],
                    "events": [
                        {
                            "id": "event-1",
                            "title": "参加周杰伦演唱会",
                            "description": "用户参加了周杰伦演唱会",
                            "event_time": "2026-06-15T20:00:00",
                            "created_at": "2026-06-16T09:00:00",
                            "participants": [
                                {
                                    "entity_id": "entity-jay",
                                    "name": "周杰伦",
                                    "type": "生命体",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )
    repository = Neo4jMemoryGraphRepository(
        driver=driver,
        database="neo4j",
        embedding_dims=1024,
    )

    trace = await repository.memory_trace("user-a", "mem-1")

    assert trace == MemoryTraceResult(
        dialogue=MemoryTraceDialogueResult(
            id="dialogue-1",
            memory_id="mem-1",
            summary="用户喜欢周杰伦。",
            created_at="2026-06-16T09:00:00",
        ),
        chunks=[
            MemoryTraceChunkResult(
                id="chunk-1",
                index=0,
                text="用户喜欢周杰伦。",
            )
        ],
        statements=[
            MemoryTraceStatementResult(
                id="statement-1",
                chunk_id="chunk-1",
                index=0,
                text="用户喜欢周杰伦。",
                statement_type="FACT",
                temporal_type="STATIC",
                importance=0.8,
                confidence=0.9,
                valid_at="2026-06-16T09:00:00",
                invalid_at=None,
                memory_layer="short_term",
            )
        ],
        entities=[
            MemoryTraceEntityResult(
                id="entity-user",
                name="用户",
                type="生命体",
                description="当前用户",
            ),
            MemoryTraceEntityResult(
                id="entity-jay",
                name="周杰伦",
                type="生命体",
                description="歌手",
            ),
        ],
        mentions=[
            MemoryTraceMentionResult(
                id="mention-1",
                statement_id="statement-1",
                entity_id="entity-jay",
            )
        ],
        relations=[
            MemoryTraceRelationResult(
                id="rel-1",
                source_entity_id="entity-user",
                source_name="用户",
                target_entity_id="entity-jay",
                target_name="周杰伦",
                name="偏好",
                evidence="用户喜欢周杰伦。",
                statement_id="statement-1",
                valid_at="2026-06-16T09:00:00",
                invalid_at=None,
                is_current=True,
            )
        ],
        events=[
            MemoryTraceEventResult(
                id="event-1",
                title="参加周杰伦演唱会",
                description="用户参加了周杰伦演唱会",
                event_time="2026-06-15T20:00:00",
                created_at="2026-06-16T09:00:00",
                participants=[
                    MemoryTimelineParticipantResult(
                        entity_id="entity-jay",
                        name="周杰伦",
                        type="生命体",
                    )
                ],
            )
        ],
    )
    query, params = driver.executed[-1]
    assert "MATCH (dialogue:Dialogue {user_id: $user_id, memory_id: $memory_id})" in query
    assert "MATCH (event:Event {user_id: $user_id})" in query
    assert "WHERE event.dialogue_id = dialogue.id" in query
    assert params == {"user_id": "user-a", "memory_id": "mem-1"}


@pytest.mark.anyio
async def test_neo4j_repository_returns_none_when_memory_trace_is_missing():
    repository = Neo4jMemoryGraphRepository(
        driver=FakeNeo4jDriver(result_rows=[]),
        database="neo4j",
        embedding_dims=1024,
    )

    assert await repository.memory_trace("user-a", "missing") is None


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


def _graph_node_row(entity_id: str, *, name: str = "周杰伦", description: str = "歌手"):
    return {
        "id": entity_id,
        "name": name,
        "type": "生命体",
        "description": description,
        "community_id": "community-music",
        "importance": 0.8,
        "memory_layer": "long_term",
        "access_count": 2,
        "mention_count": 3,
        "core_facts": ["用户长期喜欢周杰伦"],
        "traits": ["偏好华语流行"],
    }


def _graph_edge_row():
    return {
        "source": "entity-user",
        "target": "entity-1",
        "predicate": "偏好",
        "evidence": "用户喜欢周杰伦。",
        "valid_at": "2026-06-16T09:00:00",
        "invalid_at": None,
        "is_current": True,
    }
