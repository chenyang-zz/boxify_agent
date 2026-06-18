import pytest

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.application.services.memory_service import MemoryService
from app.domain.models.long_term_memory import LongTermMemory, MemorySource, MemoryStatus
from app.domain.models.memory_graph import (
    CommunityMemberResult,
    CommunityRelationResult,
    CommunityResult,
    CommunityVoteEntity,
    EntityNode,
    GraphRelationFact,
    InsightResult,
    MemoryConsolidationStats,
    MemoryEntitySubgraphResult,
    MemoryGraphEdgeResult,
    MemoryGraphNodeResult,
    MemoryGraphResult,
    MemoryGraphViewResult,
    MemoryMergeDuplicatesResult,
    MemoryProfileEntityResult,
    MemoryProfileGroupResult,
    MemoryProfileRelationResult,
    MemoryProfileResult,
    MemoryPromotionStats,
    MemoryQualityGraphCountsResult,
    MemoryQualityIssueListResult,
    MemoryQualityIssueResult,
    MemoryQualityIssueSummaryResult,
    MemoryTraceDialogueResult,
    MemoryTraceResult,
    MemoryTimelineEventResult,
    MemoryTimelineParticipantResult,
)
from app.domain.services.memory import LongTermMemoryManager, MemoryConsolidator
from app.domain.services.memory.community_clusterer import MemoryCommunityClusterer
from app.domain.services.memory.profile_summarizer import MemoryProfileSummarizer


@pytest.mark.anyio
async def test_long_term_memory_manager_remembers_text_for_current_user():
    repository = InMemoryMemoryRepository()
    dispatcher = FakeTaskDispatcher()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        task_dispatcher=dispatcher,
    )

    memory = await manager.remember_text("我喜欢周杰伦的歌", source=MemorySource.MANUAL)

    assert memory.user_id == "user-a"
    assert memory.source == MemorySource.MANUAL
    assert memory.content == "我喜欢周杰伦的歌"
    assert memory.status == MemoryStatus.PENDING
    assert memory.summary == "我喜欢周杰伦的歌"
    assert repository.saved[0] == memory
    assert dispatcher.extract_memory_calls == [memory.id]


@pytest.mark.anyio
async def test_long_term_memory_manager_search_is_scoped_to_current_user():
    repository = InMemoryMemoryRepository()
    user_a_memory = LongTermMemory(
        user_id="user-a",
        content="我喜欢周杰伦的歌",
        summary="用户喜欢周杰伦的歌",
    )
    user_b_memory = LongTermMemory(
        user_id="user-b",
        content="我喜欢林俊杰的歌",
        summary="用户喜欢林俊杰的歌",
    )
    await repository.save(user_a_memory)
    await repository.save(user_b_memory)
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    results = await manager.search("周杰伦", top_k=10)

    assert results == [user_a_memory]


@pytest.mark.anyio
async def test_long_term_memory_manager_prefers_graph_results():
    repository = InMemoryMemoryRepository()
    graph_repository = FakeGraphRepository(
        [
            MemoryGraphResult(
                entity_id="entity-1",
                entity_name="周杰伦",
                entity_type="Person",
                description="歌手",
                score=0.91,
                source_memory_id="mem-1",
                source_memory_summary="用户喜欢周杰伦的歌",
                relations=[
                    GraphRelationFact(
                        name="LIKES",
                        direction="incoming",
                        neighbor_name="用户",
                        neighbor_type="Person",
                        evidence="用户喜欢周杰伦的歌",
                    )
                ],
            )
        ]
    )
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=graph_repository,
        embedding=FakeEmbedding(),
    )

    results = await manager.search("喜欢的歌手", top_k=3)

    assert len(results) == 1
    assert results[0].content == "用户喜欢周杰伦的歌"
    assert results[0].graph_data is not None
    assert results[0].graph_data.entity_name == "周杰伦"
    assert results[0].graph_data.relations[0].evidence == "用户喜欢周杰伦的歌"
    assert graph_repository.calls == [("user-a", "喜欢的歌手", 3, [1.0])]


@pytest.mark.anyio
async def test_long_term_memory_manager_falls_back_to_pg_when_graph_has_no_result():
    repository = InMemoryMemoryRepository()
    pg_memory = LongTermMemory(
        user_id="user-a",
        content="我喜欢周杰伦的歌",
        summary="用户喜欢周杰伦",
    )
    await repository.save(pg_memory)
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=FakeGraphRepository([]),
        embedding=FakeEmbedding(),
    )

    results = await manager.search("周杰伦", top_k=3)

    assert results == [pg_memory]


@pytest.mark.anyio
async def test_long_term_memory_manager_falls_back_to_pg_when_graph_errors():
    repository = InMemoryMemoryRepository()
    pg_memory = LongTermMemory(
        user_id="user-a",
        content="我喜欢周杰伦的歌",
        summary="用户喜欢周杰伦",
    )
    await repository.save(pg_memory)
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=ExplodingGraphRepository(),
        embedding=FakeEmbedding(),
    )

    results = await manager.search("周杰伦", top_k=3)

    assert results == [pg_memory]


@pytest.mark.anyio
async def test_long_term_memory_manager_uses_graph_fulltext_when_embedding_fails():
    repository = InMemoryMemoryRepository()
    graph_repository = FakeGraphRepository(
        [
            MemoryGraphResult(
                entity_id="entity-1",
                entity_name="周杰伦",
                entity_type="生命体",
                description="歌手",
                score=0.7,
                source_memory_summary="用户喜欢周杰伦",
            )
        ]
    )
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        graph_repository=graph_repository,
        embedding=ExplodingEmbedding(),
    )

    results = await manager.search("周杰伦", top_k=3)

    assert results[0].graph_data is not None
    assert results[0].graph_data.entity_name == "周杰伦"
    assert graph_repository.calls == [("user-a", "周杰伦", 3, None)]


@pytest.mark.anyio
async def test_memory_consolidation_service_promotes_and_enhances_profiles():
    repository = FakeConsolidationGraphRepository()
    service = MemoryConsolidator(
        user_id="user-a",
        graph_repository=repository,
        profile_summarizer=FakeProfileSummarizer(
            core_facts=["用户长期喜欢周杰伦"],
            traits=["偏好华语流行"],
        ),
    )

    stats = await service.consolidate()

    assert stats == MemoryConsolidationStats(
        promoted_entities=2,
        promoted_statements=3,
        enhanced_profiles=1,
    )
    assert repository.promote_calls == [
        {
            "user_id": "user-a",
            "min_access": 3,
            "min_importance": 0.8,
            "min_mention": 3,
        }
    ]
    assert repository.profile_writes == [
        (
            "user-a",
            "entity-1",
            ["用户长期喜欢周杰伦"],
            ["偏好华语流行"],
        )
    ]


@pytest.mark.anyio
async def test_memory_consolidation_service_skips_single_profile_failure():
    repository = FakeConsolidationGraphRepository(
        top_entities=[
            EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
            EntityNode(id="entity-2", user_id="user-a", name="林俊杰", type="生命体"),
        ]
    )
    service = MemoryConsolidator(
        user_id="user-a",
        graph_repository=repository,
        profile_summarizer=FlakyProfileSummarizer(),
    )

    stats = await service.consolidate()

    assert stats.enhanced_profiles == 1
    assert repository.profile_writes == [
        ("user-a", "entity-2", ["可用事实"], ["可用特质"])
    ]


@pytest.mark.anyio
async def test_memory_profile_summarizer_parses_profile_json():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM(
            '{"core_facts":["用户长期喜欢周杰伦"],"traits":["偏好华语流行"]}'
        ),
        json_parser=FakeJSONParser(),
    )

    core_facts, traits = await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    )

    assert core_facts == ["用户长期喜欢周杰伦"]
    assert traits == ["偏好华语流行"]


@pytest.mark.anyio
async def test_memory_profile_summarizer_falls_back_for_bad_json():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM("not-json"),
        json_parser=FakeJSONParser(),
    )

    assert await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    ) == ([], [])


@pytest.mark.anyio
async def test_memory_profile_summarizer_falls_back_for_non_dict_json():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM('["not-object"]'),
        json_parser=FakeJSONParser(),
    )

    assert await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    ) == ([], [])


@pytest.mark.anyio
async def test_memory_profile_summarizer_coerces_short_string_lists():
    summarizer = MemoryProfileSummarizer(
        llm=FakeProfileLLM(
            {
                "core_facts": ["  稳定事实  ", "", "x" * 250],
                "traits": ["偏好华语流行", None],
            }
        ),
        json_parser=FakeJSONParser(),
    )

    core_facts, traits = await summarizer.summarize(
        EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体"),
        ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"],
    )

    assert core_facts == ["稳定事实", "x" * 200]
    assert traits == ["偏好华语流行", "None"]


@pytest.mark.anyio
async def test_long_term_memory_manager_remembers_session_source():
    repository = InMemoryMemoryRepository()
    dispatcher = FakeTaskDispatcher()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository),
        user_id="user-a",
        task_dispatcher=dispatcher,
    )

    memory = await manager.remember_text(
        "用户喜欢安静的工作环境",
        source=MemorySource.SESSION,
        source_session_id="session-1",
    )

    assert memory.source == MemorySource.SESSION
    assert memory.source_session_id == "session-1"
    assert memory.keywords == ["用户喜欢安静的工作环境"]
    assert dispatcher.extract_memory_calls == [memory.id]


@pytest.mark.anyio
async def test_long_term_memory_manager_delete_returns_false_when_missing():
    repository = InMemoryMemoryRepository()
    manager = LongTermMemoryManager(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    assert await manager.delete_memory("missing") is False


@pytest.mark.anyio
async def test_application_memory_service_converts_empty_content_to_bad_request():
    repository = InMemoryMemoryRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    with pytest.raises(BadRequestError) as exc:
        await service.remember_text("   ")

    assert exc.value.msg == "记忆内容不能为空"


@pytest.mark.anyio
async def test_application_memory_service_converts_empty_query_to_bad_request():
    repository = InMemoryMemoryRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    with pytest.raises(BadRequestError) as exc:
        await service.search("   ", top_k=5)

    assert exc.value.msg == "检索关键词不能为空"


@pytest.mark.anyio
async def test_application_memory_service_converts_missing_delete_to_not_found():
    repository = InMemoryMemoryRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(repository), user_id="user-a"
    )

    with pytest.raises(NotFoundError) as exc:
        await service.delete_memory("missing")

    assert exc.value.msg == "记忆不存在或无权访问"


@pytest.mark.anyio
async def test_application_memory_service_clusters_and_lists_communities():
    repository = FakeCommunityServiceGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    stats = await service.cluster()
    communities = await service.list_communities()
    members, relationships = await service.community_detail("community-music")

    assert stats.assigned_entities == 1
    assert communities == [
        CommunityResult(
            id="community-music",
            name="音乐偏好",
            summary="用户的音乐相关实体",
            member_count=1,
        )
    ]
    assert members[0].entity_name == "周杰伦"
    assert relationships[0].name == "偏好"


@pytest.mark.anyio
async def test_application_memory_service_lists_timeline_events():
    repository = FakeTimelineGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    events = await service.timeline(limit=25)

    assert repository.calls == [("user-a", 25)]
    assert events == [
        MemoryTimelineEventResult(
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
        )
    ]


@pytest.mark.anyio
async def test_application_memory_service_returns_graph_view():
    repository = FakeGraphViewRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    graph = await service.graph()

    assert graph == MemoryGraphViewResult(
        nodes=repository.nodes,
        edges=repository.edges,
        communities=repository.communities,
    )
    assert repository.calls == [
        ("graph_nodes", "user-a"),
        ("graph_edges", "user-a"),
        ("list_communities", "user-a"),
    ]


@pytest.mark.anyio
async def test_application_memory_service_entity_subgraph_requires_visible_center():
    repository = FakeGraphViewRepository(
        subgraph=MemoryEntitySubgraphResult(
            center="missing",
            nodes=[],
            edges=[],
        )
    )
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    with pytest.raises(NotFoundError) as exc:
        await service.entity_subgraph("missing")

    assert exc.value.msg == "实体不存在或无权访问"


@pytest.mark.anyio
async def test_application_memory_service_returns_profile_and_insights():
    repository = FakeMemoryManagementGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    profile = await service.profile()
    insights = await service.list_insights()

    assert profile == MemoryProfileResult(
        total=1,
        type_counts={"生命体": 1},
        groups=[
            MemoryProfileGroupResult(
                type="生命体",
                entities=[
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
                            )
                        ],
                    )
                ],
            )
        ],
    )
    assert insights == [
        InsightResult(
            id="insight-1",
            theme="音乐偏好",
            content="用户偏好华语流行音乐。",
            importance=0.8,
            confidence=0.9,
            source_count=2,
        )
    ]
    assert repository.calls == [
        ("profile_entities", "user-a"),
        ("entity_type_counts", "user-a"),
        ("list_insights", "user-a"),
    ]


@pytest.mark.anyio
async def test_application_memory_service_deletes_entity_and_insight_or_raises_not_found():
    repository = FakeMemoryManagementGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    await service.delete_entity("entity-1")
    await service.delete_insight("insight-1")

    assert repository.calls == [
        ("delete_entity", "user-a", "entity-1"),
        ("delete_insight", "user-a", "insight-1"),
    ]

    with pytest.raises(NotFoundError) as entity_exc:
        await service.delete_entity("missing")
    with pytest.raises(NotFoundError) as insight_exc:
        await service.delete_insight("missing")

    assert entity_exc.value.msg == "实体不存在或无权访问"
    assert insight_exc.value.msg == "洞察不存在或无权访问"


@pytest.mark.anyio
async def test_application_memory_service_merges_duplicate_entities():
    repository = FakeMemoryManagementGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    stats = await service.merge_duplicates()

    assert stats == MemoryMergeDuplicatesResult(removed_entities=2, merged_groups=1)
    assert repository.calls == [("merge_duplicate_entities", "user-a")]


@pytest.mark.anyio
async def test_application_memory_service_returns_quality_overview():
    memory_repository = InMemoryMemoryRepository()
    completed = LongTermMemory(user_id="user-a", content="完成记忆")
    completed.mark_completed()
    failed = LongTermMemory(user_id="user-a", content="失败记忆")
    failed.mark_failed("LLM timeout")
    other_user = LongTermMemory(user_id="user-b", content="其他用户")
    other_user.mark_failed("should not leak")
    await memory_repository.save(completed)
    await memory_repository.save(failed)
    await memory_repository.save(other_user)
    repository = FakeQualityGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(memory_repository),
        user_id="user-a",
        graph_repository=repository,
    )

    quality = await service.quality()

    assert quality.pg_total == 2
    assert quality.pg_status_counts["completed"] == 1
    assert quality.pg_status_counts["failed"] == 1
    assert quality.recent_failed[0].id == failed.id
    assert quality.recent_failed[0].error_msg == "LLM timeout"
    assert quality.graph_available is True
    assert quality.graph_counts.entities == 8
    assert quality.issue_summary.duplicate_entities == 2
    assert repository.calls == [
        ("quality_graph_counts", "user-a"),
        ("quality_issue_summary", "user-a"),
    ]


@pytest.mark.anyio
async def test_application_memory_service_returns_memory_detail_with_trace():
    memory_repository = InMemoryMemoryRepository()
    memory = LongTermMemory(
        id="memory-1",
        user_id="user-a",
        content="用户喜欢周杰伦。",
        status=MemoryStatus.COMPLETED,
        summary="用户喜欢周杰伦。",
        graph_dialogue_id="dialogue-1",
    )
    await memory_repository.save(memory)
    repository = FakeTraceGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(memory_repository),
        user_id="user-a",
        graph_repository=repository,
    )

    detail = await service.detail("memory-1")

    assert detail.id == "memory-1"
    assert detail.content == "用户喜欢周杰伦。"
    assert detail.graph_available is True
    assert detail.trace == MemoryTraceResult(
        dialogue=MemoryTraceDialogueResult(
            id="dialogue-1",
            memory_id="memory-1",
            summary="用户喜欢周杰伦。",
        )
    )
    assert repository.calls == [("memory_trace", "user-a", "memory-1")]


@pytest.mark.anyio
async def test_application_memory_service_detail_keeps_pg_when_graph_unavailable():
    memory_repository = InMemoryMemoryRepository()
    memory = LongTermMemory(
        id="memory-1",
        user_id="user-a",
        content="待萃取记忆",
        status=MemoryStatus.PENDING,
    )
    await memory_repository.save(memory)
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(memory_repository),
        user_id="user-a",
        graph_repository=ExplodingTraceGraphRepository(),
    )

    detail = await service.detail("memory-1")

    assert detail.id == "memory-1"
    assert detail.graph_available is False
    assert detail.trace is None


@pytest.mark.anyio
async def test_application_memory_service_detail_raises_not_found_for_cross_user_memory():
    memory_repository = InMemoryMemoryRepository()
    await memory_repository.save(
        LongTermMemory(
            id="memory-other",
            user_id="user-b",
            content="其他用户记忆",
        )
    )
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(memory_repository),
        user_id="user-a",
        graph_repository=FakeTraceGraphRepository(),
    )

    with pytest.raises(NotFoundError) as exc:
        await service.detail("memory-other")
    assert exc.value.msg == "记忆不存在或无权访问"


@pytest.mark.anyio
async def test_application_memory_service_keeps_pg_quality_when_graph_unavailable():
    memory_repository = InMemoryMemoryRepository()
    failed = LongTermMemory(user_id="user-a", content="失败记忆")
    failed.mark_failed("Neo4j unavailable")
    await memory_repository.save(failed)
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(memory_repository),
        user_id="user-a",
        graph_repository=ExplodingQualityGraphRepository(),
    )

    quality = await service.quality()

    assert quality.pg_total == 1
    assert quality.pg_status_counts["failed"] == 1
    assert quality.graph_available is False
    assert quality.graph_counts.entities == 0
    assert quality.issue_summary.broken_relations == 0


@pytest.mark.anyio
async def test_application_memory_service_returns_quality_issue_samples():
    repository = FakeQualityGraphRepository()
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=repository,
    )

    issues = await service.quality_issues("duplicate_entities", limit=500)

    assert issues.category == "duplicate_entities"
    assert issues.total == 1
    assert issues.items[0].entity_ids == ["entity-1", "entity-dup"]
    assert repository.calls == [("quality_issues", "user-a", "duplicate_entities", 200)]


@pytest.mark.anyio
async def test_application_memory_service_rejects_unknown_quality_issue_category():
    service = MemoryService(
        uow_factory=lambda: MemoryUnitOfWork(InMemoryMemoryRepository()),
        user_id="user-a",
        graph_repository=FakeQualityGraphRepository(),
    )

    with pytest.raises(BadRequestError) as exc:
        await service.quality_issues("unknown", limit=50)

    assert exc.value.msg == "未知质量问题类别"


@pytest.mark.anyio
async def test_memory_community_clusterer_is_available_for_service_layer():
    repository = FakeCommunityServiceGraphRepository()
    clusterer = MemoryCommunityClusterer(user_id="user-a", graph_repository=repository)

    stats = await clusterer.cluster()

    assert stats.assigned_entities == 1
    assert repository.clustered is True


class InMemoryMemoryRepository:
    def __init__(self):
        self.saved = []

    async def save(self, memory):
        self.saved.append(memory)

    async def get_by_user(self, user_id: str, memory_id: str):
        for memory in self.saved:
            if memory.user_id == user_id and memory.id == memory_id:
                return memory
        return None

    async def get_user_id_by_memory_id(self, memory_id: str):
        for memory in self.saved:
            if memory.id == memory_id:
                return memory.user_id
        return None

    async def list_by_user(self, user_id: str, page: int, page_size: int):
        memories = [memory for memory in self.saved if memory.user_id == user_id]
        return memories[(page - 1) * page_size : page * page_size], len(memories)

    async def search_by_user(self, user_id: str, query: str, top_k: int):
        return [
            memory
            for memory in self.saved
            if memory.user_id == user_id
            and (query in memory.content or (memory.summary and query in memory.summary))
        ][:top_k]

    async def delete_by_user(self, user_id: str, memory_id: str):
        before = len(self.saved)
        self.saved = [
            memory
            for memory in self.saved
            if not (memory.user_id == user_id and memory.id == memory_id)
        ]
        return len(self.saved) != before

    async def status_counts(self, user_id: str):
        counts = {status.value: 0 for status in MemoryStatus}
        for memory in self.saved:
            if memory.user_id == user_id:
                counts[memory.status.value] += 1
        return counts

    async def recent_failed(self, user_id: str, limit: int):
        from app.domain.models.memory_graph import MemoryQualityFailedMemoryResult

        failed = [
            memory
            for memory in self.saved
            if memory.user_id == user_id and memory.status == MemoryStatus.FAILED
        ]
        return [
            MemoryQualityFailedMemoryResult(
                id=memory.id,
                content=memory.content,
                error_msg=memory.error_msg,
                updated_at=memory.updated_at,
            )
            for memory in failed[:limit]
        ]


class FakeTaskDispatcher:
    def __init__(self):
        self.extract_memory_calls = []

    async def dispatch_parse_document(self, document_id: str) -> None:
        raise AssertionError("memory tests should not dispatch document parsing")

    async def dispatch_extract_memory(self, memory_id: str) -> None:
        self.extract_memory_calls.append(memory_id)


class FakeGraphRepository:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def save_graph(self, graph):
        raise AssertionError("memory manager tests should not save graphs")

    async def search(self, user_id: str, query: str, top_k: int, query_embedding=None):
        self.calls.append((user_id, query, top_k, query_embedding))
        return self.results


class ExplodingGraphRepository:
    async def save_graph(self, graph):
        raise AssertionError("memory manager tests should not save graphs")

    async def search(self, user_id: str, query: str, top_k: int, query_embedding=None):
        raise RuntimeError("neo4j unavailable")


class FakeEmbedding:
    async def embed(self, texts):
        return [[1.0] for _ in texts]

    async def embed_one(self, text):
        return [1.0]

    @property
    def model_name(self):
        return "fake-embedding"


class ExplodingEmbedding:
    async def embed(self, texts):
        raise RuntimeError("embedding unavailable")

    async def embed_one(self, text):
        raise RuntimeError("embedding unavailable")

    @property
    def model_name(self):
        return "exploding-embedding"


class FakeConsolidationGraphRepository(FakeGraphRepository):
    def __init__(self, top_entities=None):
        super().__init__([])
        self.promote_calls = []
        self.profile_writes = []
        self.top_entities = top_entities or [
            EntityNode(id="entity-1", user_id="user-a", name="周杰伦", type="生命体")
        ]

    async def promote_short_to_long(
        self, user_id, min_access, min_importance, min_mention, age_before
    ):
        self.promote_calls.append(
            {
                "user_id": user_id,
                "min_access": min_access,
                "min_importance": min_importance,
                "min_mention": min_mention,
            }
        )
        assert age_before
        return MemoryPromotionStats(promoted_entities=2, promoted_statements=3)

    async def top_long_term_entities(self, user_id, top_k):
        assert user_id == "user-a"
        assert top_k == 20
        return self.top_entities

    async def entity_statements(self, user_id, entity_id):
        assert user_id == "user-a"
        return ["用户喜欢周杰伦。", "用户经常听周杰伦的歌。"]

    async def write_entity_profile(self, user_id, entity_id, core_facts, traits):
        self.profile_writes.append((user_id, entity_id, core_facts, traits))


class FakeCommunityServiceGraphRepository(FakeGraphRepository):
    def __init__(self):
        super().__init__([])
        self.clustered = False

    async def has_communities(self, user_id):
        return False

    async def dialogue_entity_ids(self, user_id, dialogue_id):
        return []

    async def community_vote_entities(self, user_id, entity_ids=None):
        return [
            CommunityVoteEntity(
                id="entity-1",
                user_id="user-a",
                name="周杰伦",
                type="生命体",
                embedding=[1.0],
            )
        ]

    async def community_vote_neighbors(self, user_id, entity_ids):
        return {"entity-1": []}

    async def upsert_community(self, user_id, community_id):
        return None

    async def assign_entity_community(self, user_id, entity_id, community_id):
        self.clustered = True

    async def refresh_community_member_count(self, user_id, community_id):
        return 1

    async def community_members(self, user_id, community_id):
        return [
            CommunityMemberResult(
                entity_id="entity-1",
                entity_name="周杰伦",
                entity_type="生命体",
                description="歌手",
                community_id=community_id,
            )
        ]

    async def community_relationships(self, user_id, community_id):
        return [
            CommunityRelationResult(
                source_entity_id="entity-user",
                source_name="用户",
                target_entity_id="entity-1",
                target_name="周杰伦",
                name="偏好",
                evidence="用户喜欢周杰伦。",
            )
        ]

    async def update_community_metadata(self, user_id, community_id, name, summary):
        return None

    async def list_communities(self, user_id):
        return [
            CommunityResult(
                id="community-music",
                name="音乐偏好",
                summary="用户的音乐相关实体",
                member_count=1,
            )
        ]

    async def prune_empty_communities(self, user_id):
        return None


class FakeTimelineGraphRepository(FakeGraphRepository):
    def __init__(self):
        super().__init__([])
        self.calls = []

    async def event_timeline(self, user_id, limit):
        self.calls.append((user_id, limit))
        return [
            MemoryTimelineEventResult(
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
            )
        ]


class FakeGraphViewRepository(FakeGraphRepository):
    def __init__(self, subgraph=None):
        super().__init__([])
        self.nodes = [
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
        self.edges = [
            MemoryGraphEdgeResult(
                source="entity-user",
                target="entity-1",
                predicate="偏好",
                evidence="用户喜欢周杰伦。",
            )
        ]
        self.communities = [
            CommunityResult(
                id="community-music",
                name="音乐偏好",
                summary="用户的音乐相关实体",
                member_count=1,
            )
        ]
        self.subgraph = subgraph or MemoryEntitySubgraphResult(
            center="entity-1",
            nodes=self.nodes,
            edges=self.edges,
        )

    async def graph_nodes(self, user_id):
        self.calls.append(("graph_nodes", user_id))
        return self.nodes

    async def graph_edges(self, user_id):
        self.calls.append(("graph_edges", user_id))
        return self.edges

    async def entity_subgraph(self, user_id, entity_id):
        self.calls.append(("entity_subgraph", user_id, entity_id))
        return self.subgraph

    async def list_communities(self, user_id):
        self.calls.append(("list_communities", user_id))
        return self.communities


class FakeMemoryManagementGraphRepository(FakeGraphRepository):
    def __init__(self):
        super().__init__([])
        self.calls = []

    async def profile_entities(self, user_id):
        self.calls.append(("profile_entities", user_id))
        return [
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
                    )
                ],
            )
        ]

    async def entity_type_counts(self, user_id):
        self.calls.append(("entity_type_counts", user_id))
        return {"生命体": 1}

    async def list_insights(self, user_id):
        self.calls.append(("list_insights", user_id))
        return [
            InsightResult(
                id="insight-1",
                theme="音乐偏好",
                content="用户偏好华语流行音乐。",
                importance=0.8,
                confidence=0.9,
                source_count=2,
            )
        ]

    async def delete_entity(self, user_id, entity_id):
        self.calls.append(("delete_entity", user_id, entity_id))
        return entity_id != "missing"

    async def delete_insight(self, user_id, insight_id):
        self.calls.append(("delete_insight", user_id, insight_id))
        return insight_id != "missing"

    async def merge_duplicate_entities(self, user_id):
        self.calls.append(("merge_duplicate_entities", user_id))
        return MemoryMergeDuplicatesResult(removed_entities=2, merged_groups=1)


class FakeQualityGraphRepository(FakeGraphRepository):
    def __init__(self):
        super().__init__([])
        self.calls = []

    async def quality_graph_counts(self, user_id):
        self.calls.append(("quality_graph_counts", user_id))
        return MemoryQualityGraphCountsResult(
            dialogues=2,
            chunks=3,
            statements=4,
            entities=8,
            relations=6,
            events=1,
            involves=2,
            communities=1,
            insights=1,
        )

    async def quality_issue_summary(self, user_id):
        self.calls.append(("quality_issue_summary", user_id))
        return MemoryQualityIssueSummaryResult(
            duplicate_entities=2,
            missing_embeddings=1,
            broken_relations=1,
        )

    async def quality_issues(self, user_id, category, limit):
        self.calls.append(("quality_issues", user_id, category, limit))
        return MemoryQualityIssueListResult(
            category=category,
            total=1,
            items=[
                MemoryQualityIssueResult(
                    category=category,
                    severity="info",
                    title="重复实体",
                    detail="用户/生命体 存在重复节点",
                    entity_ids=["entity-1", "entity-dup"],
                    metadata={"name": "用户", "type": "生命体", "count": 2},
                )
            ],
        )


class ExplodingQualityGraphRepository(FakeQualityGraphRepository):
    async def quality_graph_counts(self, user_id):
        raise RuntimeError("neo4j unavailable")


class FakeTraceGraphRepository(FakeGraphRepository):
    def __init__(self, trace=None):
        super().__init__([])
        self.trace = trace or MemoryTraceResult(
            dialogue=MemoryTraceDialogueResult(
                id="dialogue-1",
                memory_id="memory-1",
                summary="用户喜欢周杰伦。",
            )
        )
        self.calls = []

    async def memory_trace(self, user_id, memory_id):
        self.calls.append(("memory_trace", user_id, memory_id))
        return self.trace


class ExplodingTraceGraphRepository(FakeTraceGraphRepository):
    async def memory_trace(self, user_id, memory_id):
        raise RuntimeError("neo4j unavailable")


class FakeProfileSummarizer:
    def __init__(self, core_facts=None, traits=None):
        self.core_facts = core_facts or []
        self.traits = traits or []
        self.calls = []

    async def summarize(self, entity, statements):
        self.calls.append((entity.id, statements))
        return self.core_facts, self.traits


class FlakyProfileSummarizer:
    def __init__(self):
        self.calls = 0

    async def summarize(self, entity, statements):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("profile failed")
        return ["可用事实"], ["可用特质"]


class FakeProfileLLM:
    def __init__(self, content):
        self.content = content

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        assert response_format == {"type": "json_object"}
        assert "core_facts" in messages[-1]["content"]
        return {"content": self.content}

    @property
    def model_name(self):
        return "fake-profile-llm"

    @property
    def temperature(self):
        return 0

    @property
    def max_tokens(self):
        return 0


class FakeJSONParser:
    async def invoke(self, text, default_value=None):
        import json

        return json.loads(text)


class MemoryUnitOfWork:
    def __init__(self, memory_repository):
        self.memory = memory_repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
