from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.long_term_memory import (
    LongTermMemory,
    MemorySource,
    MemoryStatus,
)


class MemoryCreateRequest(BaseModel):
    """主动记住请求。"""

    content: str = Field(min_length=1, max_length=10000)


class MemorySearchRequest(BaseModel):
    """记忆检索请求。"""

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class MemoryConsolidateResponse(BaseModel):
    """手动记忆巩固统计响应。"""

    promoted_entities: int = 0
    promoted_statements: int = 0
    enhanced_profiles: int = 0


class MemoryReflectResponse(BaseModel):
    """手动记忆反思统计响应。"""

    insights: int = 0
    skipped: str | None = None
    error: str | None = None


class MemoryClusterResponse(BaseModel):
    """手动社区聚类统计响应。"""

    communities: int = 0
    assigned_entities: int = 0
    merged_communities: int = 0
    enhanced_communities: int = 0
    skipped: str | None = None
    error: str | None = None


class MemoryMergeDuplicatesResponse(BaseModel):
    """历史重复实体合并统计响应。"""

    removed_entities: int = 0
    merged_groups: int = 0


class MemoryCommunityResponse(BaseModel):
    """记忆社区列表响应。"""

    id: str
    name: str
    summary: str = ""
    member_count: int = 0


class MemoryCommunityMemberResponse(BaseModel):
    """记忆社区成员响应。"""

    entity_id: str
    entity_name: str
    entity_type: str
    description: str = ""
    community_id: str
    importance: float = 0.5
    mention_count: int = 0
    access_count: int = 0


class MemoryCommunityRelationResponse(BaseModel):
    """记忆社区内部关系响应。"""

    source_entity_id: str
    source_name: str
    target_entity_id: str
    target_name: str
    name: str
    evidence: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryCommunityDetailResponse(BaseModel):
    """记忆社区详情响应。"""

    members: list[MemoryCommunityMemberResponse]
    relationships: list[MemoryCommunityRelationResponse]


class MemoryTimelineParticipantResponse(BaseModel):
    """记忆事件参与实体响应。"""

    entity_id: str
    name: str
    type: str


class MemoryTimelineEventResponse(BaseModel):
    """记忆事件时间线响应。"""

    id: str
    title: str
    description: str = ""
    event_time: datetime | None = None
    created_at: datetime | None = None
    participants: list[MemoryTimelineParticipantResponse]


class MemoryGraphNodeResponse(BaseModel):
    """记忆图谱可视化实体节点响应。"""

    id: str
    name: str
    type: str
    description: str = ""
    community_id: str | None = None
    importance: float = 0.5
    memory_layer: str = "short_term"
    access_count: int = 0
    mention_count: int = 0
    core_facts: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)


class MemoryGraphEdgeResponse(BaseModel):
    """记忆图谱可视化关系边响应。"""

    source: str
    target: str
    predicate: str
    evidence: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryGraphViewResponse(BaseModel):
    """记忆图谱可视化全图响应。"""

    nodes: list[MemoryGraphNodeResponse]
    edges: list[MemoryGraphEdgeResponse]
    communities: list[MemoryCommunityResponse]


class MemoryEntitySubgraphResponse(BaseModel):
    """记忆图谱可视化单实体一跳子图响应。"""

    center: str
    nodes: list[MemoryGraphNodeResponse]
    edges: list[MemoryGraphEdgeResponse]


class MemoryProfileRelationResponse(BaseModel):
    """记忆画像实体关系响应。"""

    predicate: str
    target_entity_id: str | None = None
    target_name: str | None = None
    target_type: str | None = None
    evidence: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryRelationHistoryResponse(BaseModel):
    """记忆实体关系历史响应。"""

    relation_id: str
    direction: str
    neighbor_entity_id: str
    neighbor_name: str
    neighbor_type: str
    predicate: str
    evidence: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryProfileEntityResponse(BaseModel):
    """记忆画像实体响应。"""

    id: str
    name: str
    type: str
    description: str = ""
    community_id: str | None = None
    importance: float = 0.5
    memory_layer: str = "short_term"
    access_count: int = 0
    mention_count: int = 0
    core_facts: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
    relations: list[MemoryProfileRelationResponse] = Field(default_factory=list)


class MemoryProfileGroupResponse(BaseModel):
    """记忆画像实体分组响应。"""

    type: str
    entities: list[MemoryProfileEntityResponse]


class MemoryProfileResponse(BaseModel):
    """记忆画像响应。"""

    total: int = 0
    type_counts: dict[str, int] = Field(default_factory=dict)
    groups: list[MemoryProfileGroupResponse] = Field(default_factory=list)


class MemoryQualityGraphCountsResponse(BaseModel):
    """记忆质量审计图谱数量响应。"""

    dialogues: int = 0
    chunks: int = 0
    statements: int = 0
    entities: int = 0
    relations: int = 0
    events: int = 0
    involves: int = 0
    communities: int = 0
    insights: int = 0


class MemoryQualityIssueSummaryResponse(BaseModel):
    """记忆质量审计问题摘要响应。"""

    duplicate_entities: int = 0
    missing_embeddings: int = 0
    orphan_entities: int = 0
    orphan_statements: int = 0
    broken_relations: int = 0
    expired_relations: int = 0
    empty_communities: int = 0
    orphan_insights: int = 0


class MemoryQualityFailedMemoryResponse(BaseModel):
    """记忆质量审计最近失败 PG 记忆响应。"""

    id: str
    content: str = ""
    error_msg: str | None = None
    updated_at: datetime | None = None


class MemoryQualityIssueResponse(BaseModel):
    """记忆质量审计问题样本响应。"""

    category: str
    severity: str = "info"
    title: str
    detail: str = ""
    entity_ids: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class MemoryQualityIssueListResponse(BaseModel):
    """记忆质量审计问题列表响应。"""

    category: str
    total: int = 0
    items: list[MemoryQualityIssueResponse] = Field(default_factory=list)


class MemoryQualityOverviewResponse(BaseModel):
    """记忆质量审计总览响应。"""

    generated_at: datetime
    pg_total: int = 0
    pg_status_counts: dict[str, int] = Field(default_factory=dict)
    recent_failed: list[MemoryQualityFailedMemoryResponse] = Field(default_factory=list)
    graph_available: bool = True
    graph_counts: MemoryQualityGraphCountsResponse = Field(
        default_factory=MemoryQualityGraphCountsResponse
    )
    issue_summary: MemoryQualityIssueSummaryResponse = Field(
        default_factory=MemoryQualityIssueSummaryResponse
    )


class MemoryInsightResponse(BaseModel):
    """记忆洞察响应。"""

    id: str
    theme: str
    content: str
    importance: float = 0.6
    confidence: float = 0.7
    source_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryGraphStatsResponse(BaseModel):
    """长期记忆图谱萃取统计响应。"""

    dialogue_id: str = ""
    chunks: int = 0
    statements: int = 0
    entities: int = 0
    relations: int = 0
    events: int = 0
    involves: int = 0


class MemoryResponse(BaseModel):
    """记忆响应。"""

    id: str
    content: str
    source: MemorySource
    source_session_id: str | None
    status: MemoryStatus
    summary: str | None
    keywords: list[str]
    graph_dialogue_id: str | None
    graph_stats: MemoryGraphStatsResponse
    error_msg: str | None
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_domain(cls, memory: LongTermMemory) -> "MemoryResponse":
        """从领域模型构造响应。"""
        return cls.model_validate(memory.model_dump(mode="python"))
