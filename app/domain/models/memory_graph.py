from datetime import datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, Field

MEMORY_QUALITY_ISSUE_CATEGORIES = frozenset(
    {
        "duplicate_entities",
        "missing_embeddings",
        "orphan_entities",
        "orphan_statements",
        "broken_relations",
        "expired_relations",
        "empty_communities",
        "orphan_insights",
    }
)


def stable_memory_graph_id(*parts: str) -> str:
    """生成可重复的图谱 ID，保证重试写入 Neo4j 时 MERGE 幂等。"""
    return str(uuid5(NAMESPACE_URL, "::".join(parts)))


class DialogueNode(BaseModel):
    """一条 PG 记忆在图谱中的对话根节点。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    memory_id: str
    summary: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class ChunkNode(BaseModel):
    """对话文本分块节点。"""

    id: str
    user_id: str
    dialogue_id: str
    index: int
    text: str


class StatementNode(BaseModel):
    """原子陈述节点。"""

    id: str
    user_id: str
    chunk_id: str
    index: int
    text: str
    statement_type: str = "FACT"
    temporal_type: str = "STATIC"
    importance: float = 0.5
    confidence: float = 0.8
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    access_count: int = 0
    last_access_at: datetime | None = None
    memory_layer: str = "short_term"


class EntityNode(BaseModel):
    """实体节点。"""

    id: str
    user_id: str
    name: str
    type: str
    description: str = ""
    embedding: list[float] = Field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.8
    mention_count: int = 1
    access_count: int = 0
    last_access_at: datetime | None = None
    memory_layer: str = "short_term"
    core_facts: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)


class EventNode(BaseModel):
    """一次性发生的用户经历事件。"""

    id: str
    user_id: str
    title: str
    description: str = ""
    event_time: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class CommunityNode(BaseModel):
    """实体社区节点。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    name: str
    summary: str = ""
    member_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = None


class CommunityVoteEntity(BaseModel):
    """社区聚类投票所需的轻量实体数据。"""

    id: str
    user_id: str
    name: str
    type: str
    description: str = ""
    embedding: list[float] = Field(default_factory=list)
    community_id: str | None = None


class CommunityVoteNeighbor(BaseModel):
    """社区聚类投票所需的一跳邻居快照。"""

    id: str
    community_id: str | None = None
    embedding: list[float] = Field(default_factory=list)


class RelationEdge(BaseModel):
    """实体间语义关系。"""

    id: str
    user_id: str
    source_entity_id: str
    target_entity_id: str
    statement_id: str
    name: str
    evidence: str
    importance: float = 0.5
    confidence: float = 0.8
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    access_count: int = 0
    last_access_at: datetime | None = None
    memory_layer: str = "short_term"


class MentionEdge(BaseModel):
    """陈述提及实体的溯源边。"""

    id: str
    user_id: str
    statement_id: str
    entity_id: str


class InvolvesEdge(BaseModel):
    """事件涉及实体的语义边。"""

    id: str
    user_id: str
    event_id: str
    entity_id: str
    role: str = ""


class MemoryGraph(BaseModel):
    """单条长期记忆萃取出的四层溯源图谱。"""

    dialogue: DialogueNode
    chunks: list[ChunkNode] = Field(default_factory=list)
    statements: list[StatementNode] = Field(default_factory=list)
    entities: list[EntityNode] = Field(default_factory=list)
    mentions: list[MentionEdge] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    events: list[EventNode] = Field(default_factory=list)
    involves: list[InvolvesEdge] = Field(default_factory=list)

    def stats(self) -> "MemoryGraphStats":
        """返回任务状态可直接持久化的图谱统计。"""
        return MemoryGraphStats(
            dialogue_id=self.dialogue.id,
            chunks=len(self.chunks),
            statements=len(self.statements),
            entities=len(self.entities),
            relations=len(self.relations),
            events=len(self.events),
            involves=len(self.involves),
        )


class MemoryGraphStats(BaseModel):
    """长期记忆图谱萃取后的结构化统计。"""

    dialogue_id: str = ""
    chunks: int = 0
    statements: int = 0
    entities: int = 0
    relations: int = 0
    events: int = 0
    involves: int = 0


class MemoryPromotionStats(BaseModel):
    """短期图谱记忆提升为长期记忆后的统计。"""

    promoted_entities: int = 0
    promoted_statements: int = 0


class MemoryConsolidationStats(BaseModel):
    """一次长期记忆巩固任务的完整统计。"""

    promoted_entities: int = 0
    promoted_statements: int = 0
    enhanced_profiles: int = 0


class MemoryReflectStats(BaseModel):
    """一次长期记忆反思任务的统计。"""

    insights: int = 0
    skipped: str | None = None
    error: str | None = None


class MemoryCommunityClusterStats(BaseModel):
    """一次社区聚类任务的统计。"""

    communities: int = 0
    assigned_entities: int = 0
    merged_communities: int = 0
    enhanced_communities: int = 0
    skipped: str | None = None
    error: str | None = None


class MemoryMergeDuplicatesResult(BaseModel):
    """历史重复实体合并统计。"""

    removed_entities: int = 0
    merged_groups: int = 0


class MemoryQualityGraphCountsResult(BaseModel):
    """当前用户记忆图谱节点和关系数量统计。"""

    dialogues: int = 0
    chunks: int = 0
    statements: int = 0
    entities: int = 0
    relations: int = 0
    events: int = 0
    involves: int = 0
    communities: int = 0
    insights: int = 0


class MemoryQualityIssueSummaryResult(BaseModel):
    """当前用户记忆质量问题摘要统计。"""

    duplicate_entities: int = 0
    missing_embeddings: int = 0
    orphan_entities: int = 0
    orphan_statements: int = 0
    broken_relations: int = 0
    expired_relations: int = 0
    empty_communities: int = 0
    orphan_insights: int = 0


class MemoryQualityFailedMemoryResult(BaseModel):
    """最近失败 PG 记忆摘要。"""

    id: str
    content: str = ""
    error_msg: str | None = None
    updated_at: datetime | None = None


class MemoryQualityIssueResult(BaseModel):
    """单条记忆质量问题样本。"""

    category: str
    severity: str = "info"
    title: str
    detail: str = ""
    entity_ids: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryQualityIssueListResult(BaseModel):
    """指定类别质量问题样本列表。"""

    category: str
    total: int = 0
    items: list[MemoryQualityIssueResult] = Field(default_factory=list)


class MemoryQualityOverviewResult(BaseModel):
    """当前用户记忆质量审计总览。"""

    generated_at: datetime = Field(default_factory=datetime.now)
    pg_total: int = 0
    pg_status_counts: dict[str, int] = Field(default_factory=dict)
    recent_failed: list[MemoryQualityFailedMemoryResult] = Field(default_factory=list)
    graph_available: bool = True
    graph_counts: MemoryQualityGraphCountsResult = Field(
        default_factory=MemoryQualityGraphCountsResult
    )
    issue_summary: MemoryQualityIssueSummaryResult = Field(
        default_factory=MemoryQualityIssueSummaryResult
    )


class MemoryEntityDedupDecision(BaseModel):
    """LLM 对两个候选实体是否为同一实体的保守判定。"""

    same_entity: bool = False
    canonical_idx: int = 0
    confidence: float = 0
    reason: str = ""


class MemoryEntityDedupCandidate(BaseModel):
    """进入 LLM 精判前的实体候选对。"""

    left: EntityNode
    right: EntityNode
    name_similarity: float = 0
    embedding_similarity: float = 0
    name_contains: bool = False


class MemoryEntityDedupResult(BaseModel):
    """实体消歧后的实体映射和 ID 重定向信息。"""

    entities: list[EntityNode] = Field(default_factory=list)
    entity_by_idx: dict[int, EntityNode] = Field(default_factory=dict)
    redirects: dict[str, str] = Field(default_factory=dict)


class InsightNode(BaseModel):
    """反思产生的高层洞察节点。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    theme: str
    content: str
    embedding: list[float] = Field(default_factory=list)
    importance: float = 0.6
    confidence: float = 0.7
    source_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = None


class InsightResult(BaseModel):
    """主动召回返回的洞察命中。"""

    id: str
    theme: str
    content: str
    importance: float = 0.6
    confidence: float = 0.7
    source_count: int = 0
    score: float = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CommunityResult(BaseModel):
    """社区列表返回项。"""

    id: str
    name: str
    summary: str = ""
    member_count: int = 0


class CommunityMemberResult(BaseModel):
    """社区成员实体返回项。"""

    entity_id: str
    entity_name: str
    entity_type: str
    description: str = ""
    community_id: str
    embedding: list[float] = Field(default_factory=list)
    importance: float = 0.5
    mention_count: int = 0
    access_count: int = 0


class CommunityRelationResult(BaseModel):
    """社区内部关系事实返回项。"""

    source_entity_id: str
    source_name: str
    target_entity_id: str
    target_name: str
    name: str
    evidence: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryTimelineParticipantResult(BaseModel):
    """时间线事件参与实体。"""

    entity_id: str
    name: str
    type: str


class MemoryTimelineEventResult(BaseModel):
    """用户记忆事件时间线项。"""

    id: str
    title: str
    description: str = ""
    event_time: datetime | None = None
    created_at: datetime | None = None
    participants: list[MemoryTimelineParticipantResult] = Field(default_factory=list)


class MemoryGraphNodeResult(BaseModel):
    """记忆图谱可视化实体节点。"""

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


class MemoryGraphEdgeResult(BaseModel):
    """记忆图谱可视化实体关系边。"""

    source: str
    target: str
    predicate: str
    evidence: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryGraphViewResult(BaseModel):
    """当前用户完整实体关系图。"""

    nodes: list[MemoryGraphNodeResult] = Field(default_factory=list)
    edges: list[MemoryGraphEdgeResult] = Field(default_factory=list)
    communities: list[CommunityResult] = Field(default_factory=list)


class MemoryEntitySubgraphResult(BaseModel):
    """单实体一跳子图。"""

    center: str
    nodes: list[MemoryGraphNodeResult] = Field(default_factory=list)
    edges: list[MemoryGraphEdgeResult] = Field(default_factory=list)


class MemoryProfileRelationResult(BaseModel):
    """用户画像实体的一跳出边关系事实。"""

    predicate: str
    target_entity_id: str | None = None
    target_name: str | None = None
    target_type: str | None = None
    evidence: str = ""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryRelationHistoryResult(BaseModel):
    """单个实体的一跳关系历史事实。"""

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


class MemoryProfileEntityResult(BaseModel):
    """用户画像实体。"""

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
    relations: list[MemoryProfileRelationResult] = Field(default_factory=list)


class MemoryProfileGroupResult(BaseModel):
    """按类型聚合的一组画像实体。"""

    type: str
    entities: list[MemoryProfileEntityResult] = Field(default_factory=list)


class MemoryProfileResult(BaseModel):
    """当前用户记忆画像视图。"""

    total: int = 0
    type_counts: dict[str, int] = Field(default_factory=dict)
    groups: list[MemoryProfileGroupResult] = Field(default_factory=list)


class GraphRelationFact(BaseModel):
    """检索返回的一跳关系事实。"""

    name: str
    direction: str
    neighbor_name: str
    neighbor_type: str
    evidence: str
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    is_current: bool = True


class MemoryGraphResult(BaseModel):
    """图谱检索返回给长期记忆管理器的结果。"""

    entity_id: str
    entity_name: str
    entity_type: str
    description: str = ""
    score: float = 0
    importance: float = 0.5
    memory_layer: str = "short_term"
    core_facts: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
    access_count: int = 0
    mention_count: int = 0
    source_memory_id: str | None = None
    source_memory_summary: str | None = None
    relations: list[GraphRelationFact] = Field(default_factory=list)


class LongTermMemoryGraphData(BaseModel):
    """长期记忆检索命中时附带的图谱上下文。"""

    entity_id: str
    entity_name: str
    entity_type: str
    description: str = ""
    score: float = 0
    importance: float = 0.5
    memory_layer: str = "short_term"
    core_facts: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
    access_count: int = 0
    mention_count: int = 0
    source_memory_id: str | None = None
    source_memory_summary: str | None = None
    relations: list[GraphRelationFact] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: MemoryGraphResult) -> "LongTermMemoryGraphData":
        """从图谱仓储检索结果构造长期记忆附加数据。"""
        return cls(
            entity_id=result.entity_id,
            entity_name=result.entity_name,
            entity_type=result.entity_type,
            description=result.description,
            score=result.score,
            importance=result.importance,
            memory_layer=result.memory_layer,
            core_facts=result.core_facts,
            traits=result.traits,
            access_count=result.access_count,
            mention_count=result.mention_count,
            source_memory_id=result.source_memory_id,
            source_memory_summary=result.source_memory_summary,
            relations=result.relations,
        )
