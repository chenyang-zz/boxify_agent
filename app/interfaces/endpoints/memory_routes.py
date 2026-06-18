from typing import Dict

from fastapi import APIRouter, Depends, Query

from app.application.services.memory_service import MemoryService
from app.interfaces.schemas.base import PageData, Response
from app.interfaces.schemas.memory import (
    MemoryClusterResponse,
    MemoryCommunityDetailResponse,
    MemoryCommunityMemberResponse,
    MemoryCommunityRelationResponse,
    MemoryCommunityResponse,
    MemoryConsolidateResponse,
    MemoryCreateRequest,
    MemoryDetailResponse,
    MemoryEntitySubgraphResponse,
    MemoryGraphViewResponse,
    MemoryInsightResponse,
    MemoryMergeDuplicatesResponse,
    MemoryProfileResponse,
    MemoryQualityIssueListResponse,
    MemoryQualityOverviewResponse,
    MemoryReflectResponse,
    MemoryRelationHistoryResponse,
    MemoryReextractRequest,
    MemoryReextractResponse,
    MemoryResponse,
    MemorySearchRequest,
    MemoryTimelineEventResponse,
)
from app.interfaces.service_dependencies import get_memory_service

router = APIRouter(prefix="/memories", tags=["记忆模块"])


@router.post(
    "",
    response_model=Response[MemoryResponse],
    summary="主动记住文本",
    description="把一段用户提供的文本保存为当前用户长期记忆。",
)
async def create_memory(
    body: MemoryCreateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """主动记住一段文本。"""
    memory = await memory_service.remember_text(body.content)
    return Response.success(msg="记忆已保存", data=MemoryResponse.from_domain(memory))


@router.get(
    "",
    response_model=Response[PageData[MemoryResponse]],
    summary="分页查询记忆",
    description="分页查询当前用户的长期记忆列表。",
)
async def list_memories(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """分页查询当前用户记忆。"""
    memories, total = await memory_service.list_memories(page, page_size)
    return Response.success(
        data=PageData[MemoryResponse].create(
            items=[MemoryResponse.from_domain(memory) for memory in memories],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.post(
    "/search",
    response_model=Response[list[MemoryResponse]],
    summary="检索记忆",
    description="在当前用户长期记忆中检索相关内容。",
)
async def search_memories(
    body: MemorySearchRequest,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """检索当前用户记忆。"""
    memories = await memory_service.search(body.query, body.top_k)
    return Response.success(
        data=[MemoryResponse.from_domain(memory) for memory in memories]
    )


@router.post(
    "/consolidate",
    response_model=Response[MemoryConsolidateResponse],
    summary="手动巩固记忆",
    description="对当前用户执行一次记忆动力学巩固和长期实体画像增强。",
)
async def consolidate_memories(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """手动巩固当前用户记忆。"""
    stats = await memory_service.consolidate()
    return Response.success(
        msg="记忆巩固完成",
        data=MemoryConsolidateResponse.model_validate(stats.model_dump()),
    )


@router.post(
    "/reflect",
    response_model=Response[MemoryReflectResponse],
    summary="手动反思记忆",
    description="对当前用户执行一次长期记忆反思并生成高层洞察。",
)
async def reflect_memories(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """手动反思当前用户记忆。"""
    stats = await memory_service.reflect()
    return Response.success(
        msg="记忆反思完成",
        data=MemoryReflectResponse.model_validate(stats.model_dump()),
    )


@router.post(
    "/cluster",
    response_model=Response[MemoryClusterResponse],
    summary="手动聚类记忆社区",
    description="对当前用户的记忆实体执行一次全量社区聚类。",
)
async def cluster_memories(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """手动聚类当前用户记忆社区。"""
    stats = await memory_service.cluster()
    return Response.success(
        msg="记忆社区聚类完成",
        data=MemoryClusterResponse.model_validate(stats.model_dump()),
    )


@router.get(
    "/communities",
    response_model=Response[list[MemoryCommunityResponse]],
    summary="查询记忆社区",
    description="查询当前用户的记忆实体社区列表。",
)
async def list_memory_communities(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户记忆社区列表。"""
    communities = await memory_service.list_communities()
    return Response.success(
        data=[
            MemoryCommunityResponse.model_validate(community.model_dump())
            for community in communities
        ]
    )


@router.get(
    "/communities/{community_id}",
    response_model=Response[MemoryCommunityDetailResponse],
    summary="查询记忆社区详情",
    description="查询当前用户指定记忆社区的成员实体和社区内关系。",
)
async def get_memory_community(
    community_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户记忆社区详情。"""
    members, relationships = await memory_service.community_detail(community_id)
    return Response.success(
        data=MemoryCommunityDetailResponse(
            members=[
                MemoryCommunityMemberResponse.model_validate(member.model_dump())
                for member in members
            ],
            relationships=[
                MemoryCommunityRelationResponse.model_validate(relation.model_dump())
                for relation in relationships
            ],
        )
    )


@router.get(
    "/graph",
    response_model=Response[MemoryGraphViewResponse],
    summary="查询记忆实体关系图",
    description="查询当前用户长期记忆图谱中的实体关系全图。",
)
async def get_memory_graph(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户记忆实体关系图。"""
    graph = await memory_service.graph()
    return Response.success(
        data=MemoryGraphViewResponse.model_validate(graph.model_dump())
    )


@router.get(
    "/graph/entity/{entity_id}",
    response_model=Response[MemoryEntitySubgraphResponse],
    summary="查询记忆实体一跳子图",
    description="查询当前用户指定实体及其一跳关系子图。",
)
async def get_memory_entity_subgraph(
    entity_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户指定实体一跳子图。"""
    subgraph = await memory_service.entity_subgraph(entity_id)
    return Response.success(
        data=MemoryEntitySubgraphResponse.model_validate(subgraph.model_dump())
    )


@router.get(
    "/profile",
    response_model=Response[MemoryProfileResponse],
    summary="查询记忆画像",
    description="查询当前用户长期记忆图谱中的实体画像分组。",
)
async def get_memory_profile(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户记忆画像。"""
    profile = await memory_service.profile()
    return Response.success(
        data=MemoryProfileResponse.model_validate(profile.model_dump())
    )


@router.get(
    "/insights",
    response_model=Response[list[MemoryInsightResponse]],
    summary="查询记忆洞察",
    description="查询当前用户长期记忆反思生成的高层洞察。",
)
async def list_memory_insights(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户记忆洞察。"""
    insights = await memory_service.list_insights()
    return Response.success(
        data=[
            MemoryInsightResponse.model_validate(insight.model_dump())
            for insight in insights
        ]
    )


@router.post(
    "/insights/{insight_id}/delete",
    response_model=Response[Dict],
    summary="删除记忆洞察",
    description="删除当前用户指定长期记忆洞察。",
)
async def delete_memory_insight(
    insight_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """删除当前用户指定记忆洞察。"""
    await memory_service.delete_insight(insight_id)
    return Response.success(msg="删除成功")


@router.post(
    "/entities/{entity_id}/delete",
    response_model=Response[Dict],
    summary="删除记忆实体",
    description="删除当前用户指定长期记忆实体及其图谱关系。",
)
async def delete_memory_entity(
    entity_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """删除当前用户指定记忆实体。"""
    await memory_service.delete_entity(entity_id)
    return Response.success(msg="删除成功")


@router.get(
    "/entities/{entity_id}/relations/history",
    response_model=Response[list[MemoryRelationHistoryResponse]],
    summary="查询记忆实体关系历史",
    description="查询当前用户指定实体的一跳关系历史事实。",
)
async def get_memory_entity_relation_history(
    entity_id: str,
    predicate: str | None = Query(default=None),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户指定实体一跳关系历史。"""
    relations = await memory_service.relation_history(entity_id, predicate=predicate)
    return Response.success(
        data=[
            MemoryRelationHistoryResponse.model_validate(relation.model_dump())
            for relation in relations
        ]
    )


@router.post(
    "/merge-duplicates",
    response_model=Response[MemoryMergeDuplicatesResponse],
    summary="合并重复记忆实体",
    description="合并当前用户历史同名同类型重复实体。",
)
async def merge_duplicate_memory_entities(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """合并当前用户历史重复实体。"""
    stats = await memory_service.merge_duplicates()
    return Response.success(
        msg="重复实体合并完成",
        data=MemoryMergeDuplicatesResponse.model_validate(stats.model_dump()),
    )


@router.get(
    "/quality",
    response_model=Response[MemoryQualityOverviewResponse],
    summary="查询记忆质量总览",
    description="查询当前用户 PG 记忆任务状态和图谱质量审计摘要。",
)
async def get_memory_quality(
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户记忆质量审计总览。"""
    quality = await memory_service.quality()
    return Response.success(
        data=MemoryQualityOverviewResponse.model_validate(quality.model_dump())
    )


@router.get(
    "/quality/issues",
    response_model=Response[MemoryQualityIssueListResponse],
    summary="查询记忆质量问题样本",
    description="按类别查询当前用户记忆质量问题样本。",
)
async def list_memory_quality_issues(
    category: str = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户指定类别记忆质量问题样本。"""
    issues = await memory_service.quality_issues(category, limit=limit)
    return Response.success(
        data=MemoryQualityIssueListResponse.model_validate(issues.model_dump())
    )


@router.post(
    "/reextract",
    response_model=Response[MemoryReextractResponse],
    summary="批量重新萃取记忆",
    description="按当前用户 PG 记忆选择集重新派发长期记忆图谱萃取任务。",
)
async def reextract_memories(
    body: MemoryReextractRequest,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """批量重新派发当前用户记忆图谱萃取任务。"""
    result = await memory_service.reextract_memories(
        memory_ids=body.memory_ids,
        statuses=body.statuses,
        only_missing_graph=body.only_missing_graph,
        dry_run=body.dry_run,
        limit=body.limit,
    )
    return Response.success(
        msg="记忆重萃取已处理",
        data=MemoryReextractResponse.model_validate(result.model_dump()),
    )


@router.get(
    "/timeline",
    response_model=Response[list[MemoryTimelineEventResponse]],
    summary="查询记忆事件时间线",
    description="查询当前用户从长期记忆图谱中萃取出的一次性经历事件。",
)
async def list_memory_timeline(
    limit: int = Query(default=50, ge=1, le=200),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户记忆事件时间线。"""
    events = await memory_service.timeline(limit)
    return Response.success(
        data=[
            MemoryTimelineEventResponse.model_validate(event.model_dump())
            for event in events
        ]
    )


@router.post(
    "/{memory_id}/reextract",
    response_model=Response[MemoryReextractResponse],
    summary="重新萃取单条记忆",
    description="重新派发当前用户指定长期记忆的图谱萃取任务。",
)
async def reextract_memory(
    memory_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """重新派发当前用户单条记忆图谱萃取任务。"""
    result = await memory_service.reextract_memory(memory_id)
    return Response.success(
        msg="记忆重萃取已派发",
        data=MemoryReextractResponse.model_validate(result.model_dump()),
    )


@router.get(
    "/{memory_id}",
    response_model=Response[MemoryDetailResponse],
    summary="查询记忆详情",
    description="查询当前用户指定长期记忆及其图谱溯源详情。",
)
async def get_memory_detail(
    memory_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """查询当前用户单条长期记忆详情。"""
    detail = await memory_service.detail(memory_id)
    return Response.success(data=MemoryDetailResponse.from_domain(detail))


@router.post(
    "/{memory_id}/delete",
    response_model=Response[Dict],
    summary="删除记忆",
    description="删除当前用户指定长期记忆。",
)
async def delete_memory(
    memory_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
):
    """删除当前用户记忆。"""
    await memory_service.delete_memory(memory_id)
    return Response.success(msg="删除成功")
