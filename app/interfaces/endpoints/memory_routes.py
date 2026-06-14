from typing import Dict

from fastapi import APIRouter, Depends, Query

from app.application.services.memory_service import MemoryService
from app.interfaces.schemas.base import PageData, Response
from app.interfaces.schemas.memory import (
    MemoryCreateRequest,
    MemoryResponse,
    MemorySearchRequest,
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
