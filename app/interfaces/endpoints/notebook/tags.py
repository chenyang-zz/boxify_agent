from fastapi import APIRouter, Depends

from app.application.services.tag_service import TagService
from app.interfaces.schemas.base import Response
from app.interfaces.schemas.notebook import NotebookTagResponse
from app.interfaces.service_dependencies import get_tag_service

router = APIRouter(prefix="/tags", tags=["Notebook标签模块"])


@router.get(
    "",
    response_model=Response[list[NotebookTagResponse]],
    summary="查询Notebook标签列表",
    description="查询当前用户Notebook文档关联过的标签列表。",
)
async def list_tags(
    tag_service: TagService = Depends(get_tag_service),
):
    """查询当前用户的知识库标签列表。"""
    tags = await tag_service.list_tags()
    return Response.success(
        data=[NotebookTagResponse(id=tag.id, name=tag.name).model_dump() for tag in tags]
    )
