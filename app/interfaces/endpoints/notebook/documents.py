from typing import Dict

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.application.services.document_service import DocumentService
from app.interfaces.schemas.base import Response
from app.interfaces.schemas.notebook import (
    KnowledgeSearchHitResponse,
    KnowledgeSearchRequest,
    NotebookDocumentListResponse,
    NotebookDocumentResponse,
    NotebookUrlImportRequest,
)
from app.interfaces.service_dependencies import get_document_service

router = APIRouter(prefix="/documents", tags=["Notebook文档模块"])


@router.post(
    "/upload",
    response_model=Response[NotebookDocumentResponse],
    summary="上传Notebook文档",
    description="上传文档原文件并创建当前用户的Notebook文档记录，后台异步解析并写入知识库索引。",
)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
):
    """上传知识库文档并异步解析。"""
    content = await file.read()
    document = await document_service.upload_document(
        file.filename or "未命名", content
    )
    return Response.success(
        msg="上传成功，正在解析",
        data=await document_service.to_response(document),
    )


@router.post(
    "/from-url",
    response_model=Response[NotebookDocumentResponse],
    summary="从URL导入Notebook文档",
    description="抓取网页正文并保存为当前用户的Notebook文档，后台异步解析并写入知识库索引。",
)
async def import_from_url(
    body: NotebookUrlImportRequest,
    document_service: DocumentService = Depends(get_document_service),
):
    """从 URL 抓取正文并导入为知识库文档。"""
    document = await document_service.import_url(body.url)
    return Response.success(
        msg="导入成功，正在解析",
        data=await document_service.to_response(document),
    )


@router.get(
    "",
    response_model=Response[NotebookDocumentListResponse],
    summary="分页查询Notebook文档",
    description="分页查询当前用户的Notebook文档列表，可按标签名称过滤。",
)
async def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    tag: str | None = Query(default=None),
    document_service: DocumentService = Depends(get_document_service),
):
    """分页查询当前用户的知识库文档。"""
    documents, total = await document_service.list_documents(page, page_size, tag)
    items = [await document_service.to_response(document) for document in documents]
    return Response.success(
        data={
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }
    )


@router.get(
    "/{document_id}",
    response_model=Response[NotebookDocumentResponse],
    summary="查询Notebook文档详情",
    description="查询当前用户指定Notebook文档的基础信息、解析状态、chunk数量和标签列表。",
)
async def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """查询当前用户的单个知识库文档详情。"""
    document = await document_service.get_document(document_id)
    return Response.success(data=await document_service.to_response(document))


@router.get(
    "/{document_id}/status",
    response_model=Response[NotebookDocumentResponse],
    summary="查询Notebook文档解析状态",
    description="查询当前用户指定Notebook文档的解析状态和进度，响应结构与文档详情一致。",
)
async def get_document_status(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """查询文档解析状态，响应结构与详情接口保持一致。"""
    document = await document_service.get_document(document_id)
    return Response.success(data=await document_service.to_response(document))


@router.post(
    "/{document_id}/retry",
    response_model=Response[NotebookDocumentResponse],
    summary="重试Notebook文档解析",
    description="将当前用户指定Notebook文档重新置为待解析状态，并重新派发后台解析任务。",
)
async def retry_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """重新提交文档解析任务。"""
    document = await document_service.retry_document(document_id)
    return Response.success(
        msg="已重新提交解析",
        data=await document_service.to_response(document),
    )


@router.post(
    "/{document_id}/delete",
    response_model=Response[Dict],
    summary="删除Notebook文档",
    description="删除当前用户指定Notebook文档，同时清理原文件和知识库检索索引。",
)
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """删除当前用户的文档、原文件和检索索引。"""
    await document_service.delete_document(document_id)
    return Response.success(msg="删除成功")


@router.post(
    "/search",
    response_model=Response[list[KnowledgeSearchHitResponse]],
    summary="检索Notebook知识库",
    description="在当前用户Notebook知识库中执行混合检索，支持top_k和标签过滤。",
)
async def search_documents(
    body: KnowledgeSearchRequest,
    document_service: DocumentService = Depends(get_document_service),
):
    """在当前用户知识库中执行混合检索。"""
    hits = await document_service.search_documents(body.query, body.top_k, body.tags)
    return Response.success(
        data=[
            KnowledgeSearchHitResponse.model_validate(hit.model_dump()).model_dump()
            for hit in hits
        ]
    )
