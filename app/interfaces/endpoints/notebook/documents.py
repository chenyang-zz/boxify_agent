from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.application.services.document_service import DocumentService
from app.interfaces.schemas.base import Response
from app.interfaces.schemas.notebook import (
    KnowledgeSearchRequest,
    NotebookDocumentListResponse,
    NotebookDocumentResponse,
    NotebookUrlImportRequest,
)
from app.interfaces.service_dependencies import get_document_service

router = APIRouter(prefix="/documents", tags=["Notebook文档模块"])


@router.post("/upload", response_model=Response[NotebookDocumentResponse])
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


@router.post("/from-url", response_model=Response[NotebookDocumentResponse])
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


@router.get("", response_model=Response[NotebookDocumentListResponse])
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


@router.get("/{document_id}", response_model=Response[NotebookDocumentResponse])
async def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """查询当前用户的单个知识库文档详情。"""
    document = await document_service.get_document(document_id)
    return Response.success(data=await document_service.to_response(document))


@router.get("/{document_id}/status", response_model=Response[NotebookDocumentResponse])
async def get_document_status(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """查询文档解析状态，响应结构与详情接口保持一致。"""
    document = await document_service.get_document(document_id)
    return Response.success(data=await document_service.to_response(document))


@router.post("/{document_id}/retry", response_model=Response[NotebookDocumentResponse])
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


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """删除当前用户的文档、原文件和检索索引。"""
    await document_service.delete_document(document_id)
    return Response.success(msg="删除成功")


@router.post("/search")
async def search_documents(
    body: KnowledgeSearchRequest,
    document_service: DocumentService = Depends(get_document_service),
):
    """在当前用户知识库中执行混合检索。"""
    hits = await document_service.search_documents(body.query, body.top_k, body.tags)
    return Response.success(data=[hit.model_dump() for hit in hits])
