import asyncio

from app.bootstrap.notebook import (
    build_document_storage,
    build_embedding_model,
    build_knowledge_search,
    ensure_knowledge_index,
)
from app.celery_app import celery_app
from app.domain.services.notebook.chunker import DocumentChunker
from app.domain.services.notebook.parser import DocumentParser
from app.infrastructure.storage.cos import get_cos
from app.infrastructure.storage.elasticsearch import get_elasticsearch
from app.infrastructure.storage.postgres import get_postgres, get_uow


async def _run(document_id: str) -> None:
    """执行单个文档解析任务，按阶段使用短事务更新文档状态。"""
    await get_postgres().init()
    await get_cos().init()
    await get_elasticsearch().init()
    storage = build_document_storage()
    await ensure_knowledge_index()

    try:
        # 状态切换单独提交，避免后续解析或外部 IO 失败影响已开始标记。
        async with get_uow() as uow:
            document = await uow.document.get_by_id(document_id)
            if not document:
                return
            document.mark_parsing()
            await uow.document.save(document)

        embedding_client = await build_embedding_model(document.user_id)
        knowledge_search = await build_knowledge_search(document.user_id)
        content = await storage.get(document.file_key)
        text = DocumentParser.parse(document.file_ext, content)
        if not text.strip():
            raise ValueError("解析结果为空")
        parents = DocumentChunker.chunk_parent_child(text)
        if not parents:
            raise ValueError("分块结果为空")

        # 父块负责回显完整上下文，子块携带向量用于实际召回。
        chunks = []
        chunk_total = 0
        for parent in parents:
            parent_chunk = DocumentChunker.build_chunk(
                user_id=document.user_id,
                source_id=document.id,
                doc_name=document.file_name,
                chunk_type=DocumentChunker.CHUNK_TYPE_PARENT,
                content=parent.content,
            )
            chunks.append(parent_chunk)
            if parent.children:
                vectors = await embedding_client.embed(parent.children)
                for child, vector in zip(parent.children, vectors):
                    chunks.append(
                        DocumentChunker.build_chunk(
                            user_id=document.user_id,
                            source_id=document.id,
                            doc_name=document.file_name,
                            chunk_type=DocumentChunker.CHUNK_TYPE_CHILD,
                            content=child,
                            vector=vector,
                            parent_id=parent_chunk.id,
                        )
                    )
                    chunk_total += 1
        # 重试解析时先清理旧 chunk，再批量写入新 chunk，保证索引幂等。
        await knowledge_search.delete_by_source(document.user_id, document.id)
        await knowledge_search.save_chunks(chunks)
        async with get_uow() as uow:
            document = await uow.document.get_by_id(document_id)
            if not document:
                return
            document.mark_done(chunk_total)
            await uow.document.save(document)
    except Exception as e:
        async with get_uow() as uow:
            document = await uow.document.get_by_id(document_id)
            if document:
                document.mark_failed(str(e))
                await uow.document.save(document)
    finally:
        # Celery worker 独立进程持有自己的连接，任务结束后主动释放。
        await get_elasticsearch().shutdown()
        await get_postgres().shutdown()
        await get_cos().shutdown()


@celery_app.task(name="app.tasks.notebook_parse_document")
def parse_document_task(document_id: str) -> str:
    """Celery 任务入口，任务名保持稳定供派发器引用。"""
    asyncio.run(_run(document_id))
    return document_id
