from app.domain.models.tag import Tag


class TagService:
    """知识库标签应用服务，所有查询都限定在当前登录用户上下文内。"""

    def __init__(self, uow_factory, user_id: str) -> None:
        self._uow_factory = uow_factory
        self._user_id = user_id

    async def list_tags(self) -> list[Tag]:
        """返回当前用户创建过的标签列表。"""
        async with self._uow_factory() as uow:
            return await uow.tag.list_by_user(self._user_id)

    async def get_document_tags(self, document_id: str) -> list[str]:
        """返回指定文档关联的标签名，用于文档响应组装。"""
        async with self._uow_factory() as uow:
            return await uow.tag.get_document_tags(document_id)
