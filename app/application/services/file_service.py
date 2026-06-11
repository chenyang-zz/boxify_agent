from typing import BinaryIO, Callable, Tuple

from fastapi import UploadFile

from app.domain.external.file_storage import FileStorage
from app.domain.models.file import File
from app.domain.repositories.vow import IUnitOfWork


class FileService:
    """Boxify文件系统服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        file_storage: FileStorage,
    ):
        """构造函数，完成文件服务的初始化"""
        self._uow = uow_factory()
        self.file_storage = file_storage

    async def upload_file(self, upload_file: UploadFile) -> File:
        """将文件上传到腾讯云cos并记录上传数据"""
        return await self.file_storage.upload_file(upload_file=upload_file)

    async def get_file_info(self, file_id: str) -> File:
        """根据传递的文件id获取文件信息"""
        async with self._uow:
            file = await self._uow.file.get_by_id(file_id)
        if not file:
            raise ValueError(f"该文件[{file_id}]不存在")

        return file

    async def download_file(self, file_id: str) -> Tuple[BinaryIO, File]:
        """根据传递的文件id下载文件"""
        return await self.file_storage.download_file(file_id)
