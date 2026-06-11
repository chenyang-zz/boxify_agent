import logging
import urllib.parse

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.responses import StreamingResponse

from app.application.services.file_service import FileService
from app.domain.models.file import File as FileInfo
from app.interfaces.schemas import Response
from app.interfaces.service_dependencies import get_file_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["文件模块"])


@router.post(
    path="",
    summary="对话文件上传接口",
    description="在对话接口中，将文件上传到cos对象存储和沙箱中",
    response_model=Response[FileInfo],
)
async def upload_file(
    file: UploadFile = File(...),
    file_service: FileService = Depends(get_file_service),
):
    fileinfo = await file_service.upload_file(upload_file=file)
    return Response.success(msg="上传成功", data=fileinfo)


@router.post(
    path="/{file_id}",
    summary="获取文件信息接口",
    description="获取指定会话中对应文件的基础信息",
    response_model=Response[FileInfo],
)
async def get_file_info(
    file_id: str,
    file_service: FileService = Depends(get_file_service),
):
    """获取指定会话中对应文件的基础信息"""
    fileinfo = await file_service.get_file_info(file_id)
    return Response.success(
        msg="获取文件信息成功",
        data=fileinfo,
    )


@router.get(
    path="/{file_id}/download",
    summary="文件下载接口",
    description="从沙箱或对象存储中下载指定的文件到本地",
)
async def download_file(
    file_id: str,
    file_service: FileService = Depends(get_file_service),
):
    """下载指定会话中的指定文件"""
    # 调用服务获取文件源数据
    file_data, fileinfo = await file_service.download_file(file_id)

    # 对文件中的中文名字进行url编码
    encoded_filename = urllib.parse.quote(fileinfo.filename)

    # 返回文件流数据
    return StreamingResponse(
        content=file_data,
        media_type=fileinfo.mime_type,
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{encoded_filename}",
            "Content-Length": str(fileinfo.size),
        },
    )
