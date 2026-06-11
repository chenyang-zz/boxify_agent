import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.application.errors.exceptions import AppException
from app.interfaces.schemas import Response

logger = logging.getLogger(__name__)


def _sanitize_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, str]]:
    """提取可展示的校验错误字段，避免返回或记录用户输入值"""
    sanitized_errors: list[dict[str, str]] = []
    for error in errors:
        loc = error.get("loc", ())
        field = ".".join(str(item) for item in loc)
        sanitized_errors.append(
            {
                "field": field,
                "message": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
        )
    return sanitized_errors


def register_exception_handlers(app: FastAPI) -> None:
    """处理项目中所有的异常并进行统一处理，涵盖：自定义业务状态异常、HTTP异常、通用异常"""

    @app.exception_handler(AppException)
    async def app_exception_handler(req: Request, e: AppException) -> JSONResponse:
        """处理业务异常，将所有状态统一响应结构"""
        logger.error(f"AppException: {e.msg}")
        return JSONResponse(
            status_code=e.status_code,
            content=Response(
                code=e.status_code,
                msg=e.msg,
                data={},
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(req: Request, e: HTTPException) -> JSONResponse:
        """处理FastAPI抛出的http异常，将所有状态统一响应结构"""
        logger.error(f"HTTPException: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content=Response(
                code=e.status_code,
                msg=e.detail,
                data={},
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        req: Request, e: RequestValidationError
    ) -> JSONResponse:
        """处理请求参数校验异常，将所有状态统一响应结构"""
        errors = _sanitize_validation_errors(e.errors())
        logger.error(f"RequestValidationError: {errors}")
        return JSONResponse(
            status_code=422,
            content=Response(
                code=422,
                msg="请求参数数据校验错误，请核实后重试",
                data={"errors": errors},
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def exception_handler(req: Request, e: Exception) -> JSONResponse:
        """处理抛出的为定义的任意异常，将状态码统一设置为500"""
        logger.error(f"Exception: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=Response(
                code=500,
                msg="服务器出现异常请稍后重试",
                data={},
            ).model_dump(),
        )
