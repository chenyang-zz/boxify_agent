import logging

from fastapi import APIRouter, Depends

from app.application.services.auth_service import AuthService
from app.domain.models.user import User
from app.interfaces.schemas.auth import LoginRequest, LoginResponse, UserResponse
from app.interfaces.schemas.base import Response
from app.interfaces.service_dependencies import (
    get_auth_service,
    get_current_user,
    require_bearer_token,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["认证模块"])


def to_user_response(user: User) -> UserResponse:
    """转换用户响应，避免输出密码摘要"""
    return UserResponse(
        id=user.id,
        username=user.username,
        is_active=user.is_active,
        is_admin=user.is_admin,
    )


@router.post(
    path="/login",
    response_model=Response[LoginResponse],
    summary="用户登录",
    description="使用用户名和密码登录，返回Bearer access token",
)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """用户登录"""
    user = await auth_service.authenticate(request.username, request.password)
    access_token = auth_service.create_access_token(user)
    return Response.success(
        msg="登录成功",
        data=LoginResponse(
            access_token=access_token,
            expires_in=auth_service.expires_in,
            user=to_user_response(user),
        ),
    )


@router.get(
    path="/me",
    response_model=Response[UserResponse],
    summary="获取当前登录用户",
    description="根据Bearer access token获取当前用户基础信息",
    dependencies=[Depends(require_bearer_token)],
)
async def me(current_user: User = Depends(get_current_user)):
    """获取当前登录用户"""
    return Response.success(data=to_user_response(current_user))
