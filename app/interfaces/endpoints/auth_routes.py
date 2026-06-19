import logging
from urllib.parse import urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from app.application.services.auth_service import AuthService
from app.domain.models.user import User
from app.interfaces.schemas.auth import (
    LoginRequest,
    LoginResponse,
    OAuthAuthorizeResponse,
    UserResponse,
)
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
        email=user.email,
        avatar_url=user.avatar_url,
        oauth_provider=user.oauth_provider,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def build_oauth_redirect_url(
    frontend_redirect_uri: str,
    login_response: LoginResponse,
) -> str:
    """构建OAuth前端回跳地址，把登录结果放入URL fragment"""
    user = login_response.user
    fragment = urlencode(
        {
            "access_token": login_response.access_token,
            "token_type": login_response.token_type,
            "expires_in": str(login_response.expires_in),
            "user_id": user.id,
            "username": user.username,
            "is_active": str(user.is_active).lower(),
            "is_admin": str(user.is_admin).lower(),
        }
    )
    parsed = urlsplit(frontend_redirect_uri)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.query,
            fragment,
        )
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
    path="/oauth/{provider}/authorize",
    response_model=Response[OAuthAuthorizeResponse],
    summary="创建OAuth授权地址",
    description="为GitHub或Google登录创建第三方授权地址",
)
async def oauth_authorize(
    provider: str,
    auth_service: AuthService = Depends(get_auth_service),
):
    """创建OAuth授权地址"""
    authorization = auth_service.create_oauth_authorization(provider)
    return Response.success(
        data=OAuthAuthorizeResponse(
            authorization_url=authorization.authorization_url,
        )
    )


@router.get(
    path="/oauth/{provider}/callback",
    response_model=Response[LoginResponse],
    summary="OAuth登录回调",
    description="处理GitHub或Google授权回调，返回Bearer access token",
)
async def oauth_callback(
    provider: str,
    code: str,
    state: str,
    auth_service: AuthService = Depends(get_auth_service),
):
    """OAuth登录回调"""
    user = await auth_service.authenticate_oauth_callback(provider, code, state)
    access_token = auth_service.create_access_token(user)
    login_response = LoginResponse(
        access_token=access_token,
        expires_in=auth_service.expires_in,
        user=to_user_response(user),
    )
    if auth_service.oauth_frontend_redirect_uri:
        return RedirectResponse(
            build_oauth_redirect_url(
                auth_service.oauth_frontend_redirect_uri,
                login_response,
            ),
            status_code=302,
        )
    return Response.success(
        msg="登录成功",
        data=login_response,
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
