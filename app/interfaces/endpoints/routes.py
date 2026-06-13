from fastapi import APIRouter, Depends

from app.interfaces.service_dependencies import require_active_user, require_bearer_token

from . import (
    app_config_routes,
    auth_routes,
    file_route,
    notebook,
    session_routes,
    status_routes,
)


def create_api_routes() -> APIRouter:
    """创建API路由，涵盖整个项目的所有路由管理"""
    # 1.创建APIRouter实例
    api_router = APIRouter()
    protected_dependencies = [Depends(require_bearer_token), Depends(require_active_user)]

    # 2.将各个模块添加到api_router中
    api_router.include_router(status_routes.router)
    api_router.include_router(auth_routes.router)
    api_router.include_router(
        app_config_routes.router, dependencies=protected_dependencies
    )
    api_router.include_router(file_route.router, dependencies=protected_dependencies)
    api_router.include_router(notebook.router, dependencies=protected_dependencies)
    api_router.include_router(
        session_routes.router, dependencies=protected_dependencies
    )

    # 3.返回api路由实例
    return api_router


router = create_api_routes()
