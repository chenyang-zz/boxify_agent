from fastapi import APIRouter

from .documents import router as documents_router
from .tags import router as tags_router

# Notebook 路由聚合包只负责路径拼装，具体接口放在子模块中维护。
router = APIRouter(prefix="/notebook")
router.include_router(documents_router)
router.include_router(tags_router)
