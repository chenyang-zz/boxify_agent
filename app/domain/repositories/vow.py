#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/6/8 16:19
@Author : chenyangzhao542@gmail.com
@File   : vow.py
"""

from abc import ABC, abstractmethod
from typing import Self

from app.domain.repositories.file_repository import FileRepository
from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.user_repository import UserRepository


class IUnitOfWork(ABC):
    """Uow模式协议接口"""

    file: FileRepository
    session: SessionRepository
    user: UserRepository

    @abstractmethod
    async def commit(self):
        """提交数据库数据持久化"""
        ...

    @abstractmethod
    async def rollback(self):
        """数据库回滚"""

    @abstractmethod
    async def __aenter__(self) -> Self:
        """进入上下文管理器"""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        ...
