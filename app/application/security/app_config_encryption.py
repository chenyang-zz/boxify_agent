import base64
import copy
import hashlib
import json
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet

from app.domain.models.app_config import AppConfig


class AppConfigEncryption:
    """应用配置敏感字段加解密服务"""

    _prefix = "enc:v1:"

    def __init__(self, encryption_key: Optional[str]) -> None:
        self._fernet = (
            Fernet(self._normalize_key(encryption_key)) if encryption_key else None
        )

    def encrypt_app_config(self, app_config: AppConfig) -> Dict[str, Any]:
        """加密应用配置中的敏感字段并返回可入库字典"""
        return self.encrypt_app_config_data(app_config.model_dump(mode="json"))

    def encrypt_app_config_data(self, app_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """加密应用配置字典中的敏感字段"""
        data = copy.deepcopy(app_config_data)
        if not self._fernet:
            return data

        llm_config = data.get("llm_config") or {}
        api_key = llm_config.get("api_key")
        if api_key:
            llm_config["api_key"] = self._encrypt_text(api_key)

        for server_config in self._mcp_servers(data).values():
            env = server_config.get("env")
            if env:
                server_config["env"] = self._encrypt_json(env)
            headers = server_config.get("headers")
            if headers:
                server_config["headers"] = self._encrypt_json(headers)

        return data

    def decrypt_app_config_data(self, app_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """解密应用配置字典中的敏感字段"""
        data = copy.deepcopy(app_config_data)
        if not self._fernet:
            return data

        llm_config = data.get("llm_config") or {}
        api_key = llm_config.get("api_key")
        if isinstance(api_key, str) and self._is_encrypted(api_key):
            llm_config["api_key"] = self._decrypt_text(api_key)

        for server_config in self._mcp_servers(data).values():
            env = server_config.get("env")
            if isinstance(env, str) and self._is_encrypted(env):
                server_config["env"] = self._decrypt_json(env)
            headers = server_config.get("headers")
            if isinstance(headers, str) and self._is_encrypted(headers):
                server_config["headers"] = self._decrypt_json(headers)

        return data

    def _encrypt_json(self, value: Any) -> str:
        return self._encrypt_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    def _decrypt_json(self, value: str) -> Any:
        return json.loads(self._decrypt_text(value))

    def _encrypt_text(self, value: str) -> str:
        if self._is_encrypted(value):
            return value
        assert self._fernet is not None
        token = self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        return f"{self._prefix}{token}"

    def _decrypt_text(self, value: str) -> str:
        assert self._fernet is not None
        token = value.removeprefix(self._prefix)
        return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")

    @classmethod
    def _is_encrypted(cls, value: str) -> bool:
        return value.startswith(cls._prefix)

    @staticmethod
    def _mcp_servers(app_config_data: Dict[str, Any]) -> Dict[str, Any]:
        mcp_config = app_config_data.get("mcp_config") or {}
        return mcp_config.get("mcpServers") or {}

    @staticmethod
    def _normalize_key(encryption_key: str) -> bytes:
        key_bytes = encryption_key.encode("utf-8")
        try:
            Fernet(key_bytes)
            return key_bytes
        except ValueError:
            digest = hashlib.sha256(key_bytes).digest()
            return base64.urlsafe_b64encode(digest)
