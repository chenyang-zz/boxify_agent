import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from app.application.errors.exceptions import UnauthorizedError


class TokenService:
    """无状态JWT access token服务"""

    _algorithm = "HS256"

    def __init__(self, secret_key: str) -> None:
        if not secret_key:
            raise ValueError("AUTH_SECRET_KEY不能为空")
        self._secret_key = secret_key.encode("utf-8")

    def create_access_token(self, subject: str, expires_delta: timedelta) -> str:
        """为指定主体创建JWT access token"""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + expires_delta).timestamp()),
            "typ": "access",
        }
        header = {"alg": self._algorithm, "typ": "JWT"}
        signing_input = ".".join(
            [
                self._json_b64encode(header),
                self._json_b64encode(payload),
            ]
        )
        signature = self._sign(signing_input)
        return f"{signing_input}.{signature}"

    def verify_access_token(self, token: str) -> str:
        """校验JWT access token并返回subject"""
        try:
            header_text, payload_text, signature = token.split(".", 2)
            signing_input = f"{header_text}.{payload_text}"
            expected_signature = self._sign(signing_input)
            if not hmac.compare_digest(signature, expected_signature):
                raise UnauthorizedError()

            header = self._json_b64decode(header_text)
            payload = self._json_b64decode(payload_text)
            if header.get("alg") != self._algorithm or payload.get("typ") != "access":
                raise UnauthorizedError()

            expires_at = int(payload["exp"])
            if expires_at < int(datetime.now(timezone.utc).timestamp()):
                raise UnauthorizedError("登录已过期")

            subject = payload.get("sub")
            if not isinstance(subject, str) or not subject:
                raise UnauthorizedError()
            return subject
        except UnauthorizedError:
            raise
        except Exception as exc:
            raise UnauthorizedError() from exc

    def _sign(self, signing_input: str) -> str:
        digest = hmac.new(
            self._secret_key, signing_input.encode("ascii"), hashlib.sha256
        ).digest()
        return self._b64encode(digest)

    @classmethod
    def _json_b64encode(cls, value: Dict[str, Any]) -> str:
        data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return cls._b64encode(data)

    @staticmethod
    def _json_b64decode(value: str) -> Dict[str, Any]:
        return json.loads(TokenService._b64decode(value))

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(f"{value}{padding}")
