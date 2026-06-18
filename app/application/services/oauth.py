import base64
import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import Prehashed
from cryptography.hazmat.primitives.hashes import SHA256

from app.application.errors.exceptions import BadRequestError, UnauthorizedError


@dataclass(frozen=True)
class OAuthAuthorization:
    """OAuth授权跳转信息"""

    authorization_url: str


@dataclass(frozen=True)
class OAuthIdentity:
    """第三方登录身份信息"""

    provider: str
    subject: str
    username: str
    email: str | None = None
    avatar_url: str | None = None


@dataclass(frozen=True)
class OAuthState:
    """OAuth callback校验状态"""

    provider: str
    code_verifier: str
    nonce: str | None


class OAuthProvider(Protocol):
    provider: str

    def build_authorization_url(
        self,
        state: str,
        code_challenge: str,
        nonce: str | None,
    ) -> str:
        """构建provider授权地址"""
        ...

    async def exchange_code_for_identity(
        self,
        code: str,
        code_verifier: str,
        nonce: str | None,
    ) -> OAuthIdentity:
        """用授权码换取第三方身份"""
        ...


class OAuthStateCodec:
    """加密签名OAuth state，避免服务端保存临时会话"""

    def __init__(self, secret_key: str, ttl_seconds: int = 600) -> None:
        if not secret_key:
            raise ValueError("AUTH_SECRET_KEY不能为空")
        self._fernet = Fernet(
            base64.urlsafe_b64encode(
                hashlib.sha256(f"boxify-oauth-state:{secret_key}".encode()).digest()
            )
        )
        self._ttl_seconds = ttl_seconds

    def encode(self, provider: str, code_verifier: str, nonce: str | None) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "provider": provider,
            "code_verifier": code_verifier,
            "nonce": nonce,
            "csrf": secrets.token_urlsafe(32),
            "exp": int((now + timedelta(seconds=self._ttl_seconds)).timestamp()),
        }
        return self._fernet.encrypt(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")

    def decode(self, state: str, expected_provider: str) -> OAuthState:
        try:
            payload = json.loads(self._fernet.decrypt(state.encode("ascii")))
            expires_at = int(payload["exp"])
            if expires_at < int(datetime.now(timezone.utc).timestamp()):
                raise UnauthorizedError("OAuth登录状态无效或已过期")
            provider = str(payload["provider"])
            if provider != expected_provider:
                raise UnauthorizedError("OAuth登录状态无效或已过期")
            code_verifier = str(payload["code_verifier"])
            nonce_value = payload.get("nonce")
            nonce = str(nonce_value) if nonce_value else None
            if not code_verifier:
                raise UnauthorizedError("OAuth登录状态无效或已过期")
            return OAuthState(
                provider=provider,
                code_verifier=code_verifier,
                nonce=nonce,
            )
        except UnauthorizedError:
            raise
        except (InvalidToken, ValueError, KeyError, TypeError, UnicodeError) as exc:
            raise UnauthorizedError("OAuth登录状态无效或已过期") from exc


def create_code_verifier() -> str:
    return secrets.token_urlsafe(64)


def create_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def create_nonce() -> str:
    return secrets.token_urlsafe(32)


class GitHubOAuthProvider:
    provider = "github"
    _authorization_endpoint = "https://github.com/login/oauth/authorize"
    _token_endpoint = "https://github.com/login/oauth/access_token"
    _user_endpoint = "https://api.github.com/user"
    _emails_endpoint = "https://api.github.com/user/emails"
    _scope = "read:user user:email"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        timeout_seconds: float = 10,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._timeout_seconds = timeout_seconds

    def _ensure_configured(self) -> None:
        """校验 GitHub OAuth 必需配置，缺失时提前返回业务错误。"""
        if not self._client_id or not self._client_secret or not self._redirect_uri:
            raise BadRequestError("GitHub OAuth未配置")

    def build_authorization_url(
        self,
        state: str,
        code_challenge: str,
        nonce: str | None,
    ) -> str:
        self._ensure_configured()
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "scope": self._scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self._authorization_endpoint}?{urlencode(params)}"

    async def exchange_code_for_identity(
        self,
        code: str,
        code_verifier: str,
        nonce: str | None,
    ) -> OAuthIdentity:
        self._ensure_configured()
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            token_response = await client.post(
                self._token_endpoint,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._redirect_uri,
                    "code_verifier": code_verifier,
                },
                headers={"Accept": "application/json"},
            )
            token_payload = self._parse_json_response(token_response)
            access_token = token_payload.get("access_token")
            if not isinstance(access_token, str) or not access_token:
                raise UnauthorizedError("GitHub授权失败")

            auth_headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            user_payload = self._parse_json_response(
                await client.get(self._user_endpoint, headers=auth_headers)
            )
            emails_payload = self._parse_json_response(
                await client.get(self._emails_endpoint, headers=auth_headers)
            )
        subject = str(user_payload.get("id") or "")
        if not subject:
            raise UnauthorizedError("GitHub用户信息无效")
        email = _select_github_email(emails_payload)
        username = (
            str(user_payload.get("login") or "")
            or _username_from_email(email)
            or f"github_{subject}"
        )
        avatar_url = user_payload.get("avatar_url")
        return OAuthIdentity(
            provider=self.provider,
            subject=subject,
            username=username,
            email=email,
            avatar_url=str(avatar_url) if avatar_url else None,
        )

    @staticmethod
    def _parse_json_response(response: httpx.Response) -> Any:
        """解析 GitHub 响应，HTTP 错误统一转换为授权失败。"""
        if response.status_code >= 400:
            raise UnauthorizedError("GitHub授权失败")
        return response.json()


class GoogleOAuthProvider:
    provider = "google"
    _authorization_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
    _token_endpoint = "https://oauth2.googleapis.com/token"
    _jwks_endpoint = "https://www.googleapis.com/oauth2/v3/certs"
    _scope = "openid email profile"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        timeout_seconds: float = 10,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._timeout_seconds = timeout_seconds

    def _ensure_configured(self) -> None:
        """校验 Google OAuth 必需配置，缺失时提前返回业务错误。"""
        if not self._client_id or not self._client_secret or not self._redirect_uri:
            raise BadRequestError("Google OAuth未配置")

    def build_authorization_url(
        self,
        state: str,
        code_challenge: str,
        nonce: str | None,
    ) -> str:
        self._ensure_configured()
        params = {
            "client_id": self._client_id,
            "response_type": "code",
            "scope": self._scope,
            "redirect_uri": self._redirect_uri,
            "state": state,
            "nonce": nonce or "",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self._authorization_endpoint}?{urlencode(params)}"

    async def exchange_code_for_identity(
        self,
        code: str,
        code_verifier: str,
        nonce: str | None,
    ) -> OAuthIdentity:
        self._ensure_configured()
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            token_response = await client.post(
                self._token_endpoint,
                data={
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": self._redirect_uri,
                    "grant_type": "authorization_code",
                    "code_verifier": code_verifier,
                },
            )
            token_payload = self._parse_json_response(token_response)
            id_token = token_payload.get("id_token")
            if not isinstance(id_token, str) or not id_token:
                raise UnauthorizedError("Google授权失败")
            jwks = self._parse_json_response(await client.get(self._jwks_endpoint))
        claims = _verify_google_id_token(id_token, jwks, self._client_id, nonce)
        subject = str(claims.get("sub") or "")
        if not subject:
            raise UnauthorizedError("Google用户信息无效")
        email = claims.get("email")
        name = claims.get("name")
        picture = claims.get("picture")
        return OAuthIdentity(
            provider=self.provider,
            subject=subject,
            username=str(name or _username_from_email(email) or f"google_{subject}"),
            email=str(email) if email else None,
            avatar_url=str(picture) if picture else None,
        )

    @staticmethod
    def _parse_json_response(response: httpx.Response) -> Any:
        """解析 Google 响应，HTTP 错误统一转换为授权失败。"""
        if response.status_code >= 400:
            raise UnauthorizedError("Google授权失败")
        return response.json()


def build_oauth_providers(
    github_client_id: str,
    github_client_secret: str,
    github_redirect_uri: str,
    google_client_id: str,
    google_client_secret: str,
    google_redirect_uri: str,
    timeout_seconds: float,
) -> dict[str, OAuthProvider]:
    return {
        "github": GitHubOAuthProvider(
            client_id=github_client_id,
            client_secret=github_client_secret,
            redirect_uri=github_redirect_uri,
            timeout_seconds=timeout_seconds,
        ),
        "google": GoogleOAuthProvider(
            client_id=google_client_id,
            client_secret=google_client_secret,
            redirect_uri=google_redirect_uri,
            timeout_seconds=timeout_seconds,
        ),
    }


def _select_github_email(emails_payload: Any) -> str | None:
    """从 GitHub 邮箱列表中选择已验证的主邮箱。"""
    if not isinstance(emails_payload, list):
        return None
    for email in emails_payload:
        if not isinstance(email, dict):
            continue
        if email.get("primary") is True and email.get("verified") is True:
            value = email.get("email")
            return str(value) if value else None
    return None


def _username_from_email(email: Any) -> str | None:
    """从邮箱本地部分提取候选用户名。"""
    if not isinstance(email, str) or "@" not in email:
        return None
    return email.split("@", 1)[0] or None


def _verify_google_id_token(
    id_token: str,
    jwks: Any,
    client_id: str,
    nonce: str | None,
) -> dict[str, Any]:
    """校验 Google ID Token 签名、签发方、受众、过期时间和 nonce。"""
    try:
        header_text, payload_text, signature_text = id_token.split(".", 2)
        header = _json_b64decode(header_text)
        payload = _json_b64decode(payload_text)
        key = _find_jwk(jwks, str(header["kid"]))
        public_key = _rsa_key_from_jwk(key)
        signature = _b64decode(signature_text)
        signing_input = f"{header_text}.{payload_text}".encode("ascii")
        digest = hashlib.sha256(signing_input).digest()
        public_key.verify(
            signature,
            digest,
            padding.PKCS1v15(),
            Prehashed(SHA256()),
        )
        if header.get("alg") != "RS256":
            raise UnauthorizedError("Google授权失败")
        if payload.get("iss") not in {
            "https://accounts.google.com",
            "accounts.google.com",
        }:
            raise UnauthorizedError("Google授权失败")
        if payload.get("aud") != client_id:
            raise UnauthorizedError("Google授权失败")
        expires_at = int(payload["exp"])
        if expires_at < int(datetime.now(timezone.utc).timestamp()):
            raise UnauthorizedError("Google授权失败")
        if nonce and payload.get("nonce") != nonce:
            raise UnauthorizedError("Google授权失败")
        return payload
    except UnauthorizedError:
        raise
    except Exception as exc:
        raise UnauthorizedError("Google授权失败") from exc


def _find_jwk(jwks: Any, kid: str) -> dict[str, Any]:
    """从 Google JWKS 中查找匹配 kid 的公钥描述。"""
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        raise UnauthorizedError("Google授权失败")
    for key in keys:
        if isinstance(key, dict) and key.get("kid") == kid:
            return key
    raise UnauthorizedError("Google授权失败")


def _rsa_key_from_jwk(jwk: dict[str, Any]) -> rsa.RSAPublicKey:
    """将 RSA JWK 转换为 cryptography 可验证的公钥对象。"""
    if jwk.get("kty") != "RSA":
        raise UnauthorizedError("Google授权失败")
    numbers = rsa.RSAPublicNumbers(
        e=int.from_bytes(_b64decode(str(jwk["e"])), "big"),
        n=int.from_bytes(_b64decode(str(jwk["n"])), "big"),
    )
    return numbers.public_key()


def _json_b64decode(value: str) -> dict[str, Any]:
    """解码 JWT 片段并解析为 JSON 对象。"""
    return json.loads(_b64decode(value))


def _b64decode(value: str) -> bytes:
    """补齐 base64url 填充符后解码为字节。"""
    padding_text = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding_text}")
