import base64
import hashlib
import hmac
import secrets


class PasswordHasher:
    """基于标准库PBKDF2的密码哈希工具"""

    _algorithm = "pbkdf2_sha256"
    _iterations = 260000
    _salt_bytes = 16

    @classmethod
    def hash_password(cls, password: str) -> str:
        """生成包含算法、迭代次数、salt和hash的密码摘要"""
        salt = secrets.token_bytes(cls._salt_bytes)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, cls._iterations
        )
        return "$".join(
            [
                cls._algorithm,
                str(cls._iterations),
                cls._b64encode(salt),
                cls._b64encode(digest),
            ]
        )

    @classmethod
    def verify_password(cls, password: str, password_hash: str) -> bool:
        """校验明文密码是否匹配密码摘要"""
        try:
            algorithm, iterations_text, salt_text, digest_text = password_hash.split(
                "$", 3
            )
            if algorithm != cls._algorithm:
                return False
            iterations = int(iterations_text)
            salt = cls._b64decode(salt_text)
            expected_digest = cls._b64decode(digest_text)
        except (ValueError, TypeError):
            return False

        actual_digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual_digest, expected_digest)

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(f"{value}{padding}")
