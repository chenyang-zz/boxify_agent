import hashlib


def content_digest(content: bytes, length: int = 16) -> str:
    """返回内容的短SHA256摘要，用于日志排查而不暴露原始内容。"""
    return hashlib.sha256(content).hexdigest()[:length]
