from datetime import datetime


def parse_optional_datetime(value: str | None) -> datetime | None:
    """解析可选 ISO 时间字符串，空值、NULL 或非法值返回空。"""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.upper() == "NULL":
        return None
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
