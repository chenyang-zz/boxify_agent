import json
from typing import Any, Protocol

from app.domain.external.json_parser import JSONParser


async def parse_json_object(
    parser: JSONParser,
    content: Any,
    default_value: dict[str, Any],
) -> dict[str, Any]:
    """解析 LLM 返回的 JSON 对象，异常或非对象结果统一回退默认值。"""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    try:
        parsed = await parser.invoke(content, default_value=default_value)
    except Exception:
        return default_value
    if not isinstance(parsed, dict):
        return default_value
    return parsed
