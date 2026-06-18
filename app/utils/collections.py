def merge_unique_strings(left: list[str], right: list[str]) -> list[str]:
    """合并字符串列表，过滤空值并保持首次出现顺序。"""
    merged: list[str] = []
    for value in [*left, *right]:
        if value and value not in merged:
            merged.append(value)
    return merged
