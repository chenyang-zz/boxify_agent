from app.utils.collections import merge_unique_strings


def test_merge_unique_strings_keeps_order_and_deduplicates() -> None:
    assert merge_unique_strings(
        ["核心事实", "稳定偏好"],
        ["稳定偏好", "长期目标"],
    ) == ["核心事实", "稳定偏好", "长期目标"]


def test_merge_unique_strings_filters_empty_values() -> None:
    assert merge_unique_strings(["", "画像"], ["", "画像", "标签"]) == [
        "画像",
        "标签",
    ]
