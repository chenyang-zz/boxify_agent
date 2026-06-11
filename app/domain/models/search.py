from typing import List, Optional

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """搜索结果条目数据模型"""

    url: str  # 搜索条目url链接
    title: str  # 搜索条目标题
    snippet: str = ""  # 搜索条目摘要信息


class SearchResults(BaseModel):
    """搜索结果数据模型"""

    query: str  # 查询query
    date_range: Optional[str] = None  # 日期筛选范围
    total_results: int = 0  # 搜索结果条数
    results: List[SearchResultItem] = Field(default_factory=list)  # 搜索结果
