from typing import Protocol

from app.domain.models.web_page import WebPage


class WebCrawler(Protocol):
    """网页正文抓取器协议"""

    async def fetch(self, url: str) -> WebPage:
        """抓取网页标题和正文"""
        ...
