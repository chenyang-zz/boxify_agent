from bs4 import BeautifulSoup

from app.domain.external.web_crawler import WebCrawler
from app.domain.models.web_page import WebPage


class HttpWebCrawler(WebCrawler):
    """HTTP 网页正文抓取器，用于 URL 导入文档。"""

    async def fetch(self, url: str) -> WebPage:
        """抓取网页并清理脚本、样式等非正文节点。"""
        import httpx

        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "BoxifyNotebook/1.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = soup.get_text("\n")
        # 保留换行语义但移除空白行，方便后续统一分块。
        normalized = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        return WebPage(title=title[:120], text=normalized)
