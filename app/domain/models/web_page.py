from pydantic import BaseModel


class WebPage(BaseModel):
    """网页抓取结果领域模型"""

    title: str
    text: str
