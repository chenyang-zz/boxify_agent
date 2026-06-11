from typing import Optional, Protocol

from app.domain.models.tool_result import ToolResult


class Browser(Protocol):
    """浏览器服务扩展协议"""

    async def view_page(self) -> ToolResult:
        """获取当前浏览器页面的内容源码"""
        ...

    async def navigate(self, url: str) -> ToolResult:
        """传递对应的url使用浏览器导航到该页面"""
        ...

    async def restart(self, url: str) -> ToolResult:
        """重启浏览器并导航到指定页面"""
        ...

    async def click(
        self,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """传递对应元素的索引或者xy坐标实现点击功能"""
        ...

    async def input(
        self,
        text: str,
        press_enter: bool,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """传递文本内容和是否按回车键实现输入功能"""
        ...

    async def move_mouse(
        self,
        coordinate_x: float,
        coordinate_y: float,
    ) -> ToolResult:
        """移动鼠标到指定坐标"""
        ...

    async def press_key(
        self,
        key: str,
    ) -> ToolResult:
        """按下指定键或者组合键"""
        ...

    async def select_option(
        self,
        index: int,
        option: int,
    ) -> ToolResult:
        """选择下拉框中的选项"""
        ...

    async def scroll_up(
        self,
        to_top: Optional[bool] = None,
    ) -> ToolResult:
        """向上滚动浏览器，如果没传递to_top则向上滚动一页，反之滚动到顶部"""
        ...

    async def scroll_down(
        self,
        to_bottom: Optional[bool] = None,
    ) -> ToolResult:
        """向下滚动浏览器，如果没传递to_bottom则向下滚动一页，反之滚动到底部"""
        ...

    async def screenshot(
        self,
        full_page: Optional[bool] = None,
    ) -> bytes:
        """截取当前页面的截图，如果full_page为True则截取整个页面，反之截取当前可见部分"""
        ...

    async def console_exec(
        self,
        javascript: str,
    ) -> ToolResult:
        """执行JavaScript代码"""
        ...

    async def console_view(
        self,
        max_lines: Optional[int] = None,
    ) -> ToolResult:
        """查看控制台输出，不传递max_lines，获取所有"""
        ...
