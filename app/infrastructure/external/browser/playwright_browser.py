import asyncio
import logging
from typing import Any, List, Optional, cast

from markdownify import markdownify
from playwright.async_api import Browser, Page, Playwright, async_playwright

from app.domain.external.browser import Browser as BrowserProtocol
from app.domain.external.llm import LLM
from app.domain.models.tool_result import ToolResult
from app.infrastructure.external.browser.playwright_browser_func import (
    GET_INTERACTIVE_ELEMENTS_FUNC,
    GET_VISIBLE_CONTENT_FUNC,
    INJECT_CONSOLE_LOGS_FUNC,
)

logger = logging.getLogger(__name__)


class PlaywrightBrowser(BrowserProtocol):
    """基于Playwright管理的浏览器扩展"""

    def __init__(
        self,
        cdp_url: str,  # CDP连接地址
        llm: Optional[
            LLM
        ] = None,  # 可选参数，传递LLM，如果传递了会使用LLM对页面内容进行整理编程markdown格式
    ) -> None:
        """构造函数，完成Playwright浏览器的初始化"""

        # llm相关
        self._llm: Optional[LLM] = llm

        # 浏览器相关
        self._cdp_url: str = cdp_url
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    async def _ensure_browser(self) -> None:
        """确保浏览器存在，如果不存在则初始化"""
        if not self._browser or not self._page:
            if not await self.initialize():
                raise Exception("初始化Playwright浏览器失败")

    async def _ensure_page(self) -> None:
        """确保页面存在，如果不存在则创建新页面"""
        await self._ensure_browser()
        self._browser = cast(Browser, self._browser)

        if not self._page:
            self._page = (
                await self._browser.new_page()
            )  # 等同于 self._browser.new_context().new_page()

        else:
            # 如果已经有页面，检查是否需要更新
            contexts = self._browser.contexts
            if contexts:
                # 获取默认上下文和页面列表
                default_context = contexts[0]
                pages = default_context.pages

                if pages:
                    # 获取最新页面并更新当前页面
                    latest_page = pages[-1]

                    if self._page != latest_page:
                        self._page = latest_page

    async def _extract_content(self) -> str:
        """提取页面内容并转换为Markdown格式"""
        if not self._page:
            return ""

        # 获取页面可见内容
        visible_content = await self._page.evaluate(GET_VISIBLE_CONTENT_FUNC)

        # 将可见内容转换为Markdown格式
        markdown_content = markdownify(visible_content)

        # 模型上下文有限
        max_content_length = min(len(markdown_content), 50000)

        # 如果有llm，使用llm进行整理
        if self._llm:
            response = await self._llm.invoke(
                [
                    {
                        "role": "system",
                        "content": "你是一名专业的网页信息提取助手。请从当前页面内容中提取所有信息并将其转换为markdown格式。",
                    },
                    {
                        "role": "user",
                        "content": markdown_content[:max_content_length],
                    },
                ]
            )
            return response.get("content", "")
        else:
            return markdown_content[:max_content_length]

    async def _extract_interactive_elements(self) -> List[str]:
        """提取所有可交互元素"""

        await self._ensure_page()
        self._page = cast(Page, self._page)

        # 清空缓存
        setattr(self._page, "interactive_elements_cache", [])

        # 执行js代码获取可交互元素
        interactive_elements = await self._page.evaluate(GET_INTERACTIVE_ELEMENTS_FUNC)

        # 缓存可交互元素
        setattr(self._page, "interactive_elements_cache", interactive_elements)

        # 格式化可交互元素
        formatted_elements: List[str] = []
        for element in interactive_elements:
            formatted_elements.append(
                f"{element['index']}:<{element['tag']}>{element['text']}</{element['tag']}>"
            )

        return formatted_elements

    async def _get_element_by_id(self, index: int) -> Optional[Any]:
        """根据传递的索引/id获取对应的元素"""
        # 判断也当前页面是否存在可交互元素缓存
        if not hasattr(self._page, "interactive_elements_cache") or index >= len(
            getattr(self._page, "interactive_elements_cache")
        ):
            return None

        if not self._page:
            return None

        # 构建选择器
        selector = f'[data-boxify-id="boxify-element-{index}"]'
        return await self._page.query_selector(selector)

    async def initialize(self) -> bool:
        """初始化并确保资源是可用的"""

        # 1.定义重试次数和重试间隔
        max_retries = 5
        retry_interval = 1

        # 2.循环重试初始化Playwright浏览器
        for attempt in range(max_retries):
            try:
                # 3.启动Playwright并连接到CDP
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.connect_over_cdp(
                    self._cdp_url
                )

                # 4.获取浏览器所有上下文
                contexts = self._browser.contexts

                # 如果只有一个上下文且页面是空白或Chrome新标签页，则使用该页面
                if contexts and len(contexts) == 1:
                    page = contexts[0].pages[0]

                    if (
                        page.url == "about:blank"
                        or page.url == "chrome://newtab/"
                        or page.url == "chrome://new-tab-page"
                        or not page.url
                    ):
                        self._page = page
                    else:
                        # 如果页面不是空白或Chrome新标签页，则创建新页面
                        self._page = await contexts[0].new_page()

                else:
                    context = (
                        contexts[0] if contexts else await self._browser.new_context()
                    )
                    self._page = await context.new_page()

                return True
            except Exception as e:
                # 清理资源
                await self.cleanup()

                if attempt == max_retries - 1:
                    logger.error(
                        f"初始化Playwright浏览器失败(已重试{max_retries}次): {str(e)}"
                    )
                    return False

                # 重试间隔指数增长，最大不超过10秒
                retry_interval = min(retry_interval * 2, 10)
                logger.warning(
                    f"初始化Playwright浏览器失败，即将进行第{attempt + 1}次重试: {str(e)}"
                )
                await asyncio.sleep(retry_interval)

        return False

    async def cleanup(self) -> None:
        """清理Playwright资源"""
        try:
            # 1.检查浏览器是否存在，存在则删除该浏览器下所有tabs页面
            if self._browser:
                # 2.获取浏览器的全部上下文
                contexts = self._browser.contexts
                if contexts:
                    # 3.遍历上下文，关闭每个上下文下的所有页面
                    for context in contexts:
                        # 4.获取上下文下的所有页面
                        pages = context.pages
                        if pages:
                            for page in pages:
                                # 5.如果页面未关闭，则关闭页面
                                if not page.is_closed():
                                    await page.close()

            # 6.如果页面未关闭，则关闭页面
            if self._page and not self._page.is_closed():
                await self._page.close()

            # 7.如果浏览器未关闭，则关闭浏览器
            if self._browser:
                await self._browser.close()

            # 8.如果Playwright未停止，则停止Playwright
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            # 清理过程中出错，记录日志但不影响后续操作
            logger.error(f"清理Playwright浏览器资源出错: {str(e)}")
        finally:
            # 重置所有资源
            self._page = None
            self._browser = None
            self._playwright = None

    async def wait_for_page_load(self, timeout: int = 15) -> bool:
        """等待页面加载完成"""

        # 确保页面存在
        await self._ensure_page()
        self._page = cast(Page, self._page)

        # 使用异步任务事件循环中的时间来作为开始时间
        start_time = asyncio.get_event_loop().time()
        check_interval = 5

        # 循环检查页面是否加载成功
        while asyncio.get_event_loop().time() - start_time < timeout:
            # 使用js代码判断页面是否加载成功
            is_completed = await self._page.evaluate(
                """() => document.readyState === 'complete'"""
            )
            if is_completed:
                return True

            #  未加载成功休眠对应时间
            await asyncio.sleep(check_interval)

        return False

    async def navigate(self, url: str) -> ToolResult:
        """导航到指定URL"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        try:
            # 清空交互式元素缓存
            setattr(self._page, "interactive_elements_cache", [])

            # 导航到指定URL
            await self._page.goto(url)

            return ToolResult(
                success=True,
                data={
                    "interactive_elements": await self._extract_interactive_elements(),
                },
            )
        except Exception as e:
            # 导航过程中出错，返回失败结果
            return ToolResult(
                success=False, message=f"浏览器导航到[{url}]失败: {str(e)}"
            )

    async def view_page(self) -> ToolResult:
        """获取当前页面的内容源码"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        # 等待页面加载完成
        await self.wait_for_page_load()

        interactive_elements = await self._extract_interactive_elements()

        return ToolResult(
            success=True,
            data={
                "content": await self._extract_content(),
                "interactive_elements": interactive_elements,
            },
        )

    async def input(
        self,
        text: str,
        press_enter: bool,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """在页面上输入文本"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        if coordinate_x is not None and coordinate_y is not None:
            await self._page.mouse.click(coordinate_x, coordinate_y)
            await self._page.keyboard.type(text)
        elif index is not None:
            try:
                # 尝试通过索引获取元素
                element = await self._get_element_by_id(index)
                if not element:
                    return ToolResult(
                        success=False, message="输入文本失败，该元素不存在"
                    )

                try:
                    # 尝试填充文本
                    await element.fill("")
                    await element.type(text)
                except Exception:
                    # 如果填充失败，点击后再输入
                    await element.click()
                    await element.type(text)
            except Exception as e:
                return ToolResult(
                    success=False,
                    message=f"输入文本失败: {str(e)}",
                )

        # 如果需要按下回车键，执行按键操作
        if press_enter:
            await self._page.keyboard.press("enter")

        return ToolResult(success=True)

    async def move_mouse(self, coordinate_x: float, coordinate_y: float) -> ToolResult:
        """移动鼠标到指定坐标"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        await self._page.mouse.move(coordinate_x, coordinate_y)
        return ToolResult(success=True)

    async def press_key(self, key: str) -> ToolResult:
        """按下指定键"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        await self._page.keyboard.press(key)
        return ToolResult(success=True)

    async def select_option(self, index: int, option: int) -> ToolResult:
        """选择下拉框中的选项"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        try:
            element = await self._get_element_by_id(index)
            if not element:
                return ToolResult(
                    success=False,
                    message=f"未找到索引为 {index} 的下拉菜单元素",
                )

            await element.select_option(index=option)
            return ToolResult(success=True)
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"选择下拉菜单选项失败: {str(e)}",
            )

    async def restart(self, url: str) -> ToolResult:
        """重启并跳转到指定url"""
        await self.cleanup()
        return await self.navigate(url)

    async def click(
        self,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult: ...

    async def scroll_up(self, to_top: Optional[bool] = None) -> ToolResult:
        """向上滚动页面"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        if to_top:
            await self._page.evaluate("window.scrollTo(0, 0)")
        else:
            await self._page.evaluate("window.scrollBy(0, -window.innerHeight)")

        return ToolResult(
            success=True,
        )

    async def scroll_down(self, to_bottom: Optional[bool] = None) -> ToolResult:
        """向下滚动页面"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        if to_bottom:
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            await self._page.evaluate("window.scrollBy(0, window.innerHeight)")

        return ToolResult(
            success=True,
        )

    async def screenshot(self, full_page: Optional[bool] = None) -> bytes:
        """截取当前页面的截图"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        screenshot_options = {
            "full_page": full_page,
            "type": "png",
        }

        return await self._page.screenshot(**screenshot_options)

    async def console_exec(self, javascript: str) -> ToolResult:
        """在页面控制台中执行js代码"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        try:
            await self._page.evaluate(INJECT_CONSOLE_LOGS_FUNC)
        except Exception as e:
            logger.error(f"注入window.console.logs失败: {str(e)}")

        result = await self._page.evaluate(javascript)
        return ToolResult(success=True, data={"result": result})

    async def console_view(self, max_lines: Optional[int] = None) -> ToolResult:
        """查看控制台输出"""
        await self._ensure_page()
        self._page = cast(Page, self._page)

        # 获取控制台日志
        logs = await self._page.evaluate("""() => {
            return window.console.logs || [];
        }""")

        if max_lines is not None:
            logs = logs[-max_lines:]

        return ToolResult(success=True, data={"logs": logs})
