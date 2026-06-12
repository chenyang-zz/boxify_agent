import asyncio
import io
import logging
import uuid
from io import SEEK_END
from typing import AsyncGenerator, BinaryIO, Callable, List, Optional

from fastapi import UploadFile
from pydantic import TypeAdapter

from app.domain.external.browser import Browser
from app.domain.external.file_storage import FileStorage
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.external.task import Task, TaskRunner
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.event import (
    A2AToolContent,
    BaseEvent,
    BrowserToolContent,
    DoneEvent,
    ErrorEvent,
    Event,
    FileToolContent,
    MCPToolContent,
    MessageEvent,
    SearchToolContent,
    ShellToolContent,
    TitleEvent,
    ToolEvent,
    ToolEventStatus,
    WaitEvent,
)
from app.domain.models.file import File
from app.domain.models.message import Message
from app.domain.models.search import SearchResults
from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import ToolResult
from app.domain.repositories.vow import IUnitOfWork
from app.domain.services.flows.planner_react import PlannerReActFlow
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.mcp import MCPTool

logger = logging.getLogger(__name__)


class AgentTaskRunner(TaskRunner):
    """基于Agent智能体的任务运行器"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: LLM,  # 大语言模型
        agent_config: AgentConfig,  # 智能体配置
        mcp_config: MCPConfig,  # mcp配置
        a2a_config: A2AConfig,  # a2a配置
        session_id: str,  # 会话id
        file_storage: FileStorage,  # 文件存储桶
        json_parser: JSONParser,  # json解析器
        browser: Browser,  # 浏览器
        search_engine: SearchEngine,  # 搜索引擎
        sandbox: Sandbox,  # 沙箱
    ) -> None:
        """构造函数，完成Agent任务运行器的创建"""
        self._uow_factory = uow_factory
        self._session_id = session_id
        self._sandbox = sandbox
        self._mcp_config = mcp_config
        self._mcp_tool = MCPTool()
        self._a2a_config = a2a_config
        self._a2a_tool = A2ATool()
        self._file_storage = file_storage
        self._browser = browser
        self._flow = PlannerReActFlow(
            uow_factory=uow_factory,
            llm=llm,
            agent_config=agent_config,
            session_id=session_id,
            json_parser=json_parser,
            browser=browser,
            sandbox=sandbox,
            search_engine=search_engine,
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
        )

    async def _put_and_add_event(
        self,
        task: Task,
        event: BaseEvent,
    ) -> None:
        """往指定任务的消息队列中添加事件"""
        # 往任务的输出消息队列中新增事件
        event_id = await task.output_stream.put(event.model_dump_json())
        if event_id:
            event.id = event_id

        # 将事件添加到对应会话中
        async with self._uow_factory() as uow:
            await uow.session.add_event(self._session_id, event)

    @classmethod
    async def _pop_event(cls, task: Task) -> Optional[Event]:
        """从任务的输入流中获取事件信息"""
        # 从任务task中读取数据
        event_id, event_str = await task.input_stream.pop()
        if event_str is None:
            logger.warning("AgentTaskRunner接收到空消息")
            return None

        # 使用pydantic和type类型将字符串转换成事件
        event = TypeAdapter(Event).validate_json(event_str)
        event.id = event_id

        return event

    async def _sync_file_to_sandbox(self, file_id: str) -> Optional[File]:
        """根据文件id将文件同步到沙箱中"""

        try:
            # 调用文件存储下载文件信息
            file_data, file = await self._file_storage.download_file(file_id)

            # 组装沙箱文件路径
            filepath = f"/home/ubuntu/upload/{file.filename}"

            # 调用沙箱将文件上传至沙箱
            tool_result = await self._sandbox.upload_file(
                file_data=file_data,
                filepath=filepath,
                filename=file.filename,
            )

            # 判断是否上传成功
            if tool_result.success:
                file.filepath = filepath
                async with self._uow_factory() as uow:
                    await uow.file.save(file)
                return file
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步文件[{file_id}]失败: {str(e)}")

    async def _sync_message_attachments_to_sandbox(self, event: MessageEvent) -> None:
        """将消息事件中的附件同步到沙箱中"""
        # 定义附件列表
        attachments: List[File] = []

        try:
            # 判断消息中是否存在附件
            if event.attachments:
                # 循环遍历所有消息附件
                for attachment in event.attachments:
                    # 根据同步文件的id将数据同步到沙箱中
                    file = await self._sync_file_to_sandbox(attachment.id)

                    # 文件是否同步成功
                    if file:
                        attachments.append(file)
                        async with self._uow_factory() as uow:
                            await uow.session.add_file(self._session_id, file)

                # 更新消息事件中的attachments
                event.attachments = attachments
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到沙箱失败: {str(e)}")

    @classmethod
    def _get_stream_size(cls, f: BinaryIO) -> int:
        """根据传递的文件流，计算文件的大小"""

        # 记录当前文件指针位置
        current_pos = f.tell()

        # 将指针移动到文件末尾，seek: 0:偏移量 2:相对文件末尾
        f.seek(0, SEEK_END)

        # 获取当前位置
        size = f.tell()

        # 回复指针到原始位置
        f.seek(current_pos)

        return size

    async def _sync_file_to_storage(self, filepath: str) -> Optional[File]:
        """将沙箱中指定文件路径数据同步到存储桶中"""
        try:
            # 根据文件路径从会话中查找文件数据
            async with self._uow_factory() as uow:
                file = await uow.session.get_file_by_path(
                    self._session_id, filepath
                )

            # 从沙箱中下载文件
            file_data = await self._sandbox.download_file(filepath)

            # 判断会话中的文件是否存在
            if file:
                async with self._uow_factory() as uow:
                    await uow.session.remove_file(self._session_id, file.id)

            # 提取文件名字、文件信息并更新文件路径
            filename = filepath.split("/")[-1]
            upload_file = UploadFile(
                file=file_data,
                filename=filename,
                size=self._get_stream_size(file_data),
            )

            # 上传文件到文件存储桶
            file = await self._file_storage.upload_file(upload_file)
            file.filepath = filepath

            # 往会话总新增一个文件信息
            async with self._uow_factory() as uow:
                await uow.session.add_file(self._session_id, file)

            return file
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到文件存储桶失败: {str(e)}")

    async def _sync_message_attachments_to_storage(self, event: MessageEvent) -> None:
        """将消息事件的附件同步到文件存储桶中"""
        # 定义附件列表存储数据
        attachments: List[File] = []

        try:
            # 判断消息中是否存在附件
            if event.attachments:
                # 循环遍历附件
                for attachment in event.attachments:
                    # 根据文件路径将数据同步到文件存储桶中
                    file = await self._sync_file_to_storage(attachment.filepath)

                    if file:
                        attachments.append(file)

                # 更新事件中的附件列表资源
                event.attachments = attachments
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到存储桶失败: {str(e)}")

    async def _get_browser_screenshot(self) -> str:
        """获取浏览器截图并返回截图文件对于的id"""
        # 调用浏览器完成截图
        screenshot = await self._browser.screenshot()

        # 将浏览器截图上传到文件存春中
        file = await self._file_storage.upload_file(
            UploadFile(file=io.BytesIO(screenshot), filename=f"{str(uuid.uuid4())}.png")
        )

        return file.id

    async def _handle_tool_event(self, event: ToolEvent) -> None:
        """额外处理工具消息，使其前端交互更友好"""
        try:
            # 如果事件状态为已调用则执行以下代码
            if event.status == ToolEventStatus.CALLED:
                if event.tool_name == "browser":
                    # 工具为浏览器则补全浏览器工具内容
                    event.tool_content = BrowserToolContent(
                        screenshot=await self._get_browser_screenshot()
                    )
                elif event.tool_name == "search" and event.function_result:
                    # 工具为搜索则添加搜索工具内容
                    search_results: ToolResult[SearchResults] = event.function_result
                    logger.info(f"搜索工具结果: {search_results}")
                    if search_results.data:
                        event.tool_content = SearchToolContent(
                            results=search_results.data.results
                        )
                elif event.tool_name == "shell":
                    # 工具为shell则生成shell工具内容
                    if "session_id" in event.function_args:
                        shell_result = await self._sandbox.read_shell_output(
                            event.function_args["session_id"],
                            console=True,
                        )
                        event.tool_content = ShellToolContent(
                            console=shell_result.data.get("console_records", [])
                        )
                    else:
                        event.tool_content = ShellToolContent(console="(No Console)")
                elif event.tool_name == "file":
                    # 工具为file则将文件同步到对象存储
                    if "filepath" in event.function_args:
                        filepath = event.function_args["filepath"]
                        file_read_result = await self._sandbox.read_file(filepath)
                        file_content: str = file_read_result.data.get("content", "")
                        event.tool_content = FileToolContent(content=file_content)
                        await self._sync_file_to_storage(filepath)
                    else:
                        event.tool_content = FileToolContent(content="(No Content)")
                elif event.tool_name in ["mcp", "a2a"]:
                    # 工具为mcp/a2a则处理调用结果
                    if event.function_result:
                        # 如果结果包含data则提取data
                        if (
                            hasattr(event.function_result, "data")
                            and event.function_result.data
                        ):
                            logger.info(
                                f"MCP/A2A工具调用结果: {event.function_result.data}"
                            )
                            event.tool_content = (
                                MCPToolContent(result=event.function_result.data)
                                if event.tool_name == "mcp"
                                else A2AToolContent(
                                    a2a_result=event.function_result.data
                                )
                            )
                        elif (
                            hasattr(event.function_result, "success")
                            and event.function_result.success
                        ):
                            # mcp/a2a工具调用正常，但是无结果产生
                            logger.info(
                                f"MCP/A2A工具调用成功返回，但无结果: {event.function_result}"
                            )
                            result_data = (
                                event.function_result.model_dump()
                                if hasattr(event.function_result, "model_dump")
                                else str(event.function_result)
                            )
                            event.tool_content = (
                                MCPToolContent(result=result_data)
                                if event.tool_name == "mcp"
                                else A2AToolContent(a2a_result=result_data)
                            )
                        else:
                            # 其他情况将结果站换成字符串进行传递
                            logger.info(f"MCP/A2A工具调用结果: {event.function_result}")
                            event.tool_content = (
                                MCPToolContent(result=str(event.function_result))
                                if event.tool_name == "mcp"
                                else A2AToolContent(
                                    a2a_result=str(event.function_result)
                                )
                            )
                    else:
                        logger.warning("MCP/A2A工具调用结果未发现")
                        event.tool_content = (
                            MCPToolContent(result="(MCP工具无可用结果)")
                            if event.tool_name == "mcp"
                            else A2AToolContent(a2a_result="(A2A智能体无可用结果")
                        )
        except Exception as e:
            logger.exception(f"AgentTaskRunner生成工具内容失败: {str(e)}")

    async def _run_flow(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """根据消息对象运行PlannerReActFlow"""
        # 判断传递的消息是否为空
        if not message.message:
            logger.warning("AgentTaskRunner接收了一条空消息")
            yield ErrorEvent(error="空消息错误")
            return

        # 调用流并运行获取事件信息
        async for event in self._flow.invoke(message):
            # 判断是否为工具事件，如果是则额外处理
            if isinstance(event, ToolEvent):
                await self._handle_tool_event(event)
            elif isinstance(event, MessageEvent):
                # 如果是消息事件则将AI消息中的附件同步到存储中
                await self._sync_message_attachments_to_storage(event)

            # 将事件直接返回
            yield event

    async def _cleanup_tools(self) -> None:
        """清理MCP和A2A工具资源，确保在同一任务上下文中释放

        注意：该方法必须在初始化MCP/A2A的同一个asyncio Task中调用，
        否则anyio的cancel scope会检测到任务上下文切换并抛出RuntimeError。
        """
        try:
            if self._mcp_tool:
                await self._mcp_tool.cleanup()
        except Exception as e:
            logger.warning(f"清理MCP工具资源时出错: {e}")
        try:
            if self._a2a_tool and self._a2a_tool.manager:
                await self._a2a_tool.manager.cleanup()
        except Exception as e:
            logger.warning(f"清理A2A工具资源时出错: {e}")

    async def invoke(self, task: Task) -> None:
        """根据传递的任务处理agent消息队列并运行agent流"""
        try:
            # 确保沙箱、mcp、a2a均初始化成功
            logger.info("AgentTaskRunner任务处理开始")
            await self._sandbox.ensure_sandbox()
            await self._mcp_tool.initialize(self._mcp_config)
            await self._a2a_tool.initialize(self._a2a_config)

            # 循环读取任务中的输入消息队列
            while not await task.input_stream.is_empty():
                # 从输入流汇中获取数据
                event = await self._pop_event(task)
                if event is None:
                    continue

                message = ""

                # 判断事件类型是否为消息事件，如果是则处理消息事件并将部件同步到沙箱中
                if isinstance(event, MessageEvent):
                    message = event.message or ""
                    await self._sync_message_attachments_to_sandbox(event)
                    logger.info(f"AgentTaskRunner接收到新消息: {message[:50]}")

                    # 将消息事件转换成消息对象
                    message_obj = Message(
                        message=message,
                        attachments=[
                            attachment.filepath for attachment in event.attachments
                        ],
                    )

                    # 传递消息对象并运行PlannerReActFlow
                    async for event in self._run_flow(message_obj):
                        # 将得到的事件添加到消息队列中
                        await self._put_and_add_event(task, event)

                        # 如果类型事件为标题事件则更新会话标题
                        if isinstance(event, TitleEvent):
                            async with self._uow_factory() as uow:
                                await uow.session.update_title(
                                    self._session_id, event.title
                                )
                        elif isinstance(event, MessageEvent):
                            # 如果事件为消息事件，则更新最新消息并新增未读消息数
                            async with self._uow_factory() as uow:
                                await uow.session.update_latest_message(
                                    self._session_id,
                                    event.message,
                                    event.created_at,
                                )
                                await uow.session.increment_unread_message_count(
                                    self._session_id
                                )
                        elif isinstance(event, WaitEvent):
                            # 如果事件为等待，则更新会话状态并终止程序
                            async with self._uow_factory() as uow:
                                await uow.session.update_status(
                                    self._session_id, SessionStatus.WAITING
                                )
                            return

                        # 判断如果输入消息队列不为空则跳出循环
                        # 表示有新的消息输入
                        if not await task.input_stream.is_empty():
                            break

            # 更新会话状态为已完成
            async with self._uow_factory() as uow:
                await uow.session.update_status(
                    self._session_id, SessionStatus.COMPLETED
                )

        except asyncio.CancelledError:
            # 异步任务被取消，推送结束事件并更新状态
            logger.info("AgentTaskRunner运行取消")
            await self._put_and_add_event(task, DoneEvent())
            async with self._uow_factory() as uow:
                await uow.session.update_status(
                    self._session_id, SessionStatus.COMPLETED
                )
            raise
        except Exception as e:
            # 记录日志并往任务队列/消息队列中写入异常事件并更新会话状态
            logger.exception(f"AgentTaskRunner运行出错: {str(e)}")
            await self._put_and_add_event(
                task, ErrorEvent(error=f"AgentTaskRunner运行出错: {str(e)}")
            )
            async with self._uow_factory() as uow:
                await uow.session.update_status(
                    self._session_id, SessionStatus.COMPLETED
                )
        finally:
            # 在同一个asyncio Task上下文中清理MCP/A2A工具资源
            # 这是关键：streamablehttp_client内部使用anyio.create_task_group()，
            # 要求在同一个Task中进入和退出cancel scope，
            # 所以必须在invoke()的finally块（即初始化MCP的同一个Task）中清理
            await self._cleanup_tools()

    async def destroy(self) -> None:
        """销毁任务运行器并释放资源"""
        # 清除沙箱
        logger.info("开始清除销毁AgentTaskRunner资源")
        if self._sandbox:
            logger.info("销毁AgentTaskRunner中的沙箱环境")
            await self._sandbox.destroy()

        # 清除mcp和a2a工具（幂等操作，如果invoke()中已清理则不会重复执行）
        await self._cleanup_tools()

    async def on_done(self, task: Task) -> None:
        """任务结束时执行的回调函数"""
        logger.info("AgentTaskRunner任务执行结束")
