from typing import Optional

from app.domain.external.sandbox import Sandbox
from app.domain.models.tool_result import ToolResult
from app.domain.services.tools.base import BaseTool, tool


class ShellTool(BaseTool):
    """Shell工具集，提供Shell交互相关功能"""

    name: str = "shell"

    def __init__(self, sandbox: Sandbox) -> None:
        """构造函数，完成Shell工具集初始化"""
        super().__init__()
        self._sandbox = sandbox

    @tool(
        name="shell_execute",
        description="在指定 Shell 会话中执行命令。可用于运行代码、安装依赖包或文件管理。",
        parameters={
            "session_id": {
                "type": "string",
                "description": "目标 Shell 会话的唯一标识符",
            },
            "exec_dir": {
                "type": "string",
                "description": "执行命令的工作目录（必须使用绝对路径）",
            },
            "command": {
                "type": "string",
                "description": "要执行的 Shell 命令",
            },
        },
        required=["session_id", "exec_dir", "command"],
    )
    async def shell_execute(
        self,
        session_id: str,
        exec_dir: str,
        command: str,
    ) -> ToolResult:
        """执行shell脚本"""
        return await self._sandbox.exec_command(session_id, exec_dir, command)

    @tool(
        name="shell_read_output",
        description="查看指定 Shell 会话的内容。用于检查命令执行结果或监控输出。",
        parameters={
            "session_id": {
                "type": "string",
                "description": "目标 Shell 会话的唯一标识符",
            }
        },
        required=["session_id"],
    )
    async def shell_read_output(self, session_id: str) -> ToolResult:
        """读取指定 Shell 会话的输出内容"""
        return await self._sandbox.read_shell_output(session_id)

    @tool(
        name="shell_wait_process",
        description="等待指定 Shell 会话中正在运行的进程返回。在运行耗时较长的命令后使用。",
        parameters={
            "session_id": {
                "type": "string",
                "description": "目标 Shell 会话的唯一标识符",
            },
            "seconds": {
                "type": "integer",
                "description": "可选参数，等待时长（秒）",
            },
        },
        required=["session_id"],
    )
    async def shell_wait_process(
        self, session_id: str, seconds: Optional[int] = None
    ) -> ToolResult:
        """等待指定 Shell 会话中正在运行的进程返回"""
        return await self._sandbox.wait_process(session_id, seconds)

    @tool(
        name="shell_write_input",
        description="向指定 Shell 会话中正在运行的进程写入输入。用于响应交互式命令提示符。",
        parameters={
            "session_id": {
                "type": "string",
                "description": "目标 Shell 会话的唯一标识符",
            },
            "input_text": {
                "type": "string",
                "description": "要写入进程的输入内容",
            },
            "process_enter": {
                "type": "boolean",
                "description": "输入后是否按下回车键",
            },
        },
        required=["session_id", "input_text", "press_enter"],
    )
    async def shell_write_input(
        self,
        session_id: str,
        input_text: str,
        press_enter: bool,
    ) -> ToolResult:
        """向指定 Shell 会话中正在运行的进程写入输入"""
        return await self._sandbox.write_shell_input(
            session_id, input_text, press_enter
        )

    @tool(
        name="shell_kill_process",
        description="在指定 Shell 会话中终止正在运行的进程。用于停止时间运行的进程或处理卡死的命令。",
        parameters={
            "session_id": {
                "type": "string",
                "description": "目标 Shell 会话的唯一标识符",
            },
        },
        required=["session_id"],
    )
    async def shell_kill_process(self, session_id: str) -> ToolResult:
        """在指定 Shell 会话中终止正在运行的进程"""
        return await self._sandbox.kill_process(session_id)
