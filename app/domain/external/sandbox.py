from typing import BinaryIO, Optional, Protocol, Self

from app.domain.external.browser import Browser
from app.domain.external.llm import LLM
from app.domain.models.tool_result import ToolResult


class Sandbox(Protocol):
    """沙箱服务扩展协议"""

    async def exec_command(
        self, session_id: str, exec_dir: str, command: str
    ) -> ToolResult:
        """执行命令"""
        ...

    async def read_shell_output(
        self, session_id: str, console: bool = False
    ) -> ToolResult:
        """读取shell执行结果"""
        ...

    async def wait_process(
        self, session_id: str, seconds: Optional[int] = None
    ) -> ToolResult:
        """等待进程执行"""
        ...

    async def write_shell_input(
        self, session_id: str, input_text: str, press_enter: bool = True
    ) -> ToolResult:
        """写入shell输入"""
        ...

    async def kill_process(self, session_id: str) -> ToolResult:
        """杀死进程"""
        ...

    async def write_file(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> ToolResult:
        """写入文件"""
        ...

    async def read_file(
        self,
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: int = 10000,
    ) -> ToolResult:
        """读取文件"""
        ...

    async def check_file_exists(self, filepath: str) -> ToolResult:
        """检查文件是否存在"""
        ...

    async def delete_file(self, filepath: str) -> ToolResult:
        """删除文件"""
        ...

    async def list_files(self, dir_path: str) -> ToolResult:
        """列出目录下的文件"""
        ...

    async def replace_in_file(
        self,
        filepath: str,
        old_str: str,
        new_str: str,
        sudo: bool = False,
    ) -> ToolResult:
        """替换文件中的字符串"""
        ...

    async def search_in_file(
        self,
        filepath: str,
        regex: str,
        sudo: bool = False,
    ) -> ToolResult:
        """在文件中搜索字符串"""
        ...

    async def find_files(
        self,
        dir_path: str,
        glob_pattern: str,
    ) -> ToolResult:
        """查找符合条件的文件"""
        ...

    async def upload_file(
        self,
        file_data: BinaryIO,
        filepath: str,
        filename: Optional[str] = None,
    ) -> ToolResult:
        """上传文件"""
        ...

    async def download_file(
        self,
        filepath: str,
    ) -> BinaryIO:
        """下载文件"""
        ...

    async def ensure_sandbox(self) -> None:
        """确保沙箱环境"""
        ...

    async def destroy(self) -> bool:
        """销毁沙箱环境"""
        ...

    async def get_browser(self, llm: Optional[LLM] = None) -> Browser:
        """获取浏览器"""
        ...

    @property
    def id(self) -> str:
        """只读属性，获取沙箱ID"""
        ...

    @property
    def cdp_url(self) -> str:
        """只读属性，获取CDP URL"""
        ...

    @property
    def vnc_url(self) -> str:
        """只读属性，获取VNC URL"""
        ...

    @classmethod
    async def create(cls) -> Self:
        """类方法，创建沙箱实例"""
        ...

    @classmethod
    async def get(cls, id: str) -> Optional[Self]:
        """类方法，获取沙箱实例"""
        ...
