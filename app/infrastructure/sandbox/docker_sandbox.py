import asyncio
import io
import logging
import socket
from typing import BinaryIO, Optional, Self, cast
from uuid import uuid4

import docker
import httpx
from async_lru import alru_cache
from docker.errors import APIError, NotFound
from docker.models.containers import Container
from docker.models.resource import Model

from app.domain.external.browser import Browser
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.models.tool_result import ToolResult
from app.infrastructure.external.browser.playwright_browser import PlaywrightBrowser
from core.config import get_settings

logger = logging.getLogger(__name__)


class DockerSandbox(Sandbox):
    """基于Docker的沙箱服务"""

    def __init__(
        self,
        ip: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> None:
        """构造函数，完成Docker沙箱扩展创建"""
        self.client = httpx.AsyncClient(timeout=6000)
        self._ip = ip
        self._container_name = container_name
        self._base_url = f"http://{ip}:8080"
        self._vnc_url = f"ws://{ip}:5901"
        self._cdp_url = f"http://{ip}:9222"

    @property
    def id(self) -> str:
        """获取沙箱的唯一id，使用容器名字作为唯一id"""
        if not self._container_name:
            return "boxify-sandbox"
        return self._container_name

    @property
    def vnc_url(self) -> str:
        return self._vnc_url

    @property
    def cdp_url(self) -> str:
        return self._cdp_url

    @classmethod
    @alru_cache(maxsize=128, typed=True)
    async def _resolve_hostname_to_ip(cls, hostname: str) -> Optional[str]:
        """将docker容器主机/地址转换成ipv4格式数据"""

        try:
            # 首先解析传递的hostname是不是ip
            try:
                socket.inet_pton(socket.AF_INET, hostname)
                return hostname
            except OSError:
                pass

            # 使用socket获取地址信息
            addr_info = socket.getaddrinfo(hostname, None, family=socket.AF_INET)

            # 判断地址信息是否存在，如果存在则返回第一个ipv4地址
            if addr_info and len(addr_info) > 0:
                return cast(str, addr_info[0][4][0])

            return None
        except Exception as e:
            logger.error(f"解析Docker容器主机地址{hostname}失败: {str(e)}")
            return None

    @classmethod
    def _get_container_ip(cls, container: Container) -> str:
        """根据传递的容器获取ip信息"""
        # 获取inspect网络设置
        network_settings = container.attrs["NetworkSettings"]
        ip_address = network_settings["IPAddress"]

        # 判断容器是否存在ip，如果不存在则从networks中获取
        if not ip_address and "Networks" in network_settings:
            networks = network_settings["Networks"]
            # 循环遍历每一项网络配置
            for network_name, network_config in networks.items():
                if "IPAddress" in network_config and network_config["IPAddress"]:
                    ip_address = network_config["IPAddress"]
                    break

        return ip_address

    @classmethod
    def _create_task(cls) -> Self:
        """创建沙箱容器的异步任务"""
        # 获取系统配置信息
        settings = get_settings()

        # 构建容器名字
        image = settings.sandbox_image
        name_prefix = settings.sandbox_name_prefix
        container_name = f"{name_prefix}-{str(uuid4())[:8]}"

        try:
            # 创建一个docker客户端
            docker_client = docker.from_env()

            # 预配置容器信息
            container_config = {
                "image": image,
                "name": container_name,
                "detach": True,
                "remove": True,
                "environment": {
                    "SERVICE_TIMEOUT_MINUTES": settings.sandbox_ttl_minutes,
                    "CHROME_ARGS": settings.sandbox_chrome_args,
                    "HTTPS_PROXY": settings.sandbox_https_proxy,
                    "HTTP_PROXY": settings.sandbox_http_proxy,
                    "NO_PROXY": settings.sandbox_no_proxy,
                },
            }

            # 判断是否传递了网络
            if settings.sandbox_network:
                container_config["network"] = settings.sandbox_network

            # 调用docker客户端容器运行参数创建沙箱
            container: Container = docker_client.containers.run(**container_config)

            # 重载并刷新容器信息
            container.reload()
            ip = cls._get_container_ip(container)

            return cls(ip=ip, container_name=container_name)
        except Exception as e:
            logger.error(f"常见Docker沙箱容器失败: {str(e)}")
            raise Exception(f"常见Docker沙箱容器失败: {str(e)}")

    @classmethod
    async def create(cls) -> Self:
        """类方法，创建沙箱容器"""
        # 获取系统配置信息
        settings = get_settings()

        # 判断是否使用现成的沙箱
        if settings.sandbox_address:
            # 将沙箱主机/地址解析成ip
            ip = await cls._resolve_hostname_to_ip(settings.sandbox_address)
            return cls(ip=ip)

        # 使用子线程创建一个容器后返回
        return await asyncio.to_thread(cls._create_task)

    async def destroy(self) -> bool:
        """销毁当前的DockerSandbox实例"""
        try:
            # 关闭httpx客户端
            if self.client:
                await self.client.aclose()

            # 关闭并移除容器
            if self._container_name:
                docker_client = docker.from_env()
                docker_client.containers.get(self._container_name).remove(force=True)

            return True
        except Exception as e:
            logger.error(f"销毁当前Docker沙箱[{self._container_name}]失败: {str(e)}")
            return False

    @classmethod
    async def get(cls, id: str) -> Optional[Self]:
        """根据传递的id获取沙箱实例"""
        # 先获取系统配置并判断是否直连沙箱
        settings = get_settings()
        if settings.sandbox_address:
            try:
                ip = await cls._resolve_hostname_to_ip(settings.sandbox_address)
                return cls(ip=ip, container_name=id)
            except Exception as e:
                logger.error(f"解析沙箱地址失败: {str(e)}")
                return None

        try:
            # 创建docker客户端并根据容器名字获取容器
            docker_client = docker.from_env()

            try:
                # 根据id获取容器
                container = docker_client.containers.get(id)
                container.reload()

                # 检查容器是否正常运行
                if container.status != "running":
                    logger.warning(f"容器存在但未运行，容器名字: {id}")
                    return None

                # 获取容器的ip地址
                ip = cls._get_container_ip(container)
                if not ip:
                    return None

                return cls(ip=ip, container_name=id)
            except NotFound:
                # 找不到容器
                logger.warning(f"该容器找不到可能被销毁: {str(id)}")
                return None
            except APIError as e:
                # Docker容器守护进程出错
                logger.error(f"Docker API出错: {str(e)}")
                return None
            finally:
                # 显示关闭docker client
                docker_client.close()
        except Exception as e:
            # 其他错误统一捕获
            logger.error(f"获取沙箱发生位置错误: {str(e)}")
            return None

    async def get_browser(self, llm: Optional[LLM] = None) -> Browser:
        """获取沙箱中的浏览器实例"""
        return PlaywrightBrowser(cdp_url=self.cdp_url, llm=llm)

    async def ensure_sandbox(self) -> None:
        """确保沙箱一定存在并且服务全部开启才执行后续步骤"""
        # 定义最大重试次数和重试间隔
        max_retries = 30
        retry_interval = 2

        # 循环请求获取supervisor状态并判断服务是否正常
        for _ in range(max_retries):
            try:
                # 调用client客户端向沙箱发起api请求获取状态
                response = await self.client.get(
                    f"{self._base_url}/api/supervisor/status"
                )
                response.raise_for_status()

                # 将响应状态转换成ToolResult
                tool_result = ToolResult.from_sandbox(**response.json())

                # 判断是否执行成功
                if not tool_result.success:
                    logger.warning(f"Supervisor进程状态检测失败: {tool_result.message}")
                    await asyncio.sleep(retry_interval)
                    continue

                # 读取services数据并判断
                services = tool_result.data or []
                if not services:
                    logger.warning("Supervisor进程中未发现任何服务")
                    await asyncio.sleep(retry_interval)
                    continue

                # 循环遍历所有服务并判断是否全部正常运行
                all_running = True
                non_running_services = []
                for service in services:
                    service_name = service.get("name", "unknown")
                    state_name = service.get("statename", "")

                    # 判断state_name是否为RUNNING
                    if state_name != "RUNNING":
                        all_running = False
                        non_running_services.append(f"{service_name}({state_name})")

                # 判断是否所有服务都启动
                if all_running:
                    logger.info("Sandbox Supervisor所有进程服务运行正常")
                    return
                else:
                    logger.info(
                        f"正在等待Sandbox Supervisor进程服务运行，还未运行的服务列表: {non_running_services}"
                    )
                    await asyncio.sleep(retry_interval)
            except Exception as e:
                logger.warning(f"无法确认Sandbox Supervisor进程状态: {str(e)}")
                await asyncio.sleep(retry_interval)

        # 经过max_retries次检测后还无法确认则抛出错误
        logger.error(f"在经过{max_retries}次尝试后仍无法确认Sandbox Supervisor状态信息")
        raise Exception(
            f"在经过{max_retries}次尝试后仍无法确认Sandbox Supervisor状态信息"
        )

    async def read_file(
        self,
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: int = 10000,
    ) -> ToolResult:
        """读取沙箱中指定路径的文件内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/read-file",
            json={
                "filepath": filepath,
                "start_line": start_line,
                "end_line": end_line,
                "sudo": sudo,
                "max_length": max_length,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def write_file(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> ToolResult:
        """向沙箱中指定文件写入内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/write-file",
            json={
                "filepath": filepath,
                "content": content,
                "append": append,
                "leading_newline": leading_newline,
                "trailing_newline": trailing_newline,
                "sudo": sudo,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def replace_in_file(
        self,
        filepath: str,
        old_str: str,
        new_str: str,
        sudo: bool = False,
    ) -> ToolResult:
        """替换沙箱中文件的旧内容为指定内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/replace-in-file",
            json={
                "filepath": filepath,
                "old_str": old_str,
                "new_str": new_str,
                "sudo": sudo,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def search_in_file(
        self,
        filepath: str,
        regex: str,
        sudo: bool = False,
    ) -> ToolResult:
        """搜索沙箱中指定文件的内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/search-in-file",
            json={
                "filepath": filepath,
                "regex": regex,
                "sudo": sudo,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def find_files(
        self,
        dir_path: str,
        glob_pattern: str,
    ) -> ToolResult:
        """查找沙箱中指定牡蛎的文件列表"""
        response = await self.client.post(
            f"{self._base_url}/api/file/find-files",
            json={
                "dir_path": dir_path,
                "glob_pattern": glob_pattern,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def list_files(self, dir_path: str) -> ToolResult:
        """传递目录列出沙箱指定目录下的所有文件"""
        return await self.find_files(dir_path, "*")

    async def check_file_exists(self, filepath: str) -> ToolResult:
        """根据指定路径检查沙箱中指定文件是否存在"""
        response = await self.client.post(
            f"{self._base_url}/api/file/check-file-exists",
            json={
                "filepath": filepath,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def delete_file(self, filepath: str) -> ToolResult:
        """传递路径删除指定文件"""
        response = await self.client.post(
            f"{self._base_url}/api/file/delete-file",
            json={
                "filepath": filepath,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def upload_file(
        self,
        file_data: BinaryIO,
        filepath: str,
        filename: Optional[str] = None,
    ) -> ToolResult:
        """将文件源上传至沙箱指定位置"""
        # 预配置上传数据
        files = {"file": (filename or "upload", file_data, "application/octet-stream")}
        data = {"filepath": filepath}

        # 发起请求上传数据获取响应
        response = await self.client.post(
            f"{self._base_url}/api/file/upload-file",
            files=files,
            data=data,
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def download_file(self, filepath: str) -> BinaryIO:
        """从沙箱中下载文件"""
        response = await self.client.get(
            f"{self._base_url}/api/file/download-file",
            params={
                "filepath": filepath,
            },
        )
        response.raise_for_status()

        return io.BytesIO(response.content)

    async def exec_command(
        self,
        session_id: str,
        exec_dir: str,
        command: str,
    ) -> ToolResult:
        """在沙箱中执行命令"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/exec-command",
            json={
                "session_id": session_id,
                "exec_dir": exec_dir,
                "command": command,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def read_shell_output(
        self,
        session_id: str,
        console: bool = False,
    ) -> ToolResult:
        """读取沙箱中shell的输出"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/read-shell-output",
            json={
                "session_id": session_id,
                "console": console,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def write_shell_input(
        self,
        session_id: str,
        input_text: str,
        press_enter: bool = True,
    ) -> ToolResult:
        """向沙箱的Shell进程写入数据"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/write-shell-input",
            json={
                "session_id": session_id,
                "input_text": input_text,
                "press_enter": press_enter,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def wait_process(
        self,
        session_id: str,
        seconds: Optional[int] = None,
    ) -> ToolResult:
        """等待沙箱中进程的执行"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/wait-process",
            json={
                "session_id": session_id,
                "seconds": seconds,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())

    async def kill_process(self, session_id: str) -> ToolResult:
        """杀死沙箱中的指定进程"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/kill-process",
            json={
                "session_id": session_id,
            },
        )
        response.raise_for_status()

        return ToolResult.from_sandbox(**response.json())
