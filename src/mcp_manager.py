# -*- coding: utf-8 -*-
"""
MCP 客户端管理模块
提供全局单例 MCP 客户端，避免重复连接服务器
"""

import asyncio
import os
import threading
from typing import Optional, Coroutine, Any
from concurrent.futures import Future

from dotenv import load_dotenv
from loguru import logger

from src.mcp_client import MCPClient

load_dotenv()

# 全局单例
_mcp_client_instance: Optional[MCPClient] = None
_mcp_client_lock = threading.Lock()
_mcp_event_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_loop_thread: Optional[threading.Thread] = None


def _run_event_loop(loop: asyncio.AbstractEventLoop):
    """在独立线程中运行事件循环"""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def get_or_create_mcp_client() -> Optional[MCPClient]:
    """
    获取或创建 MCP 客户端单例

    Returns:
        MCPClient 实例，如果初始化失败则返回 None
    """
    global _mcp_client_instance, _mcp_event_loop, _mcp_loop_thread

    if _mcp_client_instance is None:
        with _mcp_client_lock:
            if _mcp_client_instance is None:
                try:
                    logger.info("正在初始化 MCP 客户端...")

                    # 创建新的事件循环并在独立线程中运行
                    _mcp_event_loop = asyncio.new_event_loop()
                    _mcp_loop_thread = threading.Thread(
                        target=_run_event_loop,
                        args=(_mcp_event_loop,),
                        daemon=True,
                        name="MCP-EventLoop",
                    )
                    _mcp_loop_thread.start()

                    # 创建客户端
                    client = MCPClient()

                    # 连接到服务器
                    config_path = os.getenv("MCP_CONFIG_PATH", "configs/mcp_config.json")
                    if not os.path.exists(config_path):
                        logger.warning(f"MCP 配置文件不存在: {config_path}")
                        return None

                    # 使用线程安全方式在事件循环中执行连接
                    future = asyncio.run_coroutine_threadsafe(
                        client.connect_to_servers(config_path), _mcp_event_loop
                    )
                    future.result(timeout=30)  # 等待最多30秒

                    _mcp_client_instance = client
                    logger.info("MCP 客户端初始化成功")

                except Exception as e:
                    logger.error(f"初始化 MCP 客户端失败: {e}")
                    return None

    return _mcp_client_instance


def get_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """获取 MCP 使用的事件循环"""
    return _mcp_event_loop


def run_async_task(coro: Coroutine) -> Any:
    """
    在 MCP 事件循环中安全地运行异步任务

    Args:
        coro: 要运行的协程

    Returns:
        协程的执行结果

    Raises:
        RuntimeError: 如果事件循环不可用
        TimeoutError: 如果任务执行超时
    """
    if not _mcp_event_loop:
        raise RuntimeError("MCP 事件循环未初始化")

    future = asyncio.run_coroutine_threadsafe(coro, _mcp_event_loop)
    return future.result(timeout=60)  # 默认超时60秒


def cleanup_mcp_client():
    """清理 MCP 客户端资源"""
    global _mcp_client_instance, _mcp_event_loop, _mcp_loop_thread

    if _mcp_client_instance:
        try:
            logger.info("正在清理 MCP 客户端...")
            if _mcp_event_loop and not _mcp_event_loop.is_closed():
                # 使用线程安全方式执行清理
                future = asyncio.run_coroutine_threadsafe(
                    _mcp_client_instance.cleanup(), _mcp_event_loop
                )
                future.result(timeout=10)

                # 停止事件循环
                _mcp_event_loop.call_soon_threadsafe(_mcp_event_loop.stop)

                # 等待线程结束
                if _mcp_loop_thread and _mcp_loop_thread.is_alive():
                    _mcp_loop_thread.join(timeout=5)

                _mcp_event_loop.close()

            _mcp_client_instance = None
            _mcp_event_loop = None
            _mcp_loop_thread = None
            logger.info("MCP 客户端清理完成")
        except Exception as e:
            logger.error(f"清理 MCP 客户端失败: {e}")
