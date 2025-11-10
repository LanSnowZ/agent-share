# -*- coding: utf-8 -*-
"""
MCP 客户端管理模块
提供全局单例 MCP 客户端，避免重复连接服务器
"""

import asyncio
import os
import threading
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from src.mcp_client import MCPClient

load_dotenv()

# 全局单例
_mcp_client_instance: Optional[MCPClient] = None
_mcp_client_lock = threading.Lock()
_mcp_event_loop: Optional[asyncio.AbstractEventLoop] = None


def get_or_create_mcp_client() -> Optional[MCPClient]:
    """
    获取或创建 MCP 客户端单例

    Returns:
        MCPClient 实例，如果初始化失败则返回 None
    """
    global _mcp_client_instance, _mcp_event_loop

    if _mcp_client_instance is None:
        with _mcp_client_lock:
            if _mcp_client_instance is None:
                try:
                    logger.info("正在初始化 MCP 客户端...")

                    # 创建新的事件循环
                    _mcp_event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(_mcp_event_loop)

                    # 创建客户端
                    client = MCPClient()

                    # 连接到服务器
                    config_path = os.getenv("MCP_CONFIG_PATH", "configs/mcp_config.json")
                    if not os.path.exists(config_path):
                        logger.warning(f"MCP 配置文件不存在: {config_path}")
                        return None

                    _mcp_event_loop.run_until_complete(client.connect_to_servers(config_path))

                    _mcp_client_instance = client
                    logger.info("MCP 客户端初始化成功")

                except Exception as e:
                    logger.error(f"初始化 MCP 客户端失败: {e}")
                    return None

    return _mcp_client_instance


def get_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """获取 MCP 使用的事件循环"""
    return _mcp_event_loop


def cleanup_mcp_client():
    """清理 MCP 客户端资源"""
    global _mcp_client_instance, _mcp_event_loop

    if _mcp_client_instance:
        try:
            logger.info("正在清理 MCP 客户端...")
            if _mcp_event_loop and not _mcp_event_loop.is_closed():
                _mcp_event_loop.run_until_complete(_mcp_client_instance.cleanup())
                _mcp_event_loop.close()
            _mcp_client_instance = None
            _mcp_event_loop = None
            logger.info("MCP 客户端清理完成")
        except Exception as e:
            logger.error(f"清理 MCP 客户端失败: {e}")
