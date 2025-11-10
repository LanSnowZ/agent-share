#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本，验证 MCP 集成功能
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

# 测试配置
BASE_URL = "http://127.0.0.1:5002"
TEST_USERNAME = "testuser"
TEST_PASSWORD = "testpass123"


def test_login():
    """测试登录"""
    print("🔐 测试登录...")
    response = requests.post(
        f"{BASE_URL}/api/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
    )
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print("✅ 登录成功")
            # 从 cookies 中获取 token
            cookies = response.cookies
            return cookies
        else:
            print(f"❌ 登录失败: {result.get('error')}")
            return None
    else:
        print(f"❌ 登录请求失败: {response.status_code}")
        return None


def test_mcp_chat(cookies, mcp_enabled=True):
    """测试 MCP 聊天功能"""
    mode = "MCP 模式" if mcp_enabled else "普通模式"
    print(f"\n💬 测试{mode}...")

    # 构建请求
    data = {
        "message": "请帮我搜索一下 Python asyncio 的最新文档",
        "model": "gpt-4o-mini",
        "shared_memory_enabled": False,
        "personal_memory_enabled": False,
        "mcp_enabled": mcp_enabled,  # 关键参数
    }

    print(f"📤 发送请求: {data['message']}")

    try:
        # 发送流式请求
        response = requests.post(
            f"{BASE_URL}/chat_direct",
            json=data,
            cookies=cookies,
            stream=True,
            timeout=60,
        )

        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return

        print(f"📥 接收流式响应 ({mode}):\n")

        # 处理流式响应
        full_response = ""
        tool_calls = []

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])

                        # 处理不同类型的事件
                        if "content" in event_data:
                            content = event_data["content"]
                            full_response += content
                            print(content, end="", flush=True)

                        elif "tool_status" in event_data:
                            status = event_data["tool_status"]
                            tool_name = event_data.get("tool_name", "unknown")
                            if status == "start":
                                print(
                                    f"\n\n🔧 [工具调用] 开始执行: {tool_name}", flush=True
                                )
                                args = event_data.get("arguments", {})
                                print(f"   参数: {args}", flush=True)
                                tool_calls.append(tool_name)
                            elif status == "end":
                                elapsed = event_data.get("elapsed_time", 0)
                                print(
                                    f"✅ [工具调用] 完成: {tool_name} (耗时: {elapsed:.2f}s)\n",
                                    flush=True,
                                )

                        elif "thinking" in event_data:
                            print(f"\n💭 {event_data['thinking']}", flush=True)

                        elif "error" in event_data:
                            print(f"\n❌ 错误: {event_data['error']}", flush=True)

                        elif "done" in event_data:
                            print("\n\n✅ 响应完成")
                            conversation_id = event_data.get("conversation_id")
                            if conversation_id:
                                print(f"📝 对话ID: {conversation_id}")
                            break

                    except json.JSONDecodeError as e:
                        print(f"\n⚠️ 解析JSON失败: {e}")
                        print(f"原始数据: {line}")

        # 总结
        print(f"\n\n{'=' * 60}")
        print(f"📊 测试总结 ({mode}):")
        print(f"  - 响应长度: {len(full_response)} 字符")
        if tool_calls:
            print(f"  - 工具调用次数: {len(tool_calls)}")
            print(f"  - 调用的工具: {', '.join(tool_calls)}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    print("🚀 开始 MCP 集成测试\n")

    # 1. 登录
    cookies = test_login()
    if not cookies:
        print("\n❌ 无法登录，测试终止")
        return

    # 2. 测试普通模式（不使用 MCP）
    test_mcp_chat(cookies, mcp_enabled=False)

    # 3. 测试 MCP 模式
    test_mcp_chat(cookies, mcp_enabled=True)

    print("\n🎉 所有测试完成！")


if __name__ == "__main__":
    main()
