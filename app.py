# -*- coding: utf-8 -*-
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# 设置 Hugging Face 镜像源（解决连接超时问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from flask_cors import CORS

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 关键：先添加 sharememory_user，再添加 memoryos-pypi
# 这样 sharememory_user 的模块（如 prompts, config）优先级更高
sharememory_user_path = os.path.join(project_root, "sharememory_user")
sys.path.insert(0, sharememory_user_path)

memoryos_path = os.path.join(project_root, "memoryos-pypi")
sys.path.insert(0, memoryos_path)

# 导入MemoryOS用于个人记忆
from memoryos import Memoryos

# 导入评估提示词
from eval.evaluation_prompts import (
    get_baseline_answer_prompt,
    get_fusion_rag_prompt,
    get_rag_answer_prompt,
)
from sharememory_user.config import Config
from sharememory_user.models import UserProfile
from sharememory_user.pipeline_retrieve import RetrievePipeline
from sharememory_user.storage import JsonStore

app = Flask(__name__)
CORS(app)

# 全局变量
config = Config()
print(f"\n{'=' * 60}")
print("📂 配置信息:")
print(f"  - data_dir: {config.data_dir}")
print(f"  - memory_path: {config.memory_path}")
print(f"  - users_path: {config.users_path}")
print(f"{'=' * 60}\n")

store = JsonStore(config)
retrieve_pipeline = RetrievePipeline(config)
memoryos_instances = {}  # 存储每个用户的MemoryOS实例

# 延迟导入 IngestPipeline
ingest_pipeline = None


def get_ingest_pipeline():
    """延迟初始化共享记忆存储管道"""
    global ingest_pipeline
    if ingest_pipeline is None:
        # 清除 prompts 模块缓存，避免 memoryos 的 prompts 干扰
        if "prompts" in sys.modules:
            del sys.modules["prompts"]

        # 导入时确保作为包导入
        from sharememory_user.pipeline_ingest import (
            IngestPipeline as SharedIngestPipeline,
        )

        ingest_pipeline = SharedIngestPipeline(config)
    return ingest_pipeline


# 导入 MemoryOS 的工具函数用于检测思维链断裂
from memoryos import OpenAIClient

sys.path.insert(0, memoryos_path)
from utils import check_conversation_continuity

# 用户数据存储目录 - 使用memoryos_data结构，按照项目/用户层级组织
MEMORYOS_DATA_DIR = os.path.join(project_root, "eval", "memoryos_data")
os.makedirs(MEMORYOS_DATA_DIR, exist_ok=True)

# 全局：维护每个用户的当前对话链（用于检测思维链断裂）
user_conversation_chains = {}  # {user_id: [list of conversation pages]}


def save_chain_to_shared_memory(user_id: str, chain_pages: List[Dict]):
    """将对话链保存到共享记忆"""
    if not chain_pages or len(chain_pages) < 1:
        return

    try:
        # 获取ingest pipeline实例
        pipeline = get_ingest_pipeline()

        # 确保用户存在于共享记忆系统中
        user_profile = store.get_user(user_id)
        if not user_profile:
            user_config = get_user_config(user_id, "default_project")
            profile_text = user_config.get("user_profile", f"用户 {user_id}")
            pipeline.ensure_user(user_id, profile_text)

        # 将对话链转换为文本
        conversation_text = ""
        for page in chain_pages:
            user_msg = page.get("user_input", "")
            agent_msg = page.get("agent_response", "")
            timestamp = page.get("timestamp", "")
            conversation_text += f"User ({timestamp}): {user_msg}\n"
            conversation_text += f"Assistant ({timestamp}): {agent_msg}\n\n"

        # 存储到共享记忆
        memory_item = pipeline.ingest_dialog(user_id, conversation_text.strip())
        if memory_item:
            print(
                f"✅ 成功将思维链存储到共享记忆，Memory ID: {memory_item.id}, 对话轮数: {len(chain_pages)}"
            )
        else:
            print("⚠️ 思维链未通过质量检查，未存储到共享记忆")
    except Exception as e:
        print(f"❌ 存储思维链到共享记忆失败: {e}")


def check_and_store_chain_break(
    user_id: str, new_user_msg: str, new_agent_msg: str
) -> None:
    """检测思维链断裂并存储"""
    if user_id not in user_conversation_chains:
        user_conversation_chains[user_id] = []

    # 创建当前对话页面
    current_page = {
        "user_input": new_user_msg,
        "agent_response": new_agent_msg,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    chain = user_conversation_chains[user_id]

    # 如果有历史对话，检测是否连续
    if chain:
        last_page = chain[-1]

        # 获取 OpenAI 客户端用于连续性检测
        if user_id in memoryos_instances:
            memoryos_client = memoryos_instances[user_id].client
        else:
            # 使用全局配置创建临时客户端
            memoryos_client = OpenAIClient(
                api_key=config.openai_api_key, base_url=config.openai_api_base
            )

        # 检测对话连续性
        is_continuous = check_conversation_continuity(
            last_page, current_page, memoryos_client, model=config.llm_model_name
        )

        if not is_continuous:
            # 思维链断裂！保存之前的对话链
            print(f"💡 检测到用户 {user_id} 的思维链断裂！当前链长度: {len(chain)}")
            save_chain_to_shared_memory(user_id, chain)
            # 清空，开始新的思维链
            user_conversation_chains[user_id] = [current_page]
        else:
            # 对话连续，添加到当前链
            chain.append(current_page)
    else:
        # 第一条对话，直接添加
        user_conversation_chains[user_id].append(current_page)


def ensure_user_memoryos(
    user_id: str, project_name: str = "default_project"
) -> Optional[Memoryos]:
    """确保用户有MemoryOS实例，如果没有则创建"""
    project_name = user_id
    if user_id not in memoryos_instances:
        try:
            # 创建用户数据目录 - 按照项目/用户层级结构: eval/memoryos_data/{project}/users/{user_id}
            project_dir = os.path.join(MEMORYOS_DATA_DIR, project_name)
            user_data_dir = project_dir
            os.makedirs(user_data_dir, exist_ok=True)

            print(f"📁 创建MemoryOS数据目录: {user_data_dir}")

            # 初始化MemoryOS实例
            memoryos_instance = Memoryos(
                user_id=user_id,
                openai_api_key=config.openai_api_key,
                data_storage_path=user_data_dir,
                openai_base_url=config.openai_api_base,
                llm_model=config.llm_model_name,
                assistant_id="chat_assistant",
                short_term_capacity=3,
                mid_term_capacity=2000,
                long_term_knowledge_capacity=100,
                retrieval_queue_capacity=3,
                mid_term_heat_threshold=8,
                mid_term_similarity_threshold=0.7,
                embedding_model_name="all-MiniLM-L6-v2",
            )

            memoryos_instances[user_id] = memoryos_instance
            # print(f"  - 短期记忆容量: 3")
            # print(f"  - 数据存储路径: {user_data_dir}")
        except Exception as e:
            print(f"创建MemoryOS实例失败: {e}")
            import traceback

            traceback.print_exc()
            return None
    else:
        print(f"用户 {user_id} 的MemoryOS实例已存在，复用现有实例")

    return memoryos_instances.get(user_id)


def get_user_config(
    user_id: str, project_name: str = "default_project"
) -> Dict[str, Any]:
    """获取用户配置"""
    config_path = os.path.join(
        MEMORYOS_DATA_DIR, project_name, "users", user_id, "config.json"
    )
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_user_config(
    user_id: str, config_data: Dict[str, Any], project_name: str = "default_project"
) -> bool:
    """保存用户配置"""
    try:
        user_dir = os.path.join(MEMORYOS_DATA_DIR, project_name, "users", user_id)
        os.makedirs(user_dir, exist_ok=True)

        config_path = os.path.join(user_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存用户配置失败: {e}")
        return False


def get_chat_conversations(
    user_id: str, project_name: str = "default_project"
) -> List[Dict[str, Any]]:
    """获取用户的聊天对话列表"""
    conversations_path = os.path.join(
        MEMORYOS_DATA_DIR, project_name, "users", user_id, "conversations.json"
    )
    if os.path.exists(conversations_path):
        with open(conversations_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_chat_conversations(
    user_id: str,
    conversations: List[Dict[str, Any]],
    project_name: str = "default_project",
) -> bool:
    """保存用户的聊天对话列表"""
    try:
        user_dir = os.path.join(MEMORYOS_DATA_DIR, project_name, "users", user_id)
        os.makedirs(user_dir, exist_ok=True)

        conversations_path = os.path.join(user_dir, "conversations.json")
        with open(conversations_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存聊天对话失败: {e}")
        return False


def get_chat_messages(
    user_id: str, conversation_id: str, project_name: str = "default_project"
) -> Optional[Dict[str, Any]]:
    """获取指定对话的消息"""
    conversation_path = os.path.join(
        MEMORYOS_DATA_DIR,
        project_name,
        "users",
        user_id,
        "conversations",
        f"{conversation_id}.json",
    )
    if os.path.exists(conversation_path):
        with open(conversation_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_chat_conversation(
    username,
    conversation_id,
    user_message,
    ai_response,
    model,
    shared_memory_enabled=False,
    personal_memory_enabled=True,
    update_last_ai_message=False,
    user_message_only=False,
):
    """保存聊天对话到chat文件夹"""

    # 创建用户目录
    user_chat_dir = os.path.join(
        MEMORYOS_DATA_DIR, "default_project", "users", username
    )
    os.makedirs(user_chat_dir, exist_ok=True)

    # 如果没有conversation_id，创建一个新的
    if not conversation_id:
        conversation_id = f"chat_{int(time.time() * 1000)}"

    # 对话文件路径
    conversation_file = os.path.join(user_chat_dir, f"{conversation_id}.json")

    # 读取现有对话或创建新对话
    conversation_data = {
        "id": conversation_id,
        "username": username,
        "model": model,
        "shared_memory_enabled": shared_memory_enabled,
        "personal_memory_enabled": personal_memory_enabled,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [],
    }

    if os.path.exists(conversation_file):
        try:
            with open(conversation_file, "r", encoding="utf-8") as f:
                conversation_data = json.load(f)
        except Exception as e:
            print(f"读取对话文件失败: {e}")

    # 检查是否需要更新最后一条AI消息
    if update_last_ai_message and conversation_data.get("messages"):
        # 更新最后一条AI消息
        messages = conversation_data["messages"]
        ai_message_found = False
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["type"] == "assistant":
                messages[i]["content"] = ai_response
                messages[i]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ai_message_found = True
                break

        # 如果没有找到AI消息，则添加一个（这种情况发生在只保存了用户消息后）
        if not ai_message_found:
            new_ai_message = {
                "type": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            conversation_data["messages"].append(new_ai_message)
    elif user_message_only:
        # 只添加用户消息（AI回复稍后添加）
        new_user_message = {
            "type": "user",
            "content": user_message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        conversation_data["messages"].append(new_user_message)
    else:
        # 添加新消息对
        new_messages = [
            {
                "type": "user",
                "content": user_message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            {
                "type": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        ]

        conversation_data["messages"].extend(new_messages)

    conversation_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_data["shared_memory_enabled"] = shared_memory_enabled
    conversation_data["personal_memory_enabled"] = personal_memory_enabled

    # 生成对话标题（如果没有的话）
    if not conversation_data.get("title"):
        # 使用用户的第一条消息作为标题
        first_user_message = next(
            (
                msg["content"]
                for msg in conversation_data["messages"]
                if msg["type"] == "user"
            ),
            "新对话",
        )
        conversation_data["title"] = first_user_message[:30] + (
            "..." if len(first_user_message) > 30 else ""
        )

    # 保存对话
    try:
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        # print(f"✅ 对话已保存: {conversation_file}")
        return conversation_id
    except Exception as e:
        print(f"❌ 保存对话失败: {e}")
        return None


def load_conversation_history(username, conversation_id):
    """加载对话历史"""
    try:
        conversation_file = os.path.join(
            MEMORYOS_DATA_DIR,
            "default_project",
            "users",
            username,
            f"{conversation_id}.json",
        )

        if os.path.exists(conversation_file):
            with open(conversation_file, "r", encoding="utf-8") as f:
                conversation_data = json.load(f)
            print(
                f"✅ 加载对话历史成功，消息数: {len(conversation_data.get('messages', []))}"
            )
            return conversation_data
        else:
            # 新对话时文件不存在是正常的，不需要打印错误
            print(f"ℹ️ 新对话，尚无历史记录（对话ID: {conversation_id}）")
            return None
    except Exception as e:
        print(f"❌ 加载对话历史失败: {e}")
        return None


def get_chat_conversations(username, project_name="default_project"):
    """获取用户的对话列表"""
    try:
        user_dir = os.path.join(MEMORYOS_DATA_DIR, project_name, "users", username)
        if not os.path.exists(user_dir):
            return []

        conversations = []
        for filename in os.listdir(user_dir):
            if filename.endswith(".json"):
                conversation_id = filename[:-5]  # 去掉.json后缀
                conversation_file = os.path.join(user_dir, filename)

                try:
                    with open(conversation_file, "r", encoding="utf-8") as f:
                        conversation_data = json.load(f)

                    conversations.append(
                        {
                            "id": conversation_id,
                            "title": conversation_data.get("title", "新对话"),
                            "created_at": conversation_data.get("created_at", ""),
                            "updated_at": conversation_data.get("updated_at", ""),
                            "message_count": len(conversation_data.get("messages", [])),
                        }
                    )
                except Exception as e:
                    print(f"❌ 读取对话文件失败 {filename}: {e}")
                    continue

        # 按更新时间排序（最新的在前）
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations

    except Exception as e:
        print(f"❌ 获取对话列表失败: {e}")
        return []


def format_memoryos_retrieval_result(memoryos_result):
    """格式化MemoryOS检索结果，与evaluate_end_to_end.py保持一致"""
    if not memoryos_result:
        return ""

    formatted_context = ""

    # 短期记忆 (recent conversations)
    if "short_term_queue" in memoryos_result and memoryos_result["short_term_queue"]:
        formatted_context += "SHORT-TERM MEMORY (Recent Interactions):\n"
        for i, item in enumerate(memoryos_result["short_term_queue"], 1):
            user_input = item.get("user_input", "")
            agent_response = item.get("agent_response", "")
            timestamp = item.get("timestamp", "")
            formatted_context += f"{i}. [{timestamp}] User: {user_input}\n"
            formatted_context += f"   Agent: {agent_response}\n\n"

    # 中期记忆 (processed conversations/pages)
    if "mid_term_pages" in memoryos_result and memoryos_result["mid_term_pages"]:
        formatted_context += "MID-TERM MEMORY (Processed Conversations):\n"
        for i, page in enumerate(memoryos_result["mid_term_pages"], 1):
            content = page.get("content", "")
            if content:
                formatted_context += f"{i}. {content}\n\n"

    # 用户长期知识
    if "user_knowledge" in memoryos_result and memoryos_result["user_knowledge"]:
        formatted_context += "LONG-TERM KNOWLEDGE (Personal Insights):\n"
        for i, knowledge in enumerate(memoryos_result["user_knowledge"], 1):
            knowledge_text = knowledge.get("knowledge", "") or knowledge.get(
                "content", ""
            )
            if knowledge_text:
                formatted_context += f"{i}. {knowledge_text}\n\n"

    # 助手长期知识
    if (
        "assistant_knowledge" in memoryos_result
        and memoryos_result["assistant_knowledge"]
    ):
        formatted_context += "ASSISTANT KNOWLEDGE (Domain Expertise):\n"
        for i, knowledge in enumerate(memoryos_result["assistant_knowledge"], 1):
            knowledge_text = knowledge.get("knowledge", "") or knowledge.get(
                "content", ""
            )
            if knowledge_text:
                formatted_context += f"{i}. {knowledge_text}\n\n"

    return formatted_context.strip()


def get_fusion_rag_prompt(
    user_query: str,
    shared_memory_context: str,
    personal_memory_context: str,
    user_profile: str,
) -> str:
    """
    创建融合RAG提示词，结合共享记忆和个人记忆
    与evaluate_end_to_end.py中的get_fusion_rag_prompt保持一致
    """
    return f"""You are a helpful AI assistant. Your task is to answer the user's question based on the provided context from two memory sources.
The context is retrieved from both shared knowledge base and your personal memory of past conversations.
Synthesize the information from both sources to provide a comprehensive and accurate answer.
If the context is not relevant, ignore it and answer based on your own knowledge.

**USER PROFILE:**
---
{user_profile}
---

**CONTEXT FROM SHARED MEMORY:**
---
{shared_memory_context}
---

**CONTEXT FROM PERSONAL MEMORY:**
---
{personal_memory_context}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Your response should be relevant and appropriate for this specific user.
Combine insights from both shared and personal memory to provide the most helpful response.

Your Answer:
"""


def get_rag_answer_prompt(
    user_query: str, retrieved_context: str, user_profile: str
) -> str:
    """
    创建仅使用共享记忆的RAG提示词
    """
    return f"""You are a helpful AI assistant. Your task is to answer the user's question based on the provided context.
The context is retrieved from a shared knowledge base of past conversations.
Synthesize the information from the context to provide a comprehensive and accurate answer.
If the context is not relevant, ignore it and answer based on your own knowledge.

**USER PROFILE:**
---
{user_profile}
---

**CONTEXT FROM SHARED MEMORY:**
---
{retrieved_context}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Your response should be relevant and appropriate for this specific user.

Your Answer:
"""


def get_baseline_answer_prompt(user_query: str, user_profile: str) -> str:
    """
    创建不使用任何记忆的基线提示词
    """
    return f"""You are a helpful AI assistant. Please answer the user's question based on your own knowledge.

**USER PROFILE:**
---
{user_profile}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Your response should be relevant and appropriate for this specific user.

Your Answer:
"""


def get_baseline_answer_prompt_no_profile(
    user_query: str, conversation_context: str = ""
) -> str:
    """
    创建不使用任何记忆和个人信息的基线提示词
    只包含对话上下文（如果有的话）
    """
    context_section = ""
    if conversation_context:
        context_section = f"""
**CURRENT CONVERSATION CONTEXT:**
---
{conversation_context}
---

"""

    return f"""You are a helpful AI assistant. Please answer the user's question based on your own knowledge.
{context_section}**USER'S QUESTION:**
---
{user_query}
---

Please provide a helpful and accurate answer based on your knowledge. Keep your response clear and informative.

IMPORTANT: If the user asks about previous conversation content, you CAN see and reference the conversation history provided above. You should acknowledge what was said previously and provide helpful responses based on that context.

Your Answer:
"""


def generate_response_without_memory(
    user_id: str,
    message: str,
    model: str,
    project_name: str = "default_project",
    conversation_id: str = None,
) -> str:
    """
    无记忆模式：只提供当前对话上下文，不提供任何个人信息、历史记忆或共享记忆
    """
    try:
        # 获取用户配置（仅用于API调用）
        user_config = get_user_config(user_id, project_name)
        if not user_config.get("openai_api_key"):
            return "错误：请先配置OpenAI API Key"

        # 构建对话历史上下文（仅当前对话）
        conversation_context = ""
        if conversation_id:
            conversation_data = load_conversation_history(user_id, conversation_id)
            if conversation_data and conversation_data.get("messages"):
                # 获取历史消息（排除当前消息）
                history_messages = (
                    conversation_data["messages"][:-2]
                    if len(conversation_data["messages"]) >= 2
                    else []
                )
                if history_messages:
                    conversation_context = "\n".join(
                        [
                            f"{'用户' if msg['type'] == 'user' else '助手'}: {msg['content']}"
                            for msg in history_messages[-10:]  # 只取最近10条历史消息
                        ]
                    )
                    print(f"📚 使用最近 {len(history_messages)} 条对话历史作为上下文")
                else:
                    print("ℹ️ 新对话的第一轮交互")

        # 创建无记忆提示词（不包含任何个人信息）
        prompt = get_baseline_answer_prompt_no_profile(message, conversation_context)

        # 调用LLM生成回复
        from openai import OpenAI

        client = OpenAI(
            api_key=user_config.get("openai_api_key", config.openai_api_key),
            base_url=user_config.get("openai_base_url", config.openai_api_base),
            timeout=120.0,  # 增加超时时间到120秒，处理长上下文
            max_retries=2,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10000,
        )

        return response.choices[0].message.content or "抱歉，无法生成回复。"

    except Exception as e:
        print(f"无记忆模式生成回复失败: {e}")
        return f"抱歉，生成回复时出现错误: {str(e)}"


def get_rag_answer_prompt_with_context(
    user_query: str,
    retrieved_context: str,
    user_profile: str,
    conversation_context: str = "",
) -> str:
    """
    创建包含对话上下文的RAG提示词
    """
    context_section = ""
    if conversation_context:
        context_section = f"""
**CURRENT CONVERSATION CONTEXT:**
---
{conversation_context}
---

"""

    return f"""You are a helpful AI assistant. Your task is to answer the user's question based on the provided context.
The context is retrieved from a shared knowledge base of past conversations.
Synthesize the information from the context to provide a comprehensive and accurate answer.
If the context is not relevant, ignore it and answer based on your own knowledge.
{context_section}**USER PROFILE:**
---
{user_profile}
---

**CONTEXT FROM SHARED MEMORY:**
---
{retrieved_context}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Your response should be relevant and appropriate for this specific user.

IMPORTANT: You can see and reference the conversation history provided above. Use it to provide contextually relevant responses.

Your Answer:
"""


def get_fusion_rag_prompt_with_context(
    user_query: str,
    shared_memory_context: str,
    personal_memory_context: str,
    user_profile: str,
    conversation_context: str = "",
) -> str:
    """
    创建包含对话上下文的融合RAG提示词
    """
    context_section = ""
    if conversation_context:
        context_section = f"""
**CURRENT CONVERSATION CONTEXT:**
---
{conversation_context}
---

"""

    return f"""You are a helpful AI assistant. Your task is to answer the user's question based on the provided context from two memory sources.
The context is retrieved from both shared knowledge base and your personal memory of past conversations.
Synthesize the information from both sources to provide a comprehensive and accurate answer.
If the context is not relevant, ignore it and answer based on your own knowledge.
{context_section}**USER PROFILE:**
---
{user_profile}
---

**CONTEXT FROM SHARED MEMORY:**
---
{shared_memory_context}
---

**CONTEXT FROM PERSONAL MEMORY:**
---
{personal_memory_context}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Use both shared knowledge and personal context to provide a response that is relevant and appropriate for this specific user.
If there are conflicts between shared and personal memory, prioritize the information that is most relevant to the user's current question.

IMPORTANT: You can see and reference the conversation history provided above. Use it to provide contextually relevant responses.

Your Answer:
"""


def generate_response_with_memory(
    user_id: str,
    message: str,
    model: str,
    shared_memory_enabled: bool = False,
    personal_memory_enabled: bool = True,
    project_name: str = "default_project",
    conversation_id: str = None,
) -> str:
    """结合个人记忆和共享记忆生成回复，与evaluate_end_to_end.py逻辑保持一致"""
    try:
        # 获取用户配置
        user_config = get_user_config(user_id, project_name)
        if not user_config.get("openai_api_key"):
            return "错误：请先配置OpenAI API Key"

        # 确保用户存在
        user_profile = store.get_user(user_id)
        if not user_profile:
            # 创建默认用户档案
            user_profile = UserProfile(
                user_id=user_id,
                profile_text=user_config.get("user_profile", f"用户 {user_id}"),
            )
            store.add_user(user_profile)

        # 构建对话历史上下文（无论是否开启记忆都要提供）
        conversation_context = ""
        if conversation_id:
            conversation_data = load_conversation_history(user_id, conversation_id)
            if conversation_data and conversation_data.get("messages"):
                # 获取历史消息（排除当前消息）
                history_messages = (
                    conversation_data["messages"][:-2]
                    if len(conversation_data["messages"]) >= 2
                    else []
                )
                if history_messages:
                    conversation_context = "\n".join(
                        [
                            f"{'用户' if msg['type'] == 'user' else '助手'}: {msg['content']}"
                            for msg in history_messages[-10:]  # 只取最近10条历史消息
                        ]
                    )
                    print(f"📚 使用最近 {len(history_messages)} 条对话历史作为上下文")
                else:
                    print("ℹ️ 新对话的第一轮交互")

        # 获取个人记忆和增强用户档案
        personal_memory_context = ""
        enhanced_profile_text = user_profile.profile_text

        if personal_memory_enabled and user_id in memoryos_instances:
            try:
                memoryos_instance = memoryos_instances[user_id]
                # 使用MemoryOS标准检索方法获取所有记忆层 (降低阈值以提高检索成功率)
                memoryos_result = memoryos_instance.retriever.retrieve_context(
                    user_query=message,
                    user_id=user_id,
                    segment_similarity_threshold=0.1,  # 降低中期记忆相似度阈值
                    page_similarity_threshold=0.1,  # 降低页面相似度阈值
                    knowledge_threshold=0.1,  # 降低知识相似度阈值
                    top_k_sessions=3,  # 减少会话数量
                    top_k_knowledge=2,  # 增加知识数量
                )

                # 获取长期用户档案
                long_term_profile = (
                    memoryos_instance.user_long_term_memory.get_raw_user_profile(
                        user_id
                    )
                )
                if long_term_profile and long_term_profile != "None":
                    enhanced_profile_text = f"{user_profile.profile_text}\n\n**Long-term User Profile Insights (from MemoryOS):**\n{long_term_profile}"

                # 添加短期记忆到检索结果中
                context_result = memoryos_result.copy()
                # 获取短期记忆
                short_term_history = memoryos_instance.short_term_memory.get_all()
                if short_term_history:
                    context_result["short_term_queue"] = short_term_history

                # 格式化个人记忆上下文（排除user_knowledge以避免与档案重复）
                context_result.pop("user_knowledge", None)
                personal_memory_context = format_memoryos_retrieval_result(
                    context_result
                )

                print(
                    f"🧠 Retrieved and formatted personal memory for {user_id}: {len(personal_memory_context)} chars"
                )

            except Exception as e:
                print(
                    f"⚠️ Failed to retrieve or process personal memory for {user_id}: {e}"
                )
                personal_memory_context = ""
        elif not personal_memory_enabled:
            print(f"🚫 Personal memory disabled for {user_id}, using baseline mode")

        # 获取共享记忆
        shared_memory_context = ""
        if shared_memory_enabled:
            try:
                # 使用缓存的peers (与原始项目一致)
                peers = retrieve_pipeline.get_cached_peers()

                # 创建增强的用户档案对象用于检索管道
                enhanced_user_profile = UserProfile(
                    user_id=user_id, profile_text=enhanced_profile_text
                )

                # 检索共享记忆
                retrieval_result = retrieve_pipeline.retrieve(
                    user=enhanced_user_profile, task=message, peers=peers, top_k=3
                )

                if retrieval_result["items"]:
                    shared_memory_context = retrieve_pipeline.build_prompt_blocks(
                        retrieval_result["items"]
                    )

                print(
                    f"🔗 Retrieved shared memory context: {len(shared_memory_context)} chars"
                )

            except Exception as e:
                print(f"检索共享记忆失败: {e}")

        # 根据记忆状态选择提示词 (与原始项目逻辑一致)
        if (
            personal_memory_enabled
            and shared_memory_enabled
            and shared_memory_context
            and personal_memory_context
        ):
            # 使用融合RAG提示词 (个人记忆 + 共享记忆)
            prompt = get_fusion_rag_prompt_with_context(
                message,
                shared_memory_context,
                personal_memory_context,
                enhanced_profile_text,
                conversation_context,
            )
            print("🧠 Using Fusion RAG prompt (Personal + Shared Memory)")
        elif personal_memory_enabled and personal_memory_context:
            # 使用个人记忆RAG提示词 (仅个人记忆)
            prompt = get_fusion_rag_prompt_with_context(
                message,
                "",  # 无共享记忆
                personal_memory_context,
                enhanced_profile_text,
                conversation_context,
            )
            print("🧠 Using Personal Memory RAG prompt")
        elif shared_memory_enabled and shared_memory_context:
            # 使用共享记忆RAG提示词 (仅共享记忆)
            prompt = get_rag_answer_prompt_with_context(
                message,
                shared_memory_context,
                enhanced_profile_text,
                conversation_context,
            )
            print("🔗 Using Shared Memory RAG prompt")
        else:
            # 使用基线提示词 (无记忆) - 不包含用户档案信息
            prompt = get_baseline_answer_prompt_no_profile(
                message, conversation_context
            )
            print("📝 Using Baseline prompt (No Memory, No Profile)")

        # 对话上下文已经在相应的提示词函数中处理了，这里不需要额外添加

        # 调用LLM生成回复
        from openai import OpenAI

        client = OpenAI(
            api_key=user_config.get("openai_api_key", config.openai_api_key),
            base_url=user_config.get("openai_base_url", config.openai_api_base),
            timeout=120.0,  # 增加超时时间到120秒，处理长上下文
            max_retries=2,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10000,
        )

        return response.choices[0].message.content or "抱歉，我无法生成回复。"

    except Exception as e:
        print(f"生成回复失败: {e}")
        return f"错误：生成回复时出现问题 - {str(e)}"


# API路由
@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """记忆仪表盘页面"""
    return render_template("dashboard.html")


@app.route("/api/get_shared_memories", methods=["POST"])
def get_shared_memories():
    """获取共享记忆API"""
    try:
        data = request.get_json()
        username = data.get("username")
        limit = data.get("limit", 50)

        print("\n📊 获取共享记忆请求:")
        print(f"  - 用户名: {username}")
        print(f"  - 限制数量: {limit}")

        if not username:
            return jsonify({"success": False, "error": "缺少用户名"})

        # 获取所有共享记忆
        all_memories = store.list_memories()
        print(f"  - 总记忆数量: {len(all_memories)}")

        # 转换为字典格式
        memories_list = []
        for i, mem in enumerate(all_memories[:limit]):
            try:
                # 将时间戳转换为可读格式
                timestamp_str = (
                    datetime.fromtimestamp(mem.created_at).strftime("%Y-%m-%d %H:%M:%S")
                    if mem.created_at
                    else "未知时间"
                )

                # 安全地获取内容
                content = ""
                if hasattr(mem, "cot_text") and mem.cot_text and mem.cot_text.strip():
                    content = mem.cot_text.strip()
                elif hasattr(mem, "raw_text") and mem.raw_text and mem.raw_text.strip():
                    content = mem.raw_text.strip()
                else:
                    content = "无内容"

                memory_data = {
                    "id": mem.id,
                    "user_id": mem.source_user_id,
                    "content": content,
                    "timestamp": timestamp_str,
                    "source": mem.meta.get("source", "conversation")
                    if hasattr(mem, "meta") and mem.meta
                    else "conversation",
                }
                memories_list.append(memory_data)

                if i < 3:  # 打印前3条记忆的详细信息用于调试
                    print(
                        f"  - 记忆 {i + 1}: ID={mem.id}, 用户={mem.source_user_id}, 内容长度={len(memory_data['content'])}"
                    )

            except Exception as mem_error:
                print(f"  - 处理记忆 {i} 失败: {mem_error}")
                continue

        print(f"  - 成功处理记忆数量: {len(memories_list)}")

        return jsonify(
            {"success": True, "memories": memories_list, "total": len(all_memories)}
        )

    except Exception as e:
        print(f"❌ 获取共享记忆失败: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/get_memory_file", methods=["GET"])
def get_memory_file():
    """获取用户的记忆文件（短期、中期、长期）"""
    try:
        username = request.args.get("username")
        file_name = request.args.get("file")

        if not username or not file_name:
            return jsonify({"success": False, "error": "缺少参数"}), 400

        # 检查文件名是否合法
        allowed_files = ["short_term.json", "mid_term.json", "long_term_user.json"]
        if file_name not in allowed_files:
            return jsonify({"success": False, "error": "非法的文件名"}), 400

        # 构建文件路径：eval/memoryos_data/{username}/users/{username}/{file}
        file_path = os.path.join(
            MEMORYOS_DATA_DIR, username, "users", username, file_name
        )

        print(f"尝试读取记忆文件: {file_path}")

        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return jsonify({"success": False, "error": "文件不存在"}), 404

        # 读取文件
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return jsonify(data)

    except Exception as e:
        print(f"读取记忆文件失败: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat_direct", methods=["POST"])
def chat_direct():
    """流式聊天API - 使用Server-Sent Events"""
    from flask import stream_with_context

    # 在流式上下文外读取请求数据
    data = request.get_json()
    username = data.get("username")
    message = data.get("message")
    model = data.get("model", "gpt-4o-mini")
    conversation_id = data.get("conversation_id")
    shared_memory_enabled = data.get("shared_memory_enabled", False)
    personal_memory_enabled = data.get("personal_memory_enabled", True)
    project_name = data.get("project_name", "default_project")

    def generate():
        try:
            if not username or not message:
                yield f"data: {json.dumps({'error': '缺少必要参数'}, ensure_ascii=False)}\n\n"
                return

            # 确保用户有MemoryOS实例
            print(f"\n{'=' * 60}")
            print(f"📝 [流式] 开始处理用户 {username} 的消息")
            print(
                f"🔘 个人记忆: {personal_memory_enabled}, 共享记忆: {shared_memory_enabled}"
            )

            memoryos_instance = ensure_user_memoryos(username, project_name)
            if not memoryos_instance and username in memoryos_instances:
                del memoryos_instances[username]
                memoryos_instance = ensure_user_memoryos(username, project_name)

            # 获取用户配置
            user_config = get_user_config(username, project_name)
            if not user_config.get("openai_api_key"):
                yield f"data: {json.dumps({'error': '请先配置OpenAI API Key'}, ensure_ascii=False)}\n\n"
                return

            # 使用局部变量来避免作用域问题
            current_conversation_id = conversation_id

            # 构建对话历史上下文
            conversation_context = ""
            if current_conversation_id:
                conversation_data = load_conversation_history(
                    username, current_conversation_id
                )
                if conversation_data and conversation_data.get("messages"):
                    history_messages = conversation_data["messages"]
                    if history_messages:
                        conversation_context = "\n".join(
                            [
                                f"{'用户' if msg['type'] == 'user' else '助手'}: {msg['content']}"
                                for msg in history_messages[-10:]
                            ]
                        )

            # 获取用户档案
            user_profile = store.get_user(username)
            if not user_profile:
                user_profile = UserProfile(
                    user_id=username,
                    profile_text=user_config.get("user_profile", f"用户 {username}"),
                )
                store.add_user(user_profile)

            # 获取个人记忆和共享记忆
            personal_memory_context = ""
            enhanced_profile_text = user_profile.profile_text

            if personal_memory_enabled and username in memoryos_instances:
                try:
                    memoryos_instance = memoryos_instances[username]
                    memoryos_result = memoryos_instance.retriever.retrieve_context(
                        user_query=message,
                        user_id=username,
                        segment_similarity_threshold=0.1,
                        page_similarity_threshold=0.1,
                        knowledge_threshold=0.1,
                        top_k_sessions=3,
                        top_k_knowledge=2,
                    )

                    long_term_profile = (
                        memoryos_instance.user_long_term_memory.get_raw_user_profile(
                            username
                        )
                    )
                    if long_term_profile and long_term_profile != "None":
                        enhanced_profile_text = f"{user_profile.profile_text}\n\n**Long-term User Profile Insights:**\n{long_term_profile}"

                    context_result = memoryos_result.copy()
                    short_term_history = memoryos_instance.short_term_memory.get_all()
                    if short_term_history:
                        context_result["short_term_queue"] = short_term_history

                    context_result.pop("user_knowledge", None)
                    personal_memory_context = format_memoryos_retrieval_result(
                        context_result
                    )
                except Exception as e:
                    print(f"⚠️ 获取个人记忆失败: {e}")

            shared_memory_context = ""
            if shared_memory_enabled:
                try:
                    peers = retrieve_pipeline.get_cached_peers()
                    enhanced_user_profile = UserProfile(
                        user_id=username, profile_text=enhanced_profile_text
                    )
                    retrieval_result = retrieve_pipeline.retrieve(
                        user=enhanced_user_profile, task=message, peers=peers, top_k=3
                    )
                    if retrieval_result["items"]:
                        shared_memory_context = retrieve_pipeline.build_prompt_blocks(
                            retrieval_result["items"]
                        )
                except Exception as e:
                    print(f"⚠️ 获取共享记忆失败: {e}")

            # 构建提示词
            if (
                personal_memory_enabled
                and shared_memory_enabled
                and shared_memory_context
                and personal_memory_context
            ):
                prompt = get_fusion_rag_prompt_with_context(
                    message,
                    shared_memory_context,
                    personal_memory_context,
                    enhanced_profile_text,
                    conversation_context,
                )
                print("🧠 使用融合RAG提示词")
            elif personal_memory_enabled and personal_memory_context:
                prompt = get_fusion_rag_prompt_with_context(
                    message,
                    "",
                    personal_memory_context,
                    enhanced_profile_text,
                    conversation_context,
                )
                print("🧠 使用个人记忆RAG提示词")
            elif shared_memory_enabled and shared_memory_context:
                prompt = get_rag_answer_prompt_with_context(
                    message,
                    shared_memory_context,
                    enhanced_profile_text,
                    conversation_context,
                )
                print("🔗 使用共享记忆RAG提示词")
            else:
                prompt = get_baseline_answer_prompt_no_profile(
                    message, conversation_context
                )
                print("📝 使用基线提示词")

            # 🚀 立即保存用户消息，确保对话文件存在，避免切换时的"对话不存在"错误
            try:
                temp_conversation_id = save_chat_conversation(
                    username,
                    current_conversation_id,
                    message,
                    "",
                    model,
                    shared_memory_enabled,
                    personal_memory_enabled,
                    user_message_only=True,
                )
                print(f"✅ 已立即保存用户消息，对话ID: {temp_conversation_id}")
                if temp_conversation_id:
                    current_conversation_id = temp_conversation_id
            except Exception as e:
                print(f"⚠️ 立即保存用户消息失败: {e}")

            # 🔥 流式调用OpenAI API
            from openai import OpenAI

            client = OpenAI(
                api_key=user_config.get("openai_api_key", config.openai_api_key),
                base_url=user_config.get("openai_base_url", config.openai_api_base),
                timeout=120.0,
                max_retries=2,
            )

            # 🔥 创建可中断的流式调用
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=10000,
                    stream=True,  # 🔥 启用流式输出
                )
            except Exception as e:
                yield f"data: {json.dumps({'error': f'创建流式调用失败: {str(e)}'}, ensure_ascii=False)}\n\n"
                return

            # 收集完整回复
            full_response = ""
            stream_interrupted = False
            chunk_count = 0  # 用于定期保存

            # 定义一个保存函数，用于在任何情况下保存对话
            def save_interrupted_conversation():
                """保存被中断的对话"""
                # 即使没有AI回复，也要保存用户的消息
                response_to_save = (
                    full_response if full_response.strip() else "（回复被用户终止）"
                )
                print(
                    f"💾 保存被中断的对话，用户消息: {message[:50]}...，AI回复长度: {len(full_response)} 字符"
                )

                try:
                    # 保存到个人记忆（即使AI回复为空也要保存用户消息）
                    if username in memoryos_instances:
                        memoryos_instance = memoryos_instances[username]
                        memoryos_instance.add_memory(message, response_to_save)
                        print("✅ 中断的对话已保存到短期记忆")

                    # 保存对话到文件 - 更新最后一条AI消息
                    saved_conversation_id = save_chat_conversation(
                        username,
                        current_conversation_id,
                        message,
                        response_to_save,
                        model,
                        shared_memory_enabled,
                        personal_memory_enabled,
                        update_last_ai_message=True,
                    )

                    # 检测思维链断裂（只有AI有回复时才检测）
                    if shared_memory_enabled and full_response.strip():
                        try:
                            check_and_store_chain_break(
                                username, message, full_response
                            )
                        except Exception as e:
                            print(f"⚠️ 思维链检测失败: {e}")

                    print(
                        f"✅ 中断的消息已正常保存，conversation_id: {saved_conversation_id}"
                    )
                    return saved_conversation_id
                except Exception as e:
                    print(f"❌ 保存中断消息失败: {e}")
                    import traceback

                    traceback.print_exc()
                    return None

            try:
                # 逐块发送数据
                for chunk in stream:
                    try:
                        # 检查 choices 是否为空
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                content = delta.content
                                full_response += content
                                chunk_count += 1

                                # 🔄 定期保存AI回复内容（每10个chunk保存一次），确保切换时能看到已生成的内容
                                if chunk_count % 10 == 0:
                                    try:
                                        save_chat_conversation(
                                            username,
                                            current_conversation_id,
                                            message,
                                            full_response,
                                            model,
                                            shared_memory_enabled,
                                            personal_memory_enabled,
                                            update_last_ai_message=True,
                                        )
                                        # print(f"🔄 已保存AI回复片段，长度: {len(full_response)} 字符")
                                    except Exception as e:
                                        print(f"⚠️ 保存AI回复片段失败: {e}")

                                # 发送SSE格式数据
                                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                    except (GeneratorExit, StopIteration) as e:
                        # 客户端断开连接
                        print(f"🛑 检测到客户端断开连接: {e}")
                        stream_interrupted = True
                        break
                    except Exception as e:
                        print(f"⚠️ 处理流式数据时出错: {e}")
                        continue

            except GeneratorExit:
                # 客户端主动断开连接（AbortController触发）
                print("🛑 客户端主动终止连接（AbortController） - 开始保存操作")
                stream_interrupted = True

                # 尝试关闭 OpenAI 流
                try:
                    if hasattr(stream, "close"):
                        stream.close()
                    print("🔒 OpenAI 流已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭 OpenAI 流时出错: {e}")

                # 🚀 立即保存被中断的内容（在yield之前）
                try:
                    saved_conversation_id = save_interrupted_conversation()
                    print(f"💾 保存操作完成，结果: {saved_conversation_id}")
                except Exception as e:
                    print(f"❌ 保存操作出现异常: {e}")
                    saved_conversation_id = None

                # 尝试发送终止信号（如果可能的话）
                try:
                    yield f"data: {json.dumps({'done': True, 'conversation_id': saved_conversation_id}, ensure_ascii=False)}\n\n"
                    print("📤 终止信号发送成功")
                except Exception as e:
                    # 如果无法发送，也没关系，因为客户端已经断开
                    print(f"⚠️ 无法发送终止信号: {e}")

                print("🛑 GeneratorExit 异常处理完成")
                return

            except Exception as e:
                print(f"❌ 流式处理出现异常: {e}")
                yield f"data: {json.dumps({'error': f'流式处理异常: {str(e)}'}, ensure_ascii=False)}\n\n"
                return

            # 如果被中断，保存中断的内容然后返回
            if stream_interrupted:
                print("🛑 流式输出被中断，尝试保存已生成的内容")
                saved_conversation_id = save_interrupted_conversation()
                try:
                    yield f"data: {json.dumps({'done': True, 'conversation_id': saved_conversation_id}, ensure_ascii=False)}\n\n"
                except:
                    pass
                return

            # 发送结束标记
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

            print(f"✅ 流式输出完成，总长度: {len(full_response)} 字符")

            # 只有在正常完成时才保存记忆和对话（不包含被中断的情况）
            if full_response.strip():  # 确保有内容才保存
                # 保存到个人记忆
                if username in memoryos_instances:
                    try:
                        memoryos_instance = memoryos_instances[username]
                        memoryos_instance.add_memory(message, full_response)
                        print("✅ 对话已保存到短期记忆")
                    except Exception as e:
                        print(f"❌ 保存记忆失败: {e}")

                # 保存对话 - 更新最后一条AI消息
                saved_conversation_id = save_chat_conversation(
                    username,
                    current_conversation_id,
                    message,
                    full_response,
                    model,
                    shared_memory_enabled,
                    personal_memory_enabled,
                    update_last_ai_message=True,
                )

                # 检测思维链断裂
                if shared_memory_enabled:
                    try:
                        check_and_store_chain_break(username, message, full_response)
                    except Exception as e:
                        print(f"⚠️ 思维链检测失败: {e}")

                # 发送对话ID
                yield f"data: {json.dumps({'conversation_id': saved_conversation_id or current_conversation_id}, ensure_ascii=False)}\n\n"
            else:
                print("⚠️ 流式输出为空，跳过保存操作")

        except Exception as e:
            print(f"❌ 流式生成失败: {e}")
            import traceback

            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/get_chat_conversations", methods=["POST"])
def get_chat_conversations_api():
    """获取聊天对话列表"""
    try:
        data = request.get_json()
        username = data.get("username")
        project_name = data.get("project_name", "default_project")

        if not username:
            return jsonify({"success": False, "error": "缺少用户名"})

        conversations = get_chat_conversations(username, project_name)
        return jsonify({"success": True, "conversations": conversations})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_chat_messages", methods=["POST"])
def get_chat_messages_api():
    """获取指定对话的消息"""
    try:
        data = request.get_json()
        username = data.get("username")
        conversation_id = data.get("conversation_id")
        project_name = data.get("project_name", "default_project")

        if not username or not conversation_id:
            return jsonify({"success": False, "error": "缺少必要参数"})

        conversation = load_conversation_history(username, conversation_id)
        if not conversation:
            return jsonify({"success": False, "error": "对话不存在"})

        return jsonify({"success": True, "conversation": conversation})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/save_chat_user_config", methods=["POST"])
def save_chat_user_config():
    """保存用户配置"""
    try:
        data = request.get_json()
        username = data.get("username")
        project_name = data.get("project_name", "default_project")

        if not username:
            return jsonify({"success": False, "error": "缺少用户名"})

        config_data = {
            "openai_api_key": data.get("openai_api_key", ""),
            "openai_base_url": data.get("openai_base_url", "https://api.openai.com/v1"),
            "user_profile": data.get("user_profile", f"用户 {username}"),
        }

        if save_user_config(username, config_data, project_name):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "保存配置失败"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/login", methods=["POST"])
def login():
    """用户登录验证API"""
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"success": False, "error": "用户名和密码不能为空"})

        # 读取用户配置文件
        user_file_path = os.path.join(project_root, "user.json")
        if not os.path.exists(user_file_path):
            return jsonify({"success": False, "error": "用户配置文件不存在"})

        with open(user_file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        # 验证用户名和密码
        users = user_data.get("users", [])
        for user in users:
            if user.get("username") == username and user.get("password") == password:
                return jsonify(
                    {"success": True, "message": "登录成功", "username": username}
                )

        return jsonify({"success": False, "error": "用户名或密码错误"})

    except Exception as e:
        print(f"登录验证失败: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/chat/users/<username>/<filename>")
@app.route("/chat/<project_name>/users/<username>/<filename>")
def serve_user_file(username, filename, project_name="default_project"):
    """提供用户文件服务"""
    user_dir = os.path.join(MEMORYOS_DATA_DIR, project_name, "users", username)
    if os.path.exists(os.path.join(user_dir, filename)):
        return send_from_directory(user_dir, filename)
    else:
        return jsonify({"error": "文件不存在"}), 404


if __name__ == "__main__":
    print("🚀 启动Flask应用...")
    print(f"📁 数据目录: {MEMORYOS_DATA_DIR}")
    print("🌐 访问地址: http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, debug=True)
