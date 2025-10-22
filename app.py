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
# =============================
# 用户画像维度提取与同步（统一到 users.json）
import re

DIMENSION_GROUPS_CN = {
    "psych": "心理模型",
    "align": "AI对齐维度",
    "interest": "内容兴趣标签",
}

DIMENSION_MAP_EN_TO_CN: dict[str, tuple[str, str]] = {
    # 心理模型（部分，覆盖 prompts 中定义）
    "Extraversion": ("外向性", "psych"),
    "Openness": ("开放性", "psych"),
    "Agreeableness": ("宜人性", "psych"),
    "Conscientiousness": ("尽责性", "psych"),
    "Neuroticism": ("情绪稳定性", "psych"),
    "Physiological Needs": ("生理需求", "psych"),
    "Need for Security": ("安全需求", "psych"),
    "Need for Belonging": ("归属需求", "psych"),
    "Need for Self-Esteem": ("自尊需求", "psych"),
    "Cognitive Needs": ("认知需求", "psych"),
    "Aesthetic Appreciation": ("审美欣赏", "psych"),
    "Self-Actualization": ("自我实现", "psych"),
    "Need for Order": ("秩序需求", "psych"),
    "Need for Autonomy": ("自主需求", "psych"),
    "Need for Power": ("权力需求", "psych"),
    "Need for Achievement": ("成就需求", "psych"),
    # AI 对齐维度
    "Helpfulness": ("帮助性", "align"),
    "Honesty": ("诚实性", "align"),
    "Safety": ("安全性", "align"),
    "Instruction Compliance": ("指令遵从", "align"),
    "Truthfulness": ("真实度", "align"),
    "Coherence": ("连贯性", "align"),
    "Complexity": ("复杂度偏好", "align"),
    "Conciseness": ("简洁性", "align"),
    # 内容兴趣标签
    "Science Interest": ("科学兴趣", "interest"),
    "Education Interest": ("教育兴趣", "interest"),
    "Psychology Interest": ("心理学兴趣", "interest"),
    "Family Concern": ("家庭关切", "interest"),
    "Fashion Interest": ("时尚兴趣", "interest"),
    "Art Interest": ("艺术兴趣", "interest"),
    "Health Concern": ("健康关切", "interest"),
    "Financial Management Interest": ("理财兴趣", "interest"),
    "Sports Interest": ("运动兴趣", "interest"),
    "Food Interest": ("美食兴趣", "interest"),
    "Travel Interest": ("旅行兴趣", "interest"),
    "Music Interest": ("音乐兴趣", "interest"),
    "Literature Interest": ("文学兴趣", "interest"),
    "Film Interest": ("电影兴趣", "interest"),
    "Social Media Activity": ("社交媒体活跃", "interest"),
    "Tech Interest": ("科技兴趣", "interest"),
    "Environmental Concern": ("环境关切", "interest"),
    "History Interest": ("历史兴趣", "interest"),
    "Political Concern": ("政治关切", "interest"),
    "Religious Interest": ("宗教兴趣", "interest"),
    "Gaming Interest": ("游戏兴趣", "interest"),
    "Animal Concern": ("动物关切", "interest"),
    "Emotional Expression": ("情绪表达", "interest"),
    "Sense of Humor": ("幽默风格", "interest"),
    "Information Density": ("信息密度偏好", "interest"),
    "Language Style": ("语言风格", "interest"),
    "Practicality": ("实用性偏好", "interest"),
}

LEVEL_MAP_EN_TO_CN = {
    "High": "高",
    "Medium": "中",
    "Low": "低",
    # 风格取值（当维度不是高/中/低时，原样或映射）
    "Formal": "正式",
    "Informal": "口语",
    "Restrained": "克制",
    "Expressive": "外露",
    "Detailed": "详细",
    "Concise": "简洁",
}

def extract_profile_dimensions_from_text(profile_text: str) -> dict:
    """从 MemoryOS 长期画像文本中提取维度 -> { 大维度中文: { 小维度中文: 等级中文 } }。
    仅解析能识别到的维度；未命中不返回。
    """
    grouped = {v: {} for v in DIMENSION_GROUPS_CN.values()}
    if not profile_text:
        return grouped
    # 支持多种描述：Level / Preference Level / Expectation Level 等
    pattern = re.compile(r"- \*\*\s*([^（(\*:]+?)\s*[（(][^:：]*?:\s*([^）)]+)[)）]\*\*\s*[:：]?")
    for m in pattern.finditer(profile_text):
        en_name = m.group(1).strip()
        raw_level = m.group(2).strip()
        level_cn = LEVEL_MAP_EN_TO_CN.get(raw_level, raw_level)
        mapped = DIMENSION_MAP_EN_TO_CN.get(en_name)
        if not mapped:
            continue
        dim_cn, group_key = mapped
        group_cn = DIMENSION_GROUPS_CN.get(group_key, group_key)
        grouped.setdefault(group_cn, {})
        grouped[group_cn][dim_cn] = level_cn
    return grouped

def sync_user_dimensions_to_store(user_id: str, profile_text: str) -> None:
    try:
        grouped = extract_profile_dimensions_from_text(profile_text)
        # 读取现有用户，保持 profile_text
        user_profile = store.get_user(user_id)
        profile_text_to_keep = user_profile.profile_text if user_profile else f"用户 {user_id}"
        updated = UserProfile(user_id=user_id, profile_text=profile_text_to_keep, profile_dimensions=grouped)
        store.add_user(updated)
        print(f"✅ 已同步结构化用户画像维度到 users.json -> {user_id}")
    except Exception as e:
        print(f"⚠️ 同步用户画像维度失败: {e}")

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


def get_page_from_mid_term(memoryos_instance, page_id: str) -> Optional[Dict]:
    """从中期记忆中根据page_id获取页面"""
    if not page_id:
        return None
    
    try:
        mid_term = memoryos_instance.mid_term_memory
        for session_id, session in mid_term.sessions.items():
            for page in session.get("details", []):
                if page.get("page_id") == page_id:
                    return page
        return None
    except Exception as e:
        print(f"⚠️ 从中期记忆查找页面失败: {e}")
        return None


def trace_complete_chain(memoryos_instance, start_qa_list: List[Dict]) -> List[Dict]:
    """追溯完整的对话链
    
    从短期记忆的QA列表开始，向前追溯pre_page链接，找到中期记忆里的所有相关页面。
    返回完整的链（从最早到最晚）。
    
    现在短期记忆的QA也包含page_id和pre_page，可以直接追溯。
    """
    if not start_qa_list:
        return []
    
    complete_chain = []
    
    try:
        # 从短期记忆的第一条开始追溯
        first_qa = start_qa_list[0]
        current_pre_page_id = first_qa.get("pre_page")
        
        if not current_pre_page_id:
            print("📍 短期记忆第一条无pre_page链接，这是对话链的起点")
        else:
            # 有pre_page，向前追溯中期记忆
            print(f"🔍 开始追溯pre_page链接: {current_pre_page_id}")
            visited = set()
            mid_term_count = 0
            
            while current_pre_page_id and current_pre_page_id not in visited:
                visited.add(current_pre_page_id)
                page = get_page_from_mid_term(memoryos_instance, current_pre_page_id)
                
                if page:
                    # 转换为QA格式并添加到链的开头
                    qa = {
                        "user_input": page.get("user_input", ""),
                        "agent_response": page.get("agent_response", ""),
                        "timestamp": page.get("timestamp", ""),
                        "page_id": page.get("page_id"),
                        "pre_page": page.get("pre_page")
                    }
                    complete_chain.insert(0, qa)  # 插入到最前面
                    mid_term_count += 1
                    current_pre_page_id = page.get("pre_page")
                    
                    if not current_pre_page_id:
                        print(f"  ↳ 找到对话链起点（共追溯 {mid_term_count} 条中期记忆）")
                    elif mid_term_count % 5 == 0:  # 每5条打印一次进度
                        print(f"  ↳ 已追溯 {mid_term_count} 条...")
                else:
                    print(f"  ✗ 页面 {current_pre_page_id} 未在中期记忆找到，停止追溯（已追溯 {mid_term_count} 条）")
                    break
        
        # 添加短期记忆的内容
        complete_chain.extend(start_qa_list)
        
        mid_count = len(complete_chain) - len(start_qa_list)
        print(f"🔗 完整链追溯完成: 共 {len(complete_chain)} 条（中期: {mid_count}, 短期: {len(start_qa_list)}）")
        
    except Exception as e:
        print(f"⚠️ 追溯完整链失败: {e}")
        import traceback
        traceback.print_exc()
        # 失败时返回原始短期记忆内容
        return start_qa_list
    
    return complete_chain


def check_and_store_chain_break_from_memoryos(
    user_id: str, memoryos_instance
) -> None:
    """从MemoryOS短期记忆检测思维链断裂并存储到共享记忆
    
    在每次add_memory后调用，检测短期记忆中最后两条的连续性。
    如果断链，追溯完整的对话链（包括中期记忆），并发送到共享记忆。
    """
    if not memoryos_instance:
        return

    try:
        # 从MemoryOS短期记忆读取所有QA对
        short_term_qa_list = memoryos_instance.short_term_memory.get_all()
        
        if len(short_term_qa_list) < 2:
            # 少于2条，无需检测连续性
            return

        # 检测最后两条的连续性
        last_qa = short_term_qa_list[-1]
        second_last_qa = short_term_qa_list[-2]
        
        # 转换为page格式
        previous_page = {
            "user_input": second_last_qa.get("user_input", ""),
            "agent_response": second_last_qa.get("agent_response", ""),
            "timestamp": second_last_qa.get("timestamp", "")
        }
        current_page = {
            "user_input": last_qa.get("user_input", ""),
            "agent_response": last_qa.get("agent_response", ""),
            "timestamp": last_qa.get("timestamp", "")
        }
        
        # 检测对话连续性
        is_continuous = check_conversation_continuity(
            previous_page, current_page, memoryos_instance.client, model=config.llm_model_name
        )

        if not is_continuous:
            # 思维链断裂！追溯完整链并发送到共享记忆
            short_term_broken = short_term_qa_list[:-1]  # 除了最后一条
            
            # 追溯完整的对话链（包括中期记忆）
            complete_chain = trace_complete_chain(memoryos_instance, short_term_broken)
            
            print(f"💡 检测到用户 {user_id} 的思维链断裂！完整对话链长度: {len(complete_chain)}")
            
            # 转换为page格式并发送到共享记忆
            chain_pages = [
                {
                    "user_input": qa.get("user_input", ""),
                    "agent_response": qa.get("agent_response", ""),
                    "timestamp": qa.get("timestamp", "")
                }
                for qa in complete_chain
            ]
            save_chain_to_shared_memory(user_id, chain_pages)
            
            # 🔪 断开链接：将最后一条（新话题开头）的 pre_page 置空
            old_pre_page_id = last_qa.get("pre_page")
            last_qa["pre_page"] = None
            
            # 同时将倒数第二条的 next_page 置空（可能在短期或中期）
            second_last_page_id = second_last_qa.get("page_id")
            if second_last_page_id:
                # 先尝试在短期记忆中更新
                second_last_qa["next_page"] = None
                
                # 如果倒数第二条已经在中期记忆，也需要更新
                mid_page = get_page_from_mid_term(memoryos_instance, second_last_page_id)
                if mid_page:
                    mid_page["next_page"] = None
                    memoryos_instance.mid_term_memory.save()
            
            # 保存短期记忆以持久化链接断开
            memoryos_instance.short_term_memory.save()
            
            print(f"✂️ 已断开对话链链接（pre_page: {old_pre_page_id} → None，开始新链）")
            print(f"📤 完整对话链已发送到共享记忆")
        else:
            # 计算当前完整对话链长度（包括中期）
            first_qa = short_term_qa_list[0]
            current_chain = trace_complete_chain(memoryos_instance, short_term_qa_list)
            print(f"✅ 对话连续，完整对话链长度: {len(current_chain)}")
            
    except Exception as e:
        print(f"⚠️ 从MemoryOS检测思维链断裂失败: {e}")
        import traceback
        traceback.print_exc()


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
                embedding_model_name="/root/autodl-tmp/embedding_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
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
        messages = conversation_data["messages"]
        
        # 检查最后一条消息是否是用户消息
        # 如果最后一条是用户消息，说明这是新的一轮对话，应该添加新的AI回复
        # 如果最后一条是AI消息，说明正在更新当前这轮的AI回复（流式输出中的增量更新）
        if messages[-1]["type"] == "user":
            # 最后一条是用户消息，添加新的AI回复
            new_ai_message = {
                "type": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "shared_memory_enabled": shared_memory_enabled
            }
            conversation_data["messages"].append(new_ai_message)
        elif messages[-1]["type"] == "assistant":
            # 最后一条是AI消息，更新它（流式输出的增量更新）
            messages[-1]["content"] = ai_response
            messages[-1]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            messages[-1]["shared_memory_enabled"] = shared_memory_enabled
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
                "shared_memory_enabled": shared_memory_enabled
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

    return f"""
{context_section}**USER'S QUESTION:**
---
{user_query}
---
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
                    # 同步中文键值画像维度至 users.json
                    sync_user_dimensions_to_store(user_id, long_term_profile)

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

                # 打印最终选中的共享记忆ID（在构建提示词前）
                try:
                    selected_ids = [
                        it.get("memory", {}).get("id", "NO_ID_FOUND")
                        for it in retrieval_result.get("items", [])
                        if isinstance(it, dict)
                    ]
                    if selected_ids:
                        print(
                            f"✅ 共享记忆已选中ID: {', '.join(selected_ids)}"
                        )
                    else:
                        print(
                            "ℹ️ 共享记忆未选中任何条目（为空或被QC过滤）"
                        )
                except Exception as log_err:
                    print(f"⚠️ 打印共享记忆ID失败: {log_err}")

                if retrieval_result["items"]:
                    shared_memory_context = retrieve_pipeline.build_prompt_blocks(
                        retrieval_result["items"]
                    )

                print(
                    f"🔗 Retrieved shared memory context: {len(shared_memory_context)} chars"
                )

            except Exception as e:
                print(f"检索共享记忆失败: {e}")

        else:
            print("ℹ️ 共享记忆未开启（shared_memory_enabled=False）")

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
        limit = data.get("limit", 10000)  # 默认限制改为10000，可以获取所有共享记忆

        print("\n📊 获取共享记忆请求:")
        print(f"  - 用户名: {username}")
        print(f"  - 限制数量: {limit}")

        if not username:
            return jsonify({"success": False, "error": "缺少用户名"})

        # 获取所有共享记忆
        all_memories = store.list_memories()
        print(f"  - 总记忆数量: {len(all_memories)}")

        # 按照创建时间从新到旧排序
        all_memories_sorted = sorted(
            all_memories, 
            key=lambda mem: mem.created_at if mem.created_at else 0, 
            reverse=True  # 降序排列，最新的在前面
        )
        print(f"  - 已按时间倒序排序")

        # 转换为字典格式
        memories_list = []
        for i, mem in enumerate(all_memories_sorted[:limit]):
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

                # 获取focus_query
                focus_query = ""
                if hasattr(mem, "meta") and mem.meta:
                    focus_query = mem.meta.get("focus_query", "")
                
                memory_data = {
                    "id": mem.id,
                    "user_id": mem.source_user_id,
                    "content": content,
                    "timestamp": timestamp_str,
                    "created_at": mem.created_at,  # 添加原始时间戳用于调试
                    "source": mem.meta.get("source", "conversation")
                    if hasattr(mem, "meta") and mem.meta
                    else "conversation",
                    "focus_query": focus_query
                }
                memories_list.append(memory_data)

                if i < 3:  # 打印前3条记忆的详细信息用于调试
                    print(
                        f"  - 记忆 {i + 1}: ID={mem.id}, 用户={mem.source_user_id}, 时间={timestamp_str}, 内容长度={len(memory_data['content'])}"
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


@app.route("/api/get_user_dimensions", methods=["GET"])
def get_user_dimensions():
    """获取统一后的结构化用户画像维度（按三大类分组，仅显示已存在的小维度）。"""
    try:
        username = request.args.get("username")
        if not username:
            return jsonify({"success": False, "error": "缺少用户名"}), 400

        user_profile = store.get_user(username)
        grouped = None
        if user_profile and getattr(user_profile, "profile_dimensions", None):
            grouped = user_profile.profile_dimensions
        else:
            # fallback: 从长期画像文本即时解析
            user_dir = os.path.join(MEMORYOS_DATA_DIR, username, "users", username)
            ltm_path = os.path.join(user_dir, "long_term_user.json")
            if os.path.exists(ltm_path):
                with open(ltm_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                user_profiles = data.get("user_profiles", {})
                profile = user_profiles.get(username, {})
                ltm_text = profile.get("data", "")
                grouped = extract_profile_dimensions_from_text(ltm_text)
            else:
                grouped = {v: {} for v in DIMENSION_GROUPS_CN.values()}

        return jsonify({"success": True, "dimensions": grouped, "groups": DIMENSION_GROUPS_CN})
    except Exception as e:
        print(f"获取用户画像维度失败: {e}")
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
                        # 同步中文键值画像维度至 users.json
                        sync_user_dimensions_to_store(username, long_term_profile)

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
                    # 打印最终选中的共享记忆ID（在构建提示词前）
                    try:
                        selected_ids = [
                            it.get("memory", {}).get("id", "NO_ID_FOUND")
                            for it in retrieval_result.get("items", [])
                            if isinstance(it, dict)
                        ]
                        if selected_ids:
                            print(
                                f"✅ 共享记忆已选中ID: {', '.join(selected_ids)}"
                            )
                        else:
                            print(
                                "ℹ️ 共享记忆未选中任何条目（为空或被QC过滤）"
                            )
                    except Exception as log_err:
                        print(f"⚠️ 打印共享记忆ID失败: {log_err}")

                    if retrieval_result["items"]:
                        shared_memory_context = retrieve_pipeline.build_prompt_blocks(
                            retrieval_result["items"]
                        )
                except Exception as e:
                    print(f"⚠️ 获取共享记忆失败: {e}")
            else:
                print("ℹ️ 共享记忆未开启（shared_memory_enabled=False）")

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
                        
                        # 检测思维链断裂并发送到共享记忆
                        if shared_memory_enabled and full_response.strip():
                            try:
                                check_and_store_chain_break_from_memoryos(username, memoryos_instance)
                            except Exception as e:
                                print(f"⚠️ 思维链检测失败: {e}")

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
                            content_sent = False
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
                                content_sent = True
                            
                            # 🔥 检查是否是最后一个chunk（finish_reason不为None表示结束）
                            if chunk.choices[0].finish_reason is not None:
                                print(f"✅ 检测到流式输出结束，finish_reason: {chunk.choices[0].finish_reason}, 最后内容已发送: {content_sent}")
                                # 🔥 确保最后一个chunk的内容已经发送后再跳出循环
                                break
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

            print(f"✅ 流式输出完成，总长度: {len(full_response)} 字符")
            print(f"🔥 最后50个字符: {full_response[-50:] if len(full_response) > 50 else full_response}")
            
            # 先快速保存对话（更新最后一条AI消息）
            saved_conversation_id = current_conversation_id
            if full_response.strip():
                try:
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
                except Exception as e:
                    print(f"⚠️ 保存对话失败: {e}")
            
            # 🚀 立即发送完成信号和conversation_id，不要等待其他操作
            yield f"data: {json.dumps({'done': True, 'conversation_id': saved_conversation_id or current_conversation_id}, ensure_ascii=False)}\n\n"
            
            # 然后再做耗时的保存操作（这些操作在后台完成，不影响前端显示）
            if full_response.strip():
                # 保存到个人记忆
                if username in memoryos_instances:
                    try:
                        memoryos_instance = memoryos_instances[username]
                        memoryos_instance.add_memory(message, full_response)
                        print("✅ 对话已保存到短期记忆")
                        
                        # 检测思维链断裂并发送到共享记忆
                        if shared_memory_enabled:
                            try:
                                check_and_store_chain_break_from_memoryos(username, memoryos_instance)
                                print("✅ 思维链检测完成")
                            except Exception as e:
                                print(f"⚠️ 思维链检测失败: {e}")
                    except Exception as e:
                        print(f"❌ 保存记忆失败: {e}")
                        print(f"错误类型: {type(e).__name__}")
                        import traceback
                        print(f"详细错误信息: {traceback.format_exc()}")
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
