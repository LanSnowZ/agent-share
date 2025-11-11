# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import os
import random
import re
import secrets
import string
import sys
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

import dotenv
from flask import (
    Flask,
    Response,
    g,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)
from flask_cors import CORS
from loguru import logger
from openai import OpenAI

from memoryos_pypi.memoryos import Memoryos
from memoryos_pypi.utils import check_conversation_continuity
from sharememory_user.config import Config
from sharememory_user.models import UserProfile
from sharememory_user.pipeline_retrieve import RetrievePipeline
from sharememory_user.storage import JsonStore
from src.config import cache_path_settings
from src.email_utils import send_email
from src.mcp_manager import get_event_loop, get_or_create_mcp_client

dotenv.load_dotenv()

# 设置 Hugging Face 镜像源（解决连接超时问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


app = Flask(__name__)
# 仅允许可信前端来源并支持携带凭据（用于设置HttpOnly Cookie）
CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": ["https://baijia.online"]}},
)

# JWT 配置
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_MINUTES = int(
    os.getenv("JWT_EXPIRES_MINUTES", "144000")
)  # 默认24小时（1440分钟）
JWT_COOKIE_NAME = "access_token"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = 4 - (len(data) % 4)
    if padding and padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("ascii"))


def create_jwt(payload: dict, exp_minutes: int = JWT_EXPIRES_MINUTES) -> str:
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    exp_ts = int(time.time()) + exp_minutes * 60
    body = dict(payload or {})
    body["exp"] = exp_ts
    header_b64 = _b64url_encode(
        json.dumps(header, separators=(",", ":")).encode("utf-8")
    )
    payload_b64 = _b64url_encode(
        json.dumps(body, separators=(",", ":")).encode("utf-8")
    )
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(
        JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    signature_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_jwt(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(_b64url_encode(expected_sig), signature_b64):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def set_jwt_cookie(resp: Response, token: str) -> Response:
    # HttpOnly 防XSS，SameSite=Strict 防CSRF，Secure 在HTTPS下生效
    resp.set_cookie(
        JWT_COOKIE_NAME,
        token,
        max_age=JWT_EXPIRES_MINUTES * 60,
        httponly=True,
        secure=True,
        samesite="Strict",
        path="/",
    )
    return resp


def clear_jwt_cookie(resp: Response) -> Response:
    resp.delete_cookie(JWT_COOKIE_NAME, path="/")
    return resp


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.cookies.get(JWT_COOKIE_NAME) or ""
        # 允许 Authorization: Bearer 用于调试/非浏览器客户端
        if not token:
            auth = request.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                token = auth.split(" ", 1)[1]
        payload = verify_jwt(token) if token else None
        if not payload or not payload.get("username"):
            return jsonify({"success": False, "error": "未登录或登录已过期"}), 401
        g.current_user = payload["username"]
        return fn(*args, **kwargs)

    return wrapper


# 临时存储验证码（在生产环境中应使用Redis等持久化存储）
verification_codes = {}  # {email: {"code": "123456", "username": "xxx", "expires_at": datetime}}
# 登录验证码单独存储，避免与注册冲突
login_codes = {}  # {email: {"code": "654321", "expires_at": datetime}}
# 重置密码验证码存储
reset_codes = {}  # {email: {"code": "123456", "expires_at": datetime}}

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
ingest_pipeline = None  # 延迟导入 IngestPipeline

DIMENSION_GROUPS_CN = {
    "basic_info": "基础信息",
    "psych": "心理模型",
    "align": "AI对齐维度",
    "interest": "内容兴趣标签",
}

DIMENSION_MAP_EN_TO_CN: dict[str, tuple[str, str]] = {
    # 基础信息
    "Name": ("姓名", "basic_info"),
    "Gender": ("性别", "basic_info"),
    "Age": ("年龄", "basic_info"),
    "Occupation": ("职业", "basic_info"),
    "Work Details": ("工作详情", "basic_info"),
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
    """从 MemoryOS 长期画像文本中提取维度 -> { 大维度中文: { 小维度中文: 等级中文/具体值 } }。
    仅解析能识别到的维度；未命中不返回。
    支持两种格式：
    - Level: High/Medium/Low（心理模型、AI对齐、兴趣标签）
    - Value: 具体值（基础信息）
    """
    grouped = {v: {} for v in DIMENSION_GROUPS_CN.values()}
    if not profile_text:
        return grouped

    # 支持多种格式：
    # 1. - **Name (Value: xxx)**  (旧格式，带星号和破折号)
    # 2. Name（Value: xxx）       (标准格式)
    # 3. Name（AI期望）（Level: xxx） (AI对齐维度格式，带额外标注)
    # 4. DimA / DimB（Level: xxx） (合并维度)
    patterns = [
        # 格式1: - **DimName (Type: Value)**
        re.compile(
            r"- \*\*\s*([^（(\*:]+?)\s*[（(][^:：]*?:\s*([^）)]+)[)）]\*\*\s*[:：]?"
        ),
        # 格式2: DimName（可选标注）（Type: Value）
        re.compile(
            r"^([A-Za-z\s/]+?)\s*(?:[（(][^)）]*?[)）]\s*)?[（(](?:Value|Level|Preference Level|Expectation Level):\s*([^）)]+)[)）]",
            re.MULTILINE,
        ),
    ]

    for pattern in patterns:
        for m in pattern.finditer(profile_text):
            en_name = m.group(1).strip()
            raw_value = m.group(2).strip()

            # 处理合并的维度名称（如 "Coherence / Truthfulness"）
            dim_names = [name.strip() for name in en_name.split("/")]

            for dim_name in dim_names:
                # 对于基础信息维度，直接使用原始值；对于其他维度，尝试映射
                mapped = DIMENSION_MAP_EN_TO_CN.get(dim_name)
                if not mapped:
                    continue
                dim_cn, group_key = mapped
                # 如果是基础信息维度，直接使用原始值；否则尝试映射等级
                if group_key == "basic_info":
                    value_cn = raw_value  # 基础信息使用具体值
                else:
                    value_cn = LEVEL_MAP_EN_TO_CN.get(
                        raw_value, raw_value
                    )  # 其他维度映射等级
                group_cn = DIMENSION_GROUPS_CN.get(group_key, group_key)
                grouped.setdefault(group_cn, {})
                grouped[group_cn][dim_cn] = value_cn

    return grouped


def sync_user_dimensions_to_store(user_id: str, profile_text: str) -> None:
    try:
        grouped = extract_profile_dimensions_from_text(profile_text)
        # 统计提取结果
        total_dims = sum(len(dims) for dims in grouped.values())
        print(f"\n{'=' * 60}")
        print(f"🔄 开始同步用户画像维度: {user_id}")
        print("📊 提取统计:")
        for group, dims in grouped.items():
            if dims:
                print(f"   • {group}: {len(dims)} 项")
        print(f"   总计: {total_dims} 个维度")

        # 读取现有用户，保持 profile_text
        user_profile = store.get_user(user_id)
        profile_text_to_keep = (
            user_profile.profile_text if user_profile else f"用户 {user_id}"
        )
        updated = UserProfile(
            user_id=user_id,
            profile_text=profile_text_to_keep,
            profile_dimensions=grouped,
        )
        store.add_user(updated)
        print("✅ 已同步结构化用户画像维度到 users.json")
        print(f"{'=' * 60}\n")
    except Exception as e:
        print(f"⚠️ 同步用户画像维度失败: {e}")

        traceback.print_exc()


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


def save_chain_to_shared_memory(user_id: str, chain_pages: List[Dict]) -> bool:
    """将对话链保存到共享记忆"""
    if not chain_pages or len(chain_pages) < 1:
        return False

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
            return True
        else:
            print("⚠️ 思维链未通过质量检查，未存储到共享记忆")
            return False
    except Exception as e:
        print(f"❌ 存储思维链到共享记忆失败: {e}")
    return False


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
                        "pre_page": page.get("pre_page"),
                    }
                    complete_chain.insert(0, qa)  # 插入到最前面
                    mid_term_count += 1
                    current_pre_page_id = page.get("pre_page")

                    if not current_pre_page_id:
                        print(
                            f"  ↳ 找到对话链起点（共追溯 {mid_term_count} 条中期记忆）"
                        )
                    elif mid_term_count % 5 == 0:  # 每5条打印一次进度
                        print(f"  ↳ 已追溯 {mid_term_count} 条...")
                else:
                    print(
                        f"  ✗ 页面 {current_pre_page_id} 未在中期记忆找到，停止追溯（已追溯 {mid_term_count} 条）"
                    )
                    break

        # 添加短期记忆的内容
        complete_chain.extend(start_qa_list)

        mid_count = len(complete_chain) - len(start_qa_list)
        print(
            f"🔗 完整链追溯完成: 共 {len(complete_chain)} 条（中期: {mid_count}, 短期: {len(start_qa_list)}）"
        )

    except Exception as e:
        print(f"⚠️ 追溯完整链失败: {e}")

        traceback.print_exc()
        # 失败时返回原始短期记忆内容
        return start_qa_list

    return complete_chain


def check_and_store_chain_break_from_memoryos(
    user_id: str,
    memoryos_instance,
    conversation_id: Optional[str] = None,
    project_name: str = "default_project",
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
            "timestamp": second_last_qa.get("timestamp", ""),
        }
        current_page = {
            "user_input": last_qa.get("user_input", ""),
            "agent_response": last_qa.get("agent_response", ""),
            "timestamp": last_qa.get("timestamp", ""),
        }

        # 检测对话连续性
        is_continuous = check_conversation_continuity(
            previous_page,
            current_page,
            memoryos_instance.client,
            model=config.llm_model_name,
        )

        if not is_continuous:
            # 思维链断裂！追溯完整链并发送到共享记忆
            short_term_broken = short_term_qa_list[:-1]  # 除了最后一条

            # 追溯完整的对话链（包括中期记忆）
            complete_chain = trace_complete_chain(memoryos_instance, short_term_broken)

            print(
                f"💡 检测到用户 {user_id} 的思维链断裂！完整对话链长度: {len(complete_chain)}"
            )

            # 转换为page格式并发送到共享记忆
            chain_pages = [
                {
                    "user_input": qa.get("user_input", ""),
                    "agent_response": qa.get("agent_response", ""),
                    "timestamp": qa.get("timestamp", ""),
                }
                for qa in complete_chain
            ]
            stored = save_chain_to_shared_memory(user_id, chain_pages)
            if stored and conversation_id:
                mark_conversation_shared_contribution(
                    user_id, conversation_id, project_name=project_name
                )

            # 🔪 断开链接：将最后一条（新话题开头）的 pre_page 置空
            old_pre_page_id = last_qa.get("pre_page")
            last_qa["pre_page"] = None

            # 同时将倒数第二条的 next_page 置空（可能在短期或中期）
            second_last_page_id = second_last_qa.get("page_id")
            if second_last_page_id:
                # 先尝试在短期记忆中更新
                second_last_qa["next_page"] = None

                # 如果倒数第二条已经在中期记忆，也需要更新
                mid_page = get_page_from_mid_term(
                    memoryos_instance, second_last_page_id
                )
                if mid_page:
                    mid_page["next_page"] = None
                    memoryos_instance.mid_term_memory.save()

            # 保存短期记忆以持久化链接断开
            memoryos_instance.short_term_memory.save()

            print(f"✂️ 已断开对话链链接（pre_page: {old_pre_page_id} → None，开始新链）")
            print("📤 完整对话链已发送到共享记忆")
        else:
            # 计算当前完整对话链长度（包括中期）
            current_chain = trace_complete_chain(memoryos_instance, short_term_qa_list)
            print(f"✅ 对话连续，完整对话链长度: {len(current_chain)}")

    except Exception as e:
        print(f"⚠️ 从MemoryOS检测思维链断裂失败: {e}")

        traceback.print_exc()


def ensure_user_memoryos(
    user_id: str, project_name: str = "default_project"
) -> Optional[Memoryos]:
    """确保用户有MemoryOS实例，如果没有则创建"""
    project_name = user_id
    if user_id not in memoryos_instances:
        try:
            # 创建用户数据目录 - 按照项目/用户层级结构: eval/memoryos_data/{project}/users/{user_id}
            user_data_dir = os.path.join(
                cache_path_settings.MEMORYOS_DATA_DIR, project_name, 
            )
            os.makedirs(user_data_dir, exist_ok=True)
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
        cache_path_settings.MEMORYOS_DATA_DIR,
        project_name,
        "users",
        user_id,
        "config.json",
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
        user_dir = os.path.join(
            cache_path_settings.MEMORYOS_DATA_DIR, project_name, "users", user_id
        )
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
    conversations: List[Dict[str, Any]] = []
    conversations_path = os.path.join(
        cache_path_settings.MEMORYOS_DATA_DIR,
        project_name,
        "users",
        user_id,
        "conversations.json",
    )
    if os.path.exists(conversations_path):
        try:
            with open(conversations_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                for convo in loaded:
                    if isinstance(convo, dict):
                        convo.setdefault("contributed_shared_memory", False)
                if loaded:
                    return loaded
            conversations = loaded if isinstance(loaded, list) else []
        except Exception as e:
            print(f"读取对话列表失败（conversations.json损坏或格式错误）: {e}")

    # fallback: 扫描该用户目录下的所有 chat_*.json 文件并构建列表
    user_dir = os.path.join(
        cache_path_settings.MEMORYOS_DATA_DIR, project_name, "users", user_id
    )
    if not os.path.isdir(user_dir):
        return conversations

    chat_files: List[Dict[str, Any]] = []
    for filename in os.listdir(user_dir):
        if not (filename.startswith("chat_") and filename.endswith(".json")):
            continue
        chat_path = os.path.join(user_dir, filename)
        try:
            with open(chat_path, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
        except Exception as e:
            print(f"读取对话文件失败 ({chat_path}): {e}")
            continue

        conversation_id = chat_data.get("id") or filename.rsplit(".", 1)[0]
        title = chat_data.get("title") or ""
        if not title:
            first_user_message = next(
                (
                    msg.get("content", "")
                    for msg in chat_data.get("messages", [])
                    if isinstance(msg, dict) and msg.get("type") == "user"
                ),
                "新对话",
            )
            title = first_user_message[:30] + (
                "..." if len(first_user_message) > 30 else ""
            )

        chat_files.append(
            {
                "id": conversation_id,
                "title": title or "新对话",
                "created_at": chat_data.get("created_at"),
                "updated_at": chat_data.get("updated_at"),
                "model": chat_data.get("model"),
                "contributed_shared_memory": bool(
                    chat_data.get("contributed_shared_memory")
                ),
            }
        )

    # 使用 updated_at (如果存在) 倒序排序，确保最新对话在前
    chat_files.sort(
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        reverse=True,
    )

    if conversations:
        # conversations.json 已有数据，优先返回原数据，若为空则回退扫描结果
        return conversations

    return chat_files


def save_chat_conversations(
    user_id: str,
    conversations: List[Dict[str, Any]],
    project_name: str = "default_project",
) -> bool:
    """保存用户的聊天对话列表"""
    try:
        user_dir = os.path.join(
            cache_path_settings.MEMORYOS_DATA_DIR, project_name, "users", user_id
        )
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
        cache_path_settings.MEMORYOS_DATA_DIR,
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


def save_used_memories_to_conversation(
    conversation_id: str, memory_ids: List[str], username: str
) -> None:
    """保存对话中使用的共享记忆ID和focus_query"""
    try:
        print("\n🔧 开始保存使用的记忆ID:")
        print(f"  - 对话ID: {conversation_id}")
        print(f"  - 用户名: {username}")
        print(f"  - 记忆ID列表: {memory_ids}")

        # 构建对话文件路径
        conversation_file = os.path.join(
            cache_path_settings.MEMORYOS_DATA_DIR,
            "default_project",
            "users",
            username,
            f"{conversation_id}.json",
        )
        print(f"  - 对话文件路径: {conversation_file}")
        print(f"  - 对话文件是否存在: {os.path.exists(conversation_file)}")

        if os.path.exists(conversation_file):
            with open(conversation_file, "r", encoding="utf-8") as f:
                conversation_data = json.load(f)

            # 添加使用的记忆ID到对话数据中
            if "used_memories" not in conversation_data:
                conversation_data["used_memories"] = []

            # 从memory.json文件获取所有记忆，用于查找focus_query
            memory_id_to_focus_query = {}

            if os.path.exists(cache_path_settings.MEMORY_FILE_PATH):
                try:
                    with open(
                        cache_path_settings.MEMORY_FILE_PATH, "r", encoding="utf-8"
                    ) as f:
                        memory_data = json.load(f)
                        memories_list = memory_data.get("memories", [])
                        for mem in memories_list:
                            memory_id_to_focus_query[mem.get("id")] = mem.get(
                                "focus_query", ""
                            )
                    print(
                        f"  - 从memory.json加载了 {len(memory_id_to_focus_query)} 个记忆的focus_query"
                    )
                except Exception as e:
                    print(f"  - 读取memory.json失败: {e}")

            # 将新的记忆ID和focus_query添加到列表中（避免重复）
            existing_memory_ids = set()
            for existing_memory in conversation_data["used_memories"]:
                if isinstance(existing_memory, dict):
                    existing_memory_ids.add(existing_memory.get("id"))
                else:
                    existing_memory_ids.add(existing_memory)

            for memory_id in memory_ids:
                if memory_id not in existing_memory_ids:
                    focus_query = memory_id_to_focus_query.get(memory_id, "")
                    memory_info = {"id": memory_id, "focus_query": focus_query}
                    conversation_data["used_memories"].append(memory_info)
                    print(
                        f"✅ 保存记忆ID: {memory_id}, focus_query: {focus_query[:50]}..."
                    )
                else:
                    print(f"⚠️ 记忆ID已存在，跳过: {memory_id}")

            print(
                f"  - 保存前used_memories数量: {len(conversation_data['used_memories'])}"
            )

            # 保存更新后的对话数据
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)

            print(f"✅ 已保存使用的记忆ID和focus_query到对话: {conversation_id}")
    except Exception as e:
        print(f"⚠️ 保存使用的记忆ID失败: {e}")


def mark_conversation_shared_contribution(
    user_id: str, conversation_id: str, project_name: str = "default_project"
) -> None:
    """标记指定对话参与了共享记忆的构建"""
    conversation_file = os.path.join(
        cache_path_settings.MEMORYOS_DATA_DIR,
        project_name,
        "users",
        user_id,
        f"{conversation_id}.json",
    )

    if not os.path.exists(conversation_file):
        print(
            f"⚠️ 无法标记对话共享记忆贡献，文件不存在: {conversation_file}"
        )
        return

    try:
        with open(conversation_file, "r", encoding="utf-8") as f:
            conversation_data = json.load(f)
    except Exception as e:
        print(f"⚠️ 读取对话文件失败，无法标记共享记忆贡献: {e}")
        return

    if not conversation_data.get("contributed_shared_memory"):
        conversation_data["contributed_shared_memory"] = True
        try:
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            print(
                f"⭐ 已标记对话 {conversation_id} 参与构建共享记忆"
            )
        except Exception as e:
            print(f"⚠️ 更新对话文件失败: {e}")

    # 同步更新 conversations.json（如果存在的话）
    conversations_path = os.path.join(
        cache_path_settings.MEMORYOS_DATA_DIR,
        project_name,
        "users",
        user_id,
        "conversations.json",
    )
    if not os.path.exists(conversations_path):
        return

    try:
        with open(conversations_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        if isinstance(conversations, list):
            updated = False
            for convo in conversations:
                if convo.get("id") == conversation_id:
                    if not convo.get("contributed_shared_memory"):
                        convo["contributed_shared_memory"] = True
                        updated = True
                    break
            if updated:
                with open(conversations_path, "w", encoding="utf-8") as f:
                    json.dump(conversations, f, ensure_ascii=False, indent=2)
                print("⭐ 已同步 conversations.json 中的共享记忆标记")
    except Exception as e:
        print(f"⚠️ 更新 conversations.json 失败: {e}")


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
    used_shared_memory_ids=None,
):
    """保存聊天对话到chat文件夹"""

    # 创建用户目录
    user_chat_dir = os.path.join(
        cache_path_settings.MEMORYOS_DATA_DIR, "default_project", "users", username
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
        "used_memories": [],  # 添加used_memories字段
        "contributed_shared_memory": False,
    }

    if os.path.exists(conversation_file):
        try:
            with open(conversation_file, "r", encoding="utf-8") as f:
                conversation_data = json.load(f)
            # 确保used_memories字段存在
            if "used_memories" not in conversation_data:
                conversation_data["used_memories"] = []
            if "contributed_shared_memory" not in conversation_data:
                conversation_data["contributed_shared_memory"] = False
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
                "shared_memory_enabled": shared_memory_enabled,
                "used_shared_memories": used_shared_memory_ids or [],
                "shareable": False,  # 默认不可分享，点击分享按钮后会更新为True
            }
            conversation_data["messages"].append(new_ai_message)
        elif messages[-1]["type"] == "assistant":
            # 最后一条是AI消息，更新它（流式输出的增量更新）
            messages[-1]["content"] = ai_response
            messages[-1]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            messages[-1]["shared_memory_enabled"] = shared_memory_enabled
            messages[-1]["used_shared_memories"] = used_shared_memory_ids or []
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
                "shared_memory_enabled": shared_memory_enabled,
                "used_shared_memories": used_shared_memory_ids or [],
                "shareable": False,  # 默认不可分享，点击分享按钮后会更新为True
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


def increment_shared_memory_contribution(memory_ids: List[str]) -> None:
    """为指定的共享记忆增加贡献值计数"""
    if not memory_ids:
        return

    # 去重并过滤空ID
    unique_ids: List[str] = []
    for mem_id in memory_ids:
        if mem_id and mem_id not in unique_ids:
            unique_ids.append(mem_id)

    if not unique_ids:
        return

    try:
        all_memories = store.list_memories()
        memories_map = {mem.id: mem for mem in all_memories if mem.id in unique_ids}

        for mem_id in unique_ids:
            memory_item = memories_map.get(mem_id)
            if not memory_item:
                print(f"ℹ️ 未找到需要累加贡献值的记忆: {mem_id}")
                continue

            memory_item.meta = memory_item.meta or {}
            raw_score = memory_item.meta.get("contribution_score", 0)
            try:
                score_int = int(raw_score)
            except (ValueError, TypeError):
                score_int = 0

            score_int = max(score_int, 0) + 1
            memory_item.meta["contribution_score"] = score_int

            store.update_memory(memory_item)
            print(f"📈 共享记忆贡献值 +1: id={mem_id}, 当前贡献值={score_int}")

    except Exception as e:
        print(f"⚠️ 更新共享记忆贡献值失败: {e}")

        traceback.print_exc()


def load_conversation_history(username, conversation_id):
    """加载对话历史"""
    try:
        conversation_file = os.path.join(
            cache_path_settings.MEMORYOS_DATA_DIR,
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
    return f"""你是一个有用的AI助手。你的任务是基于两个记忆源提供的上下文来回答用户的问题。
上下文来自共享知识库和你对过去对话的个人记忆。
综合这两个来源的信息，提供全面准确的答案。
如果上下文不相关，忽略它，基于你自己的知识回答。

**用户画像:**
---
{user_profile}
---

**来自共享记忆的上下文:**
---
{shared_memory_context}
---

**来自个人记忆的上下文:**
---
{personal_memory_context}
---

**用户问题:**
---
{user_query}
---

重要提示：根据用户的画像、专业水平和职业背景调整你的回答。
你的回复应该与这个特定用户相关且合适。
结合共享记忆和个人记忆的见解，提供最有帮助的回复。

你的回答:
"""


def get_rag_answer_prompt(
    user_query: str, retrieved_context: str, user_profile: str
) -> str:
    """
    创建仅使用共享记忆的RAG提示词
    """
    return f"""你是一个有用的AI助手。你的任务是基于提供的上下文来回答用户的问题。
上下文来自过去对话的共享知识库。
综合上下文中的信息，提供全面准确的答案。
如果上下文不相关，忽略它，基于你自己的知识回答。

**用户画像:**
---
{user_profile}
---

**来自共享记忆的上下文:**
---
{retrieved_context}
---

**用户问题:**
---
{user_query}
---

重要提示：根据用户的画像、专业水平和职业背景调整你的回答。
你的回复应该与这个特定用户相关且合适。

你的回答:
"""


def get_baseline_answer_prompt(user_query: str, user_profile: str) -> str:
    """
    创建不使用任何记忆的基线提示词
    """
    return f"""你是一个有用的AI助手。请基于你自己的知识回答用户的问题。

**用户画像:**
---
{user_profile}
---

**用户问题:**
---
{user_query}
---

重要提示：根据用户的画像、专业水平和职业背景调整你的回答。
你的回复应该与这个特定用户相关且合适。

你的回答:
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
**当前对话上下文:**
---
{conversation_context}
---

"""

    return f"""
{context_section}**用户问题:**
---
{user_query}
---
你的回答:
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
**当前对话上下文:**
---
{conversation_context}
---

"""

    return f"""你是一个有用的AI助手。你的任务是基于提供的上下文来回答用户的问题。
上下文来自过去对话的共享知识库。
综合上下文中的信息，提供全面准确的答案。
如果上下文不相关，忽略它，基于你自己的知识回答。
{context_section}**用户画像:**
---
{user_profile}
---

**来自共享记忆的上下文:**
---
{retrieved_context}
---

**用户问题:**
---
{user_query}
---

重要提示：根据用户的画像、专业水平和职业背景调整你的回答。
你的回复应该与这个特定用户相关且合适。

重要提示：你可以查看并参考上面提供的对话历史。使用它来提供与上下文相关的回复。

你的回答:
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
**当前对话上下文:**
---
{conversation_context}
---

"""

    return f"""你是一个有用的AI助手。你的任务是基于两个记忆源提供的上下文来回答用户的问题。
上下文来自共享知识库和你对过去对话的个人记忆。
综合这两个来源的信息，提供全面准确的答案。
如果上下文不相关，忽略它，基于你自己的知识回答。
{context_section}**用户画像:**
---
{user_profile}
---

**来自共享记忆的上下文:**
---
{shared_memory_context}
---

**来自个人记忆的上下文:**
---
{personal_memory_context}
---

**用户问题:**
---
{user_query}
---

重要提示：根据用户的画像、专业水平和职业背景调整你的回答。
使用共享知识和个人上下文来提供与这个特定用户相关且合适的回复。
如果共享记忆和个人记忆之间存在冲突，优先考虑与用户当前问题最相关的信息。

重要提示：你可以查看并参考上面提供的对话历史。使用它来提供与上下文相关的回复。

你的回答:
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
                print("\n🔍 开始检索共享记忆...")
                print(f"  - 用户: {user_id}")
                print(f"  - 消息: {message[:50]}...")
                print(f"  - 对话ID: {conversation_id}")

                retrieval_result = retrieve_pipeline.retrieve(
                    user=enhanced_user_profile, task=message, peers=peers, top_k=3
                )

                print(f"  - 检索结果: {retrieval_result}")
                print(f"  - 检索到的项目数量: {len(retrieval_result.get('items', []))}")

                # 打印最终选中的共享记忆ID（在构建提示词前）
                try:
                    selected_ids = [
                        it.get("memory", {}).get("id", "NO_ID_FOUND")
                        for it in retrieval_result.get("items", [])
                        if isinstance(it, dict)
                    ]
                    print(f"  - 选中的记忆ID: {selected_ids}")

                    if selected_ids:
                        print(f"✅ 共享记忆已选中ID: {', '.join(selected_ids)}")
                        # 将选中的记忆ID保存到对话中，用于后续显示
                        if conversation_id:
                            print(f"  - 开始保存记忆ID到对话: {conversation_id}")
                            save_used_memories_to_conversation(
                                conversation_id, selected_ids, user_id
                            )
                        else:
                            print("  - 警告: conversation_id为空，无法保存记忆ID")
                    else:
                        print("ℹ️ 共享记忆未选中任何条目（为空或被QC过滤）")
                except Exception as log_err:
                    print(f"⚠️ 打印共享记忆ID失败: {log_err}")
                    traceback.print_exc()

                if retrieval_result["items"]:
                    shared_memory_context = retrieve_pipeline.build_prompt_blocks(
                        retrieval_result["items"], conversation_id, user_id
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


@app.route("/share")
def share_page():
    """分享页面"""
    return render_template("share.html")


# 分享路由需要放在其他路由之前，避免匹配冲突
@app.route("/<share_token>")
def share_view(share_token):
    """分享链接视图 - 格式: /{chat_id}{timestamp_numeric}
    返回主页，通过 URL 传递分享参数，由前端 JavaScript 处理
    """
    # 验证 share_token 格式

    match = re.match(r"(chat_\d+)(\d{14})", share_token)
    if not match:
        # 如果不是分享链接格式，返回主页（可能是其他路由）
        return render_template("index.html")

    # 是分享链接，返回主页并传递 share_token
    return render_template("index.html", share_token=share_token)


@app.route("/api/get_shared_message", methods=["GET"])
def get_shared_message():
    """获取分享的消息内容（不需要登录）"""
    try:
        share_token = request.args.get("share_token")
        if not share_token:
            return jsonify({"success": False, "error": "缺少分享令牌"})

        # 解析 share_token

        match = re.match(r"(chat_\d+)(\d{14})", share_token)
        if not match:
            return jsonify({"success": False, "error": "无效的分享令牌格式"})

        chat_id = match.group(1)
        timestamp_numeric = match.group(2)

        # 将 timestamp_numeric 转换回时间戳格式
        timestamp_str = (
            timestamp_numeric[:4]
            + "-"
            + timestamp_numeric[4:6]
            + "-"
            + timestamp_numeric[6:8]
            + " "
            + timestamp_numeric[8:10]
            + ":"
            + timestamp_numeric[10:12]
            + ":"
            + timestamp_numeric[12:14]
        )

        # 查找所有用户目录，找到包含该 chat_id 的对话
        users_dir = os.path.join(
            cache_path_settings.MEMORYOS_DATA_DIR, "default_project", "users"
        )
        if not os.path.exists(users_dir):
            return jsonify({"success": False, "error": "分享的对话不存在"})

        for username in os.listdir(users_dir):
            user_dir = os.path.join(users_dir, username)
            if not os.path.isdir(user_dir):
                continue

            conversation_file = os.path.join(user_dir, f"{chat_id}.json")
            if os.path.exists(conversation_file):
                try:
                    with open(conversation_file, "r", encoding="utf-8") as f:
                        conv_data = json.load(f)

                    # 查找匹配 timestamp 的 AI 消息
                    for msg in conv_data.get("messages", []):
                        if (
                            msg.get("type") == "assistant"
                            and msg.get("timestamp") == timestamp_str
                        ):
                            return jsonify(
                                {
                                    "success": True,
                                    "message": msg,
                                    "model": conv_data.get("model", "gpt-4o-mini"),
                                    "original_username": username,
                                    "share_token": share_token,
                                }
                            )
                except Exception as e:
                    print(f"读取对话文件失败: {e}")
                    continue

        return jsonify({"success": False, "error": "分享的消息不存在"})

    except Exception as e:
        print(f"获取分享消息失败: {e}")
        return jsonify({"success": False, "error": f"获取分享消息失败: {str(e)}"})


@app.route("/api/get_shared_memories", methods=["POST"])
@login_required
def get_shared_memories():
    """获取共享记忆API - 只返回当前登录用户参与的共享记忆，master用户可以看到全部"""
    try:
        data = request.get_json()
        username = g.get("current_user") or data.get("username")
        limit = data.get("limit", 10000)  # 默认限制改为10000，可以获取所有共享记忆

        print("\n📊 获取共享记忆请求:")
        print(f"  - 用户名: {username}")
        print(f"  - 限制数量: {limit}")

        if not username:
            return jsonify({"success": False, "error": "缺少用户名"})

        # 检查用户是否为master
        is_master = False
        try:
            if os.path.exists(cache_path_settings.USER_FILE_PATH):
                with open(
                    cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8"
                ) as f:
                    user_data = json.load(f)
                users = user_data.get("users", [])
                u = next((x for x in users if x.get("username") == username), None)
                if u and u.get("role") == "master":
                    is_master = True
                    print(f"  - 用户 {username} 是 master，将返回所有共享记忆")
        except Exception as e:
            print(f"  - 检查用户role时出错: {e}")

        # 获取所有共享记忆
        all_memories = store.list_memories()
        print(f"  - 总记忆数量: {len(all_memories)}")

        # 如果是master，直接使用所有记忆；否则过滤出当前用户参与的共享记忆
        if is_master:
            user_memories = all_memories
            print(f"  - master用户，返回所有共享记忆: {len(user_memories)}")
        else:
            user_memories = []
            for mem in all_memories:
                # 获取merged_users字段，如果不存在则使用source_user_id
                merged_users = []
                if hasattr(mem, "meta") and mem.meta:
                    merged_users = mem.meta.get("merged_users", [])

                # 如果merged_users为空，使用source_user_id作为fallback
                if (
                    not merged_users
                    and hasattr(mem, "source_user_id")
                    and mem.source_user_id
                ):
                    merged_users = [mem.source_user_id]

                # 检查当前用户是否参与了该记忆
                if username in merged_users:
                    user_memories.append(mem)
                    print(f"  - 用户 {username} 参与了记忆: {mem.id}")

            print(f"  - 用户参与的共享记忆数量: {len(user_memories)}")

        # 按照创建时间从新到旧排序
        user_memories_sorted = sorted(
            user_memories,
            key=lambda mem: mem.created_at if mem.created_at else 0,
            reverse=True,  # 降序排列，最新的在前面
        )
        print("  - 已按时间倒序排序")

        # 转换为字典格式
        memories_list = []
        for i, mem in enumerate(user_memories_sorted[:limit]):
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

                # 获取merged_users字段，如果不存在则使用source_user_id
                merged_users = []
                if hasattr(mem, "meta") and mem.meta:
                    merged_users = mem.meta.get("merged_users", [])

                # 如果merged_users为空，使用source_user_id作为fallback
                if (
                    not merged_users
                    and hasattr(mem, "source_user_id")
                    and mem.source_user_id
                ):
                    merged_users = [mem.source_user_id]

                raw_contribution = 0
                if hasattr(mem, "meta") and mem.meta:
                    raw_contribution = mem.meta.get("contribution_score", 0)
                try:
                    contribution_score = max(int(raw_contribution), 0)
                except (ValueError, TypeError):
                    contribution_score = 0

                memory_data = {
                    "id": mem.id,
                    "user_id": mem.source_user_id,
                    "content": content,
                    "timestamp": timestamp_str,
                    "created_at": mem.created_at,  # 添加原始时间戳用于调试
                    "source": mem.meta.get("source", "conversation")
                    if hasattr(mem, "meta") and mem.meta
                    else "conversation",
                    "focus_query": focus_query,
                    "merged_users": merged_users,
                    "contribution_score": contribution_score,
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
            {"success": True, "memories": memories_list, "total": len(user_memories)}
        )

    except Exception as e:
        print(f"❌ 获取共享记忆失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/get_memory_file", methods=["GET"])
@login_required
def get_memory_file():
    """获取用户的记忆文件（短期、中期、长期）"""
    try:
        username = g.get("current_user") or request.args.get("username")
        file_name = request.args.get("file")

        if not username or not file_name:
            return jsonify({"success": False, "error": "缺少参数"}), 400

        # 检查文件名是否合法
        allowed_files = ["short_term.json", "mid_term.json", "long_term_user.json"]
        if file_name not in allowed_files:
            return jsonify({"success": False, "error": "非法的文件名"}), 400

        # 构建文件路径：eval/memoryos_data/{username}/users/{username}/{file}
        file_path = os.path.join(
            cache_path_settings.MEMORYOS_DATA_DIR,
            username,
            "users",
            username,
            file_name,
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

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/get_user_dimensions", methods=["GET"])
@login_required
def get_user_dimensions():
    """获取统一后的结构化用户画像维度（按三大类分组，仅显示已存在的小维度）。"""
    try:
        username = g.get("current_user") or request.args.get("username")
        if not username:
            return jsonify({"success": False, "error": "缺少用户名"}), 400

        user_profile = store.get_user(username)
        grouped = None
        if user_profile and getattr(user_profile, "profile_dimensions", None):
            grouped = user_profile.profile_dimensions
        else:
            # fallback: 从长期画像文本即时解析
            user_dir = os.path.join(
                cache_path_settings.MEMORYOS_DATA_DIR, username, "users", username
            )
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

        return jsonify(
            {"success": True, "dimensions": grouped, "groups": DIMENSION_GROUPS_CN}
        )
    except Exception as e:
        print(f"获取用户画像维度失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/get_quota", methods=["GET"])
@login_required
def get_quota():
    """获取用户额度信息（从 user.json 读取）"""
    try:
        username = (request.args.get("username") or "").strip()
        if not username:
            return jsonify({"success": False, "error": "缺少用户名"}), 400

        if not os.path.exists(cache_path_settings.USER_FILE_PATH):
            return jsonify({"success": True, "quota_total": 100000, "quota_used": 0})

        with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        users = user_data.get("users", [])
        u = next((x for x in users if x.get("username") == username), None)
        if not u:
            return jsonify({"success": True, "quota_total": 100000, "quota_used": 0})

        total = int(u.get("quota_total", 100000) or 100000)
        used = int(u.get("quota_used", 0) or 0)
        return jsonify({"success": True, "quota_total": total, "quota_used": used})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 登录态自检
@app.route("/api/me", methods=["GET"])
def me():
    token = request.cookies.get(JWT_COOKIE_NAME) or ""
    payload = verify_jwt(token) if token else None
    if not payload or not payload.get("username"):
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "username": payload["username"]})


# 安全退出：清除Cookie
@app.route("/api/logout", methods=["POST"])
def logout():
    resp = make_response(jsonify({"success": True}))
    return clear_jwt_cookie(resp)


def generate_with_mcp_tools(
    prompt: str,
    username: str,
    conversation_id: str,
    message: str,
    model: str,
    shared_memory_enabled: bool,
    personal_memory_enabled: bool,
    used_shared_memory_ids: List[str],
    api_key: str,
    base_url: str,
):
    """
    使用 MCP 工具调用的流式生成器

    Args:
        prompt: 构建好的提示词
        username: 用户名
        conversation_id: 对话 ID
        message: 原始用户消息
        model: 模型名称
        shared_memory_enabled: 是否启用共享记忆
        personal_memory_enabled: 是否启用个人记忆
        used_shared_memory_ids: 使用的共享记忆 ID 列表
        api_key: OpenAI API Key
        base_url: OpenAI Base URL

    Yields:
        SSE 格式的数据流
    """
    try:
        # 获取或创建 MCP 客户端
        mcp_client = get_or_create_mcp_client()

        if not mcp_client:
            logger.warning("MCP 客户端初始化失败，回退到普通模式")
            yield f"data: {json.dumps({'error': 'MCP 客户端初始化失败'}, ensure_ascii=False)}\n\n"
            return

        # 设置用户的 API 配置
        try:
            mcp_client.set_api_config(api_key=api_key, base_url=base_url, model=model)
            logger.info(f"为用户 {username} 设置 MCP API 配置: {base_url}, {model}")
        except Exception as e:
            logger.error(f"设置 API 配置失败: {e}")
            yield f"data: {json.dumps({'error': f'设置 API 配置失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            return

        # 获取事件循环
        loop = get_event_loop()
        if not loop:
            logger.error("无法获取事件循环")
            yield f"data: {json.dumps({'error': '无法获取事件循环'}, ensure_ascii=False)}\n\n"
            return

        # 使用队列在异步和同步代码之间传递事件
        import queue
        event_queue = queue.Queue()
        exception_holder = []

        # 创建异步任务，将事件放入队列
        async def async_producer():
            try:
                async for event in mcp_client.process_query_streaming(prompt):
                    event_queue.put(event)
                event_queue.put(None)  # 结束标记
            except Exception as e:
                exception_holder.append(e)
                event_queue.put(None)

        # 在事件循环中启动异步任务
        import asyncio
        asyncio.run_coroutine_threadsafe(async_producer(), loop)

        full_response = ""
        stream_interrupted = False

        try:
            while True:
                try:
                    # 从队列中获取事件（带超时避免永久阻塞）
                    event = event_queue.get(timeout=300)  # 5分钟超时

                    # 检查是否有异常
                    if exception_holder:
                        raise exception_holder[0]

                    # 检查是否结束
                    if event is None:
                        break

                    if event["type"] == "content":
                        # LLM 生成的文本内容
                        content = event["data"]
                        full_response += content
                        yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

                    elif event["type"] == "tool_call_start":
                        # 工具调用开始
                        yield f"data: {json.dumps({'tool_status': 'start', 'tool_name': event['tool_name'], 'arguments': event.get('arguments', {})}, ensure_ascii=False)}\n\n"

                    elif event["type"] == "tool_call_end":
                        # 工具调用完成
                        yield f"data: {json.dumps({'tool_status': 'end', 'tool_name': event['tool_name'], 'elapsed_time': event.get('elapsed_time', 0)}, ensure_ascii=False)}\n\n"

                    elif event["type"] == "thinking":
                        # AI 思考状态
                        yield f"data: {json.dumps({'thinking': event.get('status', 'Thinking...')}, ensure_ascii=False)}\n\n"

                    elif event["type"] == "error":
                        # 错误信息
                        logger.error(f"MCP 错误: {event.get('error')}")
                        yield f"data: {json.dumps({'error': event.get('error')}, ensure_ascii=False)}\n\n"

                    elif event["type"] == "done":
                        # 处理完成
                        break

                except queue.Empty:
                    # 队列超时，可能是处理时间过长
                    logger.warning("从事件队列获取事件超时")
                    yield f"data: {json.dumps({'error': '处理超时'}, ensure_ascii=False)}\n\n"
                    break

                except GeneratorExit:
                    # 客户端断开连接
                    logger.warning("客户端断开连接 (MCP 模式)")
                    stream_interrupted = True
                    break

        except GeneratorExit:
            logger.warning("流式输出被中断 (MCP 模式)")
            stream_interrupted = True

        except Exception as e:
            logger.exception(f"MCP 流式处理异常: {e}")
            yield f"data: {json.dumps({'error': f'处理异常: {str(e)}'}, ensure_ascii=False)}\n\n"
            return

        # 保存对话
        if full_response.strip():
            try:
                saved_conversation_id = save_chat_conversation(
                    username,
                    conversation_id,
                    message,
                    full_response,
                    model,
                    shared_memory_enabled,
                    personal_memory_enabled,
                    used_shared_memory_ids=used_shared_memory_ids,
                    update_last_ai_message=True,
                )

                # 发送完成信号
                if not stream_interrupted:
                    yield f"data: {json.dumps({'done': True, 'conversation_id': saved_conversation_id or conversation_id}, ensure_ascii=False)}\n\n"

                # 更新共享记忆贡献值
                if used_shared_memory_ids:
                    try:
                        increment_shared_memory_contribution(used_shared_memory_ids)
                    except Exception as e:
                        logger.warning(f"累计共享记忆贡献值失败: {e}")

            except Exception as e:
                logger.error(f"保存对话失败: {e}")
                yield f"data: {json.dumps({'error': f'保存对话失败: {str(e)}'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.exception(f"generate_with_mcp_tools 异常: {e}")
        yield f"data: {json.dumps({'error': f'严重错误: {str(e)}'}, ensure_ascii=False)}\n\n"


@app.route("/chat_direct", methods=["POST"])
@login_required
def chat_direct():
    """流式聊天API - 使用Server-Sent Events"""

    # 在流式上下文外读取请求数据
    data = request.get_json()
    username = g.get("current_user") or data.get("username")
    message = data.get("message")
    model = data.get("model", "gpt-4o-mini")
    conversation_id = data.get("conversation_id")
    shared_memory_enabled = data.get("shared_memory_enabled", False)
    personal_memory_enabled = data.get("personal_memory_enabled", True)
    project_name = data.get("project_name", "default_project")
    mcp_enabled = data.get("mcp_enabled", False)  # MCP 工具调用开关

    # 处理分享消息（如果是从分享链接访问）
    shared_message_content = data.get("shared_message_content")
    shared_message_timestamp = data.get("shared_message_timestamp")
    shared_message_memory_enabled = data.get("shared_message_memory_enabled", False)

    # 如果有分享消息，先保存到对话中（只保存AI消息，不保存用户消息）
    if shared_message_content and shared_message_timestamp and conversation_id:
        try:
            # 直接创建对话文件，只包含分享的AI消息
            user_chat_dir = os.path.join(
                cache_path_settings.MEMORYOS_DATA_DIR,
                "default_project",
                "users",
                username,
            )
            os.makedirs(user_chat_dir, exist_ok=True)
            conversation_file = os.path.join(user_chat_dir, f"{conversation_id}.json")

            # 如果文件不存在，创建新对话
            if not os.path.exists(conversation_file):
                conversation_data = {
                    "id": conversation_id,
                    "username": username,
                    "model": model,
                    "shared_memory_enabled": shared_message_memory_enabled,
                    "personal_memory_enabled": personal_memory_enabled,
                    "created_at": shared_message_timestamp,
                    "updated_at": shared_message_timestamp,
                    "messages": [
                        {
                            "type": "assistant",
                            "content": shared_message_content,
                            "timestamp": shared_message_timestamp,
                            "shared_memory_enabled": shared_message_memory_enabled,
                            "used_shared_memories": [],
                            "shareable": False,
                        }
                    ],
                    "used_memories": [],
                    "title": shared_message_content[:30]
                    + ("..." if len(shared_message_content) > 30 else ""),
                }
                with open(conversation_file, "w", encoding="utf-8") as f:
                    json.dump(conversation_data, f, ensure_ascii=False, indent=2)
                print(f"✅ 已保存分享的AI消息到新对话 {conversation_id}")
            else:
                # 如果文件已存在，检查是否已有该消息，如果没有则添加
                with open(conversation_file, "r", encoding="utf-8") as f:
                    conversation_data = json.load(f)

                # 检查是否已有该消息
                message_exists = any(
                    msg.get("type") == "assistant"
                    and msg.get("timestamp") == shared_message_timestamp
                    and msg.get("content") == shared_message_content
                    for msg in conversation_data.get("messages", [])
                )

                if not message_exists:
                    # 在开头插入分享的消息
                    if "messages" not in conversation_data:
                        conversation_data["messages"] = []
                    conversation_data["messages"].insert(
                        0,
                        {
                            "type": "assistant",
                            "content": shared_message_content,
                            "timestamp": shared_message_timestamp,
                            "shared_memory_enabled": shared_message_memory_enabled,
                            "used_shared_memories": [],
                            "shareable": False,
                        },
                    )
                    conversation_data["updated_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    with open(conversation_file, "w", encoding="utf-8") as f:
                        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
                    print(f"✅ 已添加分享的AI消息到对话 {conversation_id}")
        except Exception as e:
            print(f"⚠️ 保存分享消息失败: {e}")

            traceback.print_exc()

    def generate():
        try:
            if not username or not message:
                yield f"data: {json.dumps({'error': '缺少必要参数'}, ensure_ascii=False)}\n\n"
                return

            # 确保用户有MemoryOS实例
            print(f"\n{'=' * 60}")
            print(f"[流式] 开始处理用户 {username} 的消息")
            print(
                f"🔘 个人记忆: {personal_memory_enabled}, 共享记忆: {shared_memory_enabled}"
            )

            memoryos_instance = ensure_user_memoryos(username, project_name)
            if not memoryos_instance and username in memoryos_instances:
                del memoryos_instances[username]
                memoryos_instance = ensure_user_memoryos(username, project_name)

            # 获取用户配置
            user_config = get_user_config(username, project_name)

            # 读取用户额度（决定使用他人配置还是个人配置）
            quota_total = 100000
            quota_used = 0
            try:
                if os.path.exists(cache_path_settings.USER_FILE_PATH):
                    with open(
                        cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8"
                    ) as f:
                        user_data = json.load(f)
                    users = user_data.get("users", [])
                    u = next((x for x in users if x.get("username") == username), None)
                    if u:
                        quota_total = int(u.get("quota_total", 100000) or 100000)
                        quota_used = int(u.get("quota_used", 0) or 0)
            except Exception as e:
                print(f"⚠️ 读取额度失败: {e}")

            # 当额度未满时，优先使用 othersApi.json 中的配置
            use_others = quota_used < quota_total
            others_api_key = None
            others_base_url = None
            if use_others:
                try:
                    others_api_key = os.getenv("OTHER_API_KEY")
                    others_base_url = os.getenv("OTHER_API_BASE")
                except Exception as e:
                    if not others_api_key or not others_base_url:
                        print(f"⚠️ 读取OTHER_API_KEY或OTHER_API_BASE失败: {e}")
                        return

            # 选择最终的 Key/URL：额度未满优先 others，否则使用用户个人配置（或全局config作为兜底）
            final_api_key = None
            final_base_url = None
            if use_others and others_api_key:
                final_api_key = others_api_key
                final_base_url = others_base_url or config.openai_api_base
            else:
                final_api_key = user_config.get("openai_api_key", config.openai_api_key)
                final_base_url = user_config.get(
                    "openai_base_url", config.openai_api_base
                )

            # 仅当额度已满且没有个人Key时阻止
            if quota_used >= quota_total and not user_config.get("openai_api_key"):
                yield f"data: {json.dumps({'error': '额度已用满，请在设置中配置个人 OpenAI API Key'}, ensure_ascii=False)}\n\n"
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
            used_shared_memory_ids = []  # 初始化记忆ID列表
            if shared_memory_enabled:
                try:
                    peers = retrieve_pipeline.get_cached_peers()
                    enhanced_user_profile = UserProfile(
                        user_id=username, profile_text=enhanced_profile_text
                    )
                    print("\n🔍 [流式聊天] 开始检索共享记忆...")
                    print(f"  - 用户: {username}")
                    print(f"  - 消息: {message[:50]}...")
                    print(f"  - 对话ID: {conversation_id}")

                    retrieval_result = retrieve_pipeline.retrieve(
                        user=enhanced_user_profile, task=message, peers=peers, top_k=3
                    )

                    print(f"  - [流式聊天] 检索结果: {retrieval_result}")
                    print(
                        f"  - [流式聊天] 检索到的项目数量: {len(retrieval_result.get('items', []))}"
                    )

                    # 收集使用的共享记忆ID
                    try:
                        selected_ids = [
                            it.get("memory", {}).get("id", "NO_ID_FOUND")
                            for it in retrieval_result.get("items", [])
                            if isinstance(it, dict)
                        ]
                        # 过滤掉无效的ID
                        used_shared_memory_ids = [
                            id for id in selected_ids if id != "NO_ID_FOUND"
                        ]
                        print(f"  - [流式聊天] 选中的记忆ID: {used_shared_memory_ids}")

                        if used_shared_memory_ids:
                            print(
                                f"✅ [流式聊天] 共享记忆已选中ID: {', '.join(used_shared_memory_ids)}"
                            )
                        else:
                            print(
                                "ℹ️ [流式聊天] 共享记忆未选中任何条目（为空或被QC过滤）"
                            )
                    except Exception as log_err:
                        print(f"⚠️ [流式聊天] 收集共享记忆ID失败: {log_err}")

                        traceback.print_exc()

                    if retrieval_result["items"]:
                        shared_memory_context = retrieve_pipeline.build_prompt_blocks(
                            retrieval_result["items"], conversation_id, username
                        )
                except Exception as e:
                    print(f"⚠️ 获取共享记忆失败: {e}")
            else:
                print("共享记忆未开启（shared_memory_enabled=False）")

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
                print("使用融合RAG提示词")
            elif personal_memory_enabled and personal_memory_context:
                prompt = get_fusion_rag_prompt_with_context(
                    message,
                    "",
                    personal_memory_context,
                    enhanced_profile_text,
                    conversation_context,
                )
                print("使用个人记忆RAG提示词")
            elif shared_memory_enabled and shared_memory_context:
                prompt = get_rag_answer_prompt_with_context(
                    message,
                    shared_memory_context,
                    enhanced_profile_text,
                    conversation_context,
                )
                print("使用共享记忆RAG提示词")
            else:
                prompt = get_baseline_answer_prompt_no_profile(
                    message, conversation_context
                )
                print("使用基线提示词")

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

            # 🔧 检查是否启用 MCP 工具调用
            if mcp_enabled:
                print("🛠️ MCP 模式已启用，使用工具调用")
                # 使用 MCP 工具调用的流式生成器
                yield from generate_with_mcp_tools(
                    prompt=prompt,
                    username=username,
                    conversation_id=current_conversation_id,
                    message=message,
                    model=model,
                    shared_memory_enabled=shared_memory_enabled,
                    personal_memory_enabled=personal_memory_enabled,
                    used_shared_memory_ids=used_shared_memory_ids,
                    api_key=final_api_key,  # 传递用户的 API Key
                    base_url=final_base_url,  # 传递用户的 Base URL
                )
                # MCP 模式处理完成，直接返回
                return

            # 普通模式：不使用 MCP 工具
            print("💬 普通模式，不使用工具调用")
            client = OpenAI(
                api_key=final_api_key,
                base_url=final_base_url,
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
            conversation_saved_to_memory = False  # 标记是否已保存到记忆
            chunk_count = 0  # 用于定期保存

            # 定义一个保存函数，用于在任何情况下保存对话
            def save_interrupted_conversation():
                """保存被中断的对话（不保存到记忆，只保存到文件）"""
                nonlocal conversation_saved_to_memory
                # 标记此对话已被中断，不应该保存到记忆
                conversation_saved_to_memory = (
                    True  # 设置为True表示"已处理过"，不保存到记忆
                )

                # 即使没有AI回复，也要保存用户的消息
                response_to_save = (
                    full_response if full_response.strip() else "（回复被用户终止）"
                )
                print(
                    f"💾 保存被中断的对话（不保存到记忆），用户消息: {message[:50]}...，AI回复长度: {len(full_response)} 字符"
                )

                try:
                    # 🚫 被中断的对话不保存到短期记忆和共享记忆，只保存到对话文件
                    # 这样用户可以看到被中断的内容，但不会影响记忆系统

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
                        used_shared_memory_ids=used_shared_memory_ids,
                    )

                    print(
                        f"✅ 中断的消息已保存到对话文件（未保存到记忆），conversation_id: {saved_conversation_id}"
                    )
                    return saved_conversation_id
                except Exception as e:
                    print(f"❌ 保存中断消息失败: {e}")

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
                                print(
                                    f"✅ 检测到流式输出结束，finish_reason: {chunk.choices[0].finish_reason}, 最后内容已发送: {content_sent}"
                                )
                                # 🔥 确保最后一个chunk的内容已经发送后再跳出循环
                                break
                    except (GeneratorExit, StopIteration) as e:
                        # 客户端断开连接
                        print(f"🛑 检测到客户端断开连接: {e}")
                        stream_interrupted = True
                        conversation_saved_to_memory = (
                            True  # 标记为已处理，不保存到记忆
                        )
                        break
                    except Exception as e:
                        print(f"⚠️ 处理流式数据时出错: {e}")
                        continue

            except GeneratorExit:
                # 客户端主动断开连接（AbortController触发）
                print("🛑 客户端主动终止连接（AbortController） - 开始保存操作")
                stream_interrupted = True
                conversation_saved_to_memory = True  # 标记为已处理，不保存到记忆

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
                except Exception:
                    pass
                return

            print(f"✅ 流式输出完成，总长度: {len(full_response)} 字符")
            print(
                f"🔥 最后50个字符: {full_response[-50:] if len(full_response) > 50 else full_response}"
            )

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
                        used_shared_memory_ids=used_shared_memory_ids,
                        update_last_ai_message=True,
                    )
                except Exception as e:
                    print(f"⚠️ 保存对话失败: {e}")

                if used_shared_memory_ids:
                    try:
                        increment_shared_memory_contribution(used_shared_memory_ids)
                    except Exception as e:
                        print(f"⚠️ 累计共享记忆贡献值失败: {e}")

            # 🚀 立即发送完成信号和conversation_id，不要等待其他操作
            active_conversation_id = saved_conversation_id or current_conversation_id
            yield f"data: {json.dumps({'done': True, 'conversation_id': active_conversation_id}, ensure_ascii=False)}\n\n"

            # 然后再做耗时的保存操作（这些操作在后台完成，不影响前端显示）
            # 🚫 如果对话被中断（通过 save_interrupted_conversation 处理），则不保存到记忆
            if not conversation_saved_to_memory and full_response.strip():
                # 保存到个人记忆
                if username in memoryos_instances:
                    try:
                        memoryos_instance = memoryos_instances[username]
                        memoryos_instance.add_memory(message, full_response)
                        print("✅ 对话已保存到短期记忆")

                        # 检测思维链断裂并发送到共享记忆
                        if shared_memory_enabled:
                            try:
                                check_and_store_chain_break_from_memoryos(
                                    username,
                                    memoryos_instance,
                                    conversation_id=active_conversation_id,
                                    project_name=project_name,
                                )
                                print("✅ 思维链检测完成")
                            except Exception as e:
                                print(f"⚠️ 思维链检测失败: {e}")
                        conversation_saved_to_memory = True  # 标记已保存到记忆
                    except Exception as e:
                        print(f"❌ 保存记忆失败: {e}")
                        print(f"错误类型: {type(e).__name__}")

                        print(f"详细错误信息: {traceback.format_exc()}")
            elif conversation_saved_to_memory:
                print("🚫 对话被中断，已跳过保存到记忆")
            else:
                print("⚠️ 流式输出为空，跳过保存操作")

            # 🎯 对话结束后，累计用户额度：每轮 +50
            try:
                if os.path.exists(cache_path_settings.USER_FILE_PATH) and username:
                    with open(
                        cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8"
                    ) as f:
                        user_data = json.load(f)
                    users = user_data.get("users", [])
                    for u in users:
                        if u.get("username") == username:
                            total = int(u.get("quota_total", 100000) or 100000)
                            used = int(u.get("quota_used", 0) or 0)
                            used = min(total, used + 50)
                            u["quota_total"] = total
                            u["quota_used"] = used
                            break
                    user_data["users"] = users
                    with open(
                        cache_path_settings.USER_FILE_PATH, "w", encoding="utf-8"
                    ) as f:
                        json.dump(user_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ 累计额度失败: {e}")

        except Exception as e:
            print(f"❌ 流式生成失败: {e}")

            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/get_chat_conversations", methods=["POST"])
@login_required
def get_chat_conversations_api():
    """获取聊天对话列表"""
    try:
        data = request.get_json()
        username = g.get("current_user") or data.get("username")
        project_name = data.get("project_name", "default_project")

        if not username:
            return jsonify({"success": False, "error": "缺少用户名"})

        conversations = get_chat_conversations(username, project_name)
        return jsonify({"success": True, "conversations": conversations})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_chat_messages", methods=["POST"])
@login_required
def get_chat_messages_api():
    """获取指定对话的消息"""
    try:
        data = request.get_json()
        username = g.get("current_user") or data.get("username")
        conversation_id = data.get("conversation_id")

        if not username or not conversation_id:
            return jsonify({"success": False, "error": "缺少必要参数"})

        conversation = load_conversation_history(username, conversation_id)
        if not conversation:
            return jsonify({"success": False, "error": "对话不存在"})

        return jsonify({"success": True, "conversation": conversation})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/save_chat_user_config", methods=["POST"])
@login_required
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
    try:
        data = request.get_json()
        username = data.get("username")  # 兼容旧字段
        email = (data.get("email") or "").strip()
        password = data.get("password")

        if not password or (not username and not email):
            return jsonify({"success": False, "error": "邮箱或用户名与密码不能为空"})

        # 读取用户配置文件
        if not os.path.exists(cache_path_settings.USER_FILE_PATH):
            return jsonify({"success": False, "error": "用户配置文件不存在"})

        with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        users = user_data.get("users", [])

        # 新增：支持邮箱+密码登录
        if email:
            matched = next(
                (
                    u
                    for u in users
                    if u.get("email") == email and u.get("password") == password
                ),
                None,
            )
            if matched:
                username_val = matched.get("username")
                token = create_jwt({"username": username_val})
                resp = make_response(
                    jsonify(
                        {
                            "success": True,
                            "message": "登录成功",
                            "username": username_val,
                        }
                    )
                )
                return set_jwt_cookie(resp, token)

        # 兼容：原有的用户名+密码登录
        if username:
            for user in users:
                if (
                    user.get("username") == username
                    and user.get("password") == password
                ):
                    token = create_jwt({"username": username})
                    resp = make_response(
                        jsonify(
                            {
                                "success": True,
                                "message": "登录成功",
                                "username": username,
                            }
                        )
                    )
                    return set_jwt_cookie(resp, token)

        return jsonify({"success": False, "error": "账号或密码错误"})

    except Exception as e:
        print(f"登录验证失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/send_login_code", methods=["POST"])
def send_login_code():
    """发送登录验证码到邮箱（仅允许已注册邮箱）"""
    try:
        data = request.get_json()
        email = (data.get("email") or "").strip()

        if not email:
            return jsonify({"success": False, "error": "邮箱不能为空"})

        # 邮箱格式校验
        if "@" not in email or "." not in email.split("@")[1]:
            return jsonify({"success": False, "error": "邮箱格式不正确"})

        # 检查邮箱是否已注册
        if not os.path.exists(cache_path_settings.USER_FILE_PATH):
            return jsonify({"success": False, "error": "邮箱未注册"})

        with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
            users = user_data.get("users", [])
            email_exists = any(u.get("email") == email for u in users)
            if not email_exists:
                return jsonify({"success": False, "error": "邮箱未注册"})

        # 生成6位验证码
        code = "".join(random.choices(string.digits, k=6))
        login_codes[email] = {
            "code": code,
            "expires_at": datetime.now() + timedelta(minutes=5),
        }

        ok = send_email(email, code)
        if not ok:
            return jsonify({"success": False, "error": "发送验证码失败，请稍后重试"})

        print(f"✅ 登录验证码已发送到 {email}，验证码: {code}")
        return jsonify(
            {"success": True, "message": "登录验证码已发送，请在5分钟内使用"}
        )
    except Exception as e:
        print(f"发送登录验证码失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/login_with_code", methods=["POST"])
def login_with_code():
    """通过邮箱验证码登录（返回匹配到的用户名）"""
    try:
        data = request.get_json()
        email = (data.get("email") or "").strip()
        code = (data.get("verification_code") or "").strip()

        if not email or not code:
            return jsonify({"success": False, "error": "邮箱和验证码不能为空"})

        # 校验验证码
        stored = login_codes.get(email)
        if not stored:
            return jsonify({"success": False, "error": "验证码无效或已过期"})

        if datetime.now() > stored["expires_at"]:
            del login_codes[email]
            return jsonify({"success": False, "error": "验证码已过期"})

        if stored["code"] != code:
            return jsonify({"success": False, "error": "验证码错误"})

        # 使用邮箱查找用户
        if not os.path.exists(cache_path_settings.USER_FILE_PATH):
            return jsonify({"success": False, "error": "用户不存在"})

        with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
            users = user_data.get("users", [])
            matched_user = next((u for u in users if u.get("email") == email), None)

        if not matched_user:
            return jsonify({"success": False, "error": "用户不存在"})

        # 一次性验证码，使用后移除
        del login_codes[email]

        username = matched_user.get("username")
        print(f"✅ 邮箱验证码登录成功: {username}")
        token = create_jwt({"username": username})
        resp = make_response(
            jsonify({"success": True, "message": "登录成功", "username": username})
        )
        return set_jwt_cookie(resp, token)
    except Exception as e:
        print(f"验证码登录失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/send_reset_code", methods=["POST"])
def send_reset_code():
    """发送重置密码验证码到已注册邮箱"""
    try:
        data = request.get_json()
        email = (data.get("email") or "").strip()

        if not email:
            return jsonify({"success": False, "error": "邮箱不能为空"})

        if "@" not in email or "." not in email.split("@")[1]:
            return jsonify({"success": False, "error": "邮箱格式不正确"})

        if not os.path.exists(cache_path_settings.USER_FILE_PATH):
            return jsonify({"success": False, "error": "邮箱未注册"})

        with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
            users = user_data.get("users", [])
            email_exists = any(u.get("email") == email for u in users)
            if not email_exists:
                return jsonify({"success": False, "error": "邮箱未注册"})

        code = "".join(random.choices(string.digits, k=6))
        reset_codes[email] = {
            "code": code,
            "expires_at": datetime.now() + timedelta(minutes=5),
        }

        ok = send_email(email, code)
        if not ok:
            return jsonify({"success": False, "error": "发送验证码失败，请稍后重试"})

        print(f"✅ 重置密码验证码已发送到 {email}，验证码: {code}")
        return jsonify({"success": True, "message": "验证码已发送，请在5分钟内使用"})
    except Exception as e:
        print(f"发送重置验证码失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/reset_password", methods=["POST"])
def reset_password():
    """校验验证码并重置该邮箱用户的密码"""
    try:
        data = request.get_json()
        email = (data.get("email") or "").strip()
        code = (data.get("verification_code") or "").strip()
        new_password = (data.get("new_password") or "").strip()
        confirm_password = (data.get("confirm_password") or "").strip()

        if not email or not code or not new_password or not confirm_password:
            return jsonify({"success": False, "error": "邮箱、验证码及新密码不能为空"})

        if new_password != confirm_password:
            return jsonify({"success": False, "error": "两次输入的新密码不一致"})

        if len(new_password) < 6:
            return jsonify({"success": False, "error": "密码长度需至少6位"})

        stored = reset_codes.get(email)
        if not stored:
            return jsonify({"success": False, "error": "验证码无效或已过期"})
        if datetime.now() > stored["expires_at"]:
            del reset_codes[email]
            return jsonify({"success": False, "error": "验证码已过期"})
        if stored["code"] != code:
            return jsonify({"success": False, "error": "验证码错误"})

        if not os.path.exists(cache_path_settings.USER_FILE_PATH):
            return jsonify({"success": False, "error": "用户不存在"})

        with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        users = user_data.get("users", [])
        updated = False
        for u in users:
            if u.get("email") == email:
                u["password"] = new_password
                updated = True
                break

        if not updated:
            return jsonify({"success": False, "error": "用户不存在"})

        user_data["users"] = users
        with open(cache_path_settings.USER_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)

        # 一次性验证码
        del reset_codes[email]

        return jsonify({"success": True, "message": "密码已重置，请使用新密码登录"})
    except Exception as e:
        print(f"重置密码失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/send_verification_code", methods=["POST"])
def send_verification_code():
    """发送验证码到邮箱"""
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()

        if not username:
            return jsonify({"success": False, "error": "用户名不能为空"})

        if not email:
            return jsonify({"success": False, "error": "邮箱不能为空"})

        # 简单的邮箱格式验证
        if "@" not in email or "." not in email.split("@")[1]:
            return jsonify({"success": False, "error": "邮箱格式不正确"})

        # 检查用户名是否已存在
        if os.path.exists(cache_path_settings.USER_FILE_PATH):
            with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
                user_data = json.load(f)
                users = user_data.get("users", [])
                for user in users:
                    if user.get("username") == username:
                        return jsonify({"success": False, "error": "用户名已存在"})
                # 邮箱唯一性校验
                for user in users:
                    if user.get("email") and user.get("email") == email:
                        return jsonify({"success": False, "error": "该邮箱已被注册"})

        # 生成6位随机验证码
        code = "".join(random.choices(string.digits, k=6))

        # 存储验证码（5分钟有效期）
        expires_at = datetime.now() + timedelta(minutes=5)
        verification_codes[email] = {
            "code": code,
            "username": username,
            "expires_at": expires_at,
        }

        # 发送邮件
        success = send_email(email, code)

        if success:
            print(f"✅ 验证码已发送到 {email}, 用户名: {username}, 验证码: {code}")
            return jsonify(
                {
                    "success": True,
                    "message": "验证码已发送到您的邮箱，请查收（5分钟内有效）",
                }
            )
        else:
            return jsonify(
                {"success": False, "error": "验证码发送失败，请检查邮箱配置或稍后重试"}
            )

    except Exception as e:
        print(f"发送验证码失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/register", methods=["POST"])
def register():
    """验证验证码并注册用户"""
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        verification_code = data.get("verification_code", "").strip()
        password = data.get("password", "").strip()

        if not username:
            return jsonify({"success": False, "error": "用户名不能为空"})

        if not email:
            return jsonify({"success": False, "error": "邮箱不能为空"})

        if not verification_code:
            return jsonify({"success": False, "error": "验证码不能为空"})

        if not password:
            return jsonify({"success": False, "error": "密码不能为空"})

        # 简单密码校验（长度≥6）
        if len(password) < 6:
            return jsonify({"success": False, "error": "密码长度需至少6位"})

        # 验证验证码
        if email not in verification_codes:
            return jsonify(
                {"success": False, "error": "验证码已过期或无效，请重新获取"}
            )

        stored_data = verification_codes[email]

        # 检查验证码是否过期
        if datetime.now() > stored_data["expires_at"]:
            del verification_codes[email]
            return jsonify({"success": False, "error": "验证码已过期，请重新获取"})

        # 检查用户名是否匹配
        if stored_data["username"] != username:
            return jsonify({"success": False, "error": "用户名与验证码不匹配"})

        # 验证验证码
        if stored_data["code"] != verification_code:
            return jsonify({"success": False, "error": "验证码错误"})

        # 验证码正确，创建用户

        # 读取现有用户数据
        if os.path.exists(cache_path_settings.USER_FILE_PATH):
            with open(cache_path_settings.USER_FILE_PATH, "r", encoding="utf-8") as f:
                user_data = json.load(f)
        else:
            user_data = {"users": []}

        users = user_data.get("users", [])

        # 再次检查用户名是否已存在（防止并发注册）
        for user in users:
            if user.get("username") == username:
                # 删除已使用的验证码
                del verification_codes[email]
                return jsonify({"success": False, "error": "用户名已存在"})

        # 邮箱唯一性校验
        for user in users:
            if user.get("email") and user.get("email") == email:
                del verification_codes[email]
                return jsonify({"success": False, "error": "该邮箱已被注册"})

        # 创建新用户（使用用户设置的密码）并初始化额度
        new_user = {
            "username": username,
            "password": password,
            "email": email,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quota_total": 100000,
            "quota_used": 0,
        }
        users.append(new_user)
        user_data["users"] = users

        # 保存用户数据
        with open(cache_path_settings.USER_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)

        # 删除已使用的验证码
        del verification_codes[email]

        print(f"✅ 新用户注册成功: {username}, 邮箱: {email}")

        return jsonify(
            {
                "success": True,
                "message": "注册成功！请使用设置的密码登录",
                "username": username,
            }
        )

    except Exception as e:
        print(f"注册失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/chat/users/<username>/<filename>")
@app.route("/chat/<project_name>/users/<username>/<filename>")
@login_required
def serve_user_file(username, filename, project_name="default_project"):
    """提供用户文件服务"""
    # 仅允许本人访问
    if g.get("current_user") != username:
        return jsonify({"error": "无权限"}), 403
    user_dir = os.path.join(
        cache_path_settings.MEMORYOS_DATA_DIR, project_name, "users", username
    )
    if os.path.exists(os.path.join(user_dir, filename)):
        return send_from_directory(user_dir, filename)
    else:
        return jsonify({"error": "文件不存在"}), 404


@app.route("/api/get_used_shared_memories", methods=["POST"])
@login_required
def get_used_shared_memories():
    """获取实际使用的共享记忆API"""
    try:
        data = request.get_json()
        username = g.get("current_user") or data.get("username")
        conversation_id = data.get("conversation_id")
        message_index = data.get("message_index")  # 新增：消息索引
        used_shared_memories = data.get(
            "used_shared_memories", []
        )  # 新增：特定记忆ID列表

        print("\n📊 获取使用的共享记忆请求:")
        print(f"  - 用户名: {username}")
        print(f"  - 对话ID: {conversation_id}")
        print(f"  - 消息索引: {message_index}")
        print(f"  - 特定记忆ID: {used_shared_memories}")

        if not username:
            return jsonify({"success": False, "error": "缺少用户名"})

        if not conversation_id:
            return jsonify({"success": False, "error": "缺少对话ID"})

        # 如果提供了特定的记忆ID列表，直接使用这些ID
        # 注意：used_shared_memories 可能是空列表 []，需要检查是否为 None
        if used_shared_memories is not None and len(used_shared_memories) > 0:
            used_memory_ids = used_shared_memories
            print(f"  - 使用提供的特定记忆ID: {used_memory_ids}")
        else:
            # 从对话文件中获取使用的记忆信息
            conversation_file = os.path.join(
                cache_path_settings.MEMORYOS_DATA_DIR,
                "default_project",
                "users",
                username,
                f"{conversation_id}.json",
            )

            used_memory_ids = []
            if os.path.exists(conversation_file):
                with open(conversation_file, "r", encoding="utf-8") as f:
                    conversation_data = json.load(f)
                    messages = conversation_data.get("messages", [])

                    # 如果提供了消息索引，只获取该消息的记忆
                    if message_index is not None:
                        if message_index < len(messages):
                            message = messages[message_index]
                            if message.get("type") == "assistant" and message.get(
                                "used_shared_memories"
                            ):
                                used_memory_ids = message.get(
                                    "used_shared_memories", []
                                )
                                print(
                                    f"  - 获取第{message_index}条消息的记忆: {used_memory_ids}"
                                )
                        else:
                            print(
                                f"  - 消息索引{message_index}超出范围，总消息数: {len(messages)}"
                            )
                    else:
                        # 从所有assistant消息中收集used_shared_memories
                        for message in messages:
                            if message.get("type") == "assistant" and message.get(
                                "used_shared_memories"
                            ):
                                used_memory_ids.extend(
                                    message.get("used_shared_memories", [])
                                )

            print(f"  - 对话中使用的记忆ID: {used_memory_ids}")
            print(f"  - 对话文件路径: {conversation_file}")
            print(f"  - 对话文件是否存在: {os.path.exists(conversation_file)}")

        if not used_memory_ids:
            print("  - 没有找到使用的记忆信息，返回空结果")
            return jsonify({"success": True, "memories": [], "total": 0})

        # 直接从memory.json文件读取记忆内容
        used_memories = []

        # 读取memory.json文件
        all_memories_data = {}
        if os.path.exists(cache_path_settings.MEMORY_FILE_PATH):
            try:
                with open(
                    cache_path_settings.MEMORY_FILE_PATH, "r", encoding="utf-8"
                ) as f:
                    memory_data = json.load(f)
                    memories_list = memory_data.get("memories", [])
                    for mem in memories_list:
                        all_memories_data[mem.get("id")] = mem
                print(f"  - 从memory.json加载了 {len(all_memories_data)} 个记忆")
            except Exception as e:
                print(f"  - 读取memory.json失败: {e}")

        for memory_id in used_memory_ids:
            # 从memory.json中查找对应的记忆
            if memory_id in all_memories_data:
                memory_data = all_memories_data[memory_id]

                # 获取内容 - 优先使用cot_text，其次使用raw_text
                content = ""
                if memory_data.get("cot_text") and memory_data.get("cot_text").strip():
                    content = memory_data.get("cot_text").strip()
                elif (
                    memory_data.get("raw_text") and memory_data.get("raw_text").strip()
                ):
                    content = memory_data.get("raw_text").strip()
                else:
                    content = "无内容"

                # 获取时间戳
                created_at = memory_data.get("created_at", 0)
                timestamp_str = (
                    datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
                    if created_at
                    else "未知时间"
                )

                # 从memory.json获取focus_query
                focus_query = memory_data.get("focus_query", "")

                # 获取merged_users字段，如果不存在则使用source_user_id
                merged_users = []
                if memory_data.get("meta") and isinstance(
                    memory_data.get("meta"), dict
                ):
                    merged_users = memory_data.get("meta", {}).get("merged_users", [])

                # 如果merged_users为空，使用source_user_id作为fallback
                if not merged_users and memory_data.get("source_user_id"):
                    merged_users = [memory_data.get("source_user_id")]

                used_memories.append(
                    {
                        "id": memory_id,
                        "user_id": memory_data.get("source_user_id", "未知"),
                        "content": content,
                        "focus_query": focus_query,
                        "timestamp": timestamp_str,
                        "created_at": created_at,
                        "merged_users": merged_users,  # 添加merged_users字段
                    }
                )
                print(f"  - 找到记忆: {memory_id}, 内容长度: {len(content)}")
            else:
                print(f"  - 未找到记忆: {memory_id}")

        print(f"  - 返回使用的记忆数量: {len(used_memories)}")

        return jsonify(
            {"success": True, "memories": used_memories, "total": len(used_memories)}
        )

    except Exception as e:
        print(f"❌ 获取使用的共享记忆失败: {e}")

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("🚀 启动Flask应用...")
    print(f"📁 数据目录: {cache_path_settings.MEMORYOS_DATA_DIR}")
    app.run(host="127.0.0.1", port=5002, debug=True)
