from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import Config
from .embedding import Embedder
from .llm_qc import LLMQC
from .models import MemoryItem, UserProfile
from .storage import JsonStore
from .utils import l2_normalize


@dataclass
class Peer:
    """Peer class for compatibility, but not used in hybrid retrieval strategy."""
    user_id: str
    profile_text: str


class RetrievePipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.store = JsonStore(cfg)
        self.embed = Embedder(cfg)
        self.llm = LLMQC(cfg)

    def get_cached_peers(self) -> List[Peer]:
        """Returns empty list for compatibility with existing code."""
        return []

    def _encode_query(self, profile_text: str, task: str) -> np.ndarray:
        """
        Encode query by separately encoding profile and task, then fusing them.
        This prevents long profiles from diluting short task queries.
        """
        # If no task is provided, use profile only
        if not task or task.strip() == "":
            text = profile_text or ""
            vec = np.array(self.embed.embed_text(text), dtype=np.float64)
            return l2_normalize(vec)

        # Separately encode profile and task
        profile_vec = np.array(
            self.embed.embed_text(profile_text or ""), dtype=np.float64
        )
        task_vec = np.array(self.embed.embed_text(task), dtype=np.float64)

        # Normalize both vectors
        profile_vec = l2_normalize(profile_vec)
        task_vec = l2_normalize(task_vec)

        # Weighted fusion: use configurable weights
        task_weight = self.cfg.task_weight
        profile_weight = self.cfg.profile_weight

        # Ensure weights sum to 1.0 for proper normalization
        total_weight = task_weight + profile_weight
        if total_weight > 0:
            task_weight = task_weight / total_weight
            profile_weight = profile_weight / total_weight
        else:
            task_weight, profile_weight = 0.7, 0.3  # fallback

        fused_vec = task_weight * task_vec + profile_weight * profile_vec
        return l2_normalize(fused_vec)

    def _hybrid_retrieve_with_focus_and_cot(
        self, memories: List[MemoryItem], query_vec: np.ndarray, top_k: int
    ) -> List[MemoryItem]:
        """
        综合检索：基于 focus_query 和 COT 的加权相似度召回候选记忆
        
        Args:
            memories: 所有记忆项
            query_vec: 查询向量
            top_k: 返回的top-K数量
            
        Returns:
            召回的记忆列表（按综合分数排序）
        """
        if not memories or top_k <= 0:
            return []

        # 1. 计算 focus_query 相似度
        focus_queries = [m.focus_query for m in memories]
        focus_vectors = np.array(self.embed.embed_many(focus_queries), dtype=np.float64)
        # L2 normalize each row (each vector)
        focus_vectors = focus_vectors / (np.linalg.norm(focus_vectors, axis=1, keepdims=True) + 1e-12)
        focus_similarities = focus_vectors @ query_vec

        # 2. 计算 COT 相似度
        cot_texts = [m.cot_text for m in memories]
        cot_vectors = np.array(self.embed.embed_many(cot_texts), dtype=np.float64)
        # L2 normalize each row (each vector)
        cot_vectors = cot_vectors / (np.linalg.norm(cot_vectors, axis=1, keepdims=True) + 1e-12)
        cot_similarities = cot_vectors @ query_vec

        # 3. 加权融合
        # 归一化权重
        focus_weight = self.cfg.focus_query_weight
        cot_weight = self.cfg.cot_weight
        total_weight = focus_weight + cot_weight
        if total_weight > 0:
            focus_weight = focus_weight / total_weight
            cot_weight = cot_weight / total_weight
        
        hybrid_scores = focus_weight * focus_similarities + cot_weight * cot_similarities

        # 4. 排序并取 top-K
        num_memories = len(memories)
        k = min(top_k, num_memories)
        top_k_indices = np.argsort(-hybrid_scores)[:k]

        # 5. 日志输出前10个
        print("\n" + "="*80)
        print("📊 [粗召回阶段] 综合检索结果 (focus_query + COT)")
        print(f"   总记忆数: {num_memories}, 召回数: {k}")
        print(f"   权重配置: focus_query={focus_weight:.2f}, COT={cot_weight:.2f}")
        print("-"*80)
        display_count = min(10, len(top_k_indices))
        for rank, idx in enumerate(top_k_indices[:display_count], start=1):
            memory = memories[idx]
            hybrid_score = float(hybrid_scores[idx])
            focus_score = float(focus_similarities[idx])
            cot_score = float(cot_similarities[idx])
            focus_preview = memory.focus_query[:50].replace("\n", " ") if memory.focus_query else "无"
            print(f"  {rank:2d}. ID: {memory.id:20s} | 综合分数: {hybrid_score:.4f}")
            print(f"       └─ Focus分数: {focus_score:.4f} | COT分数: {cot_score:.4f}")
            print(f"       └─ Focus预览: {focus_preview}")
        print("="*80 + "\n")

        return [memories[i] for i in top_k_indices]

    def retrieve(
        self, user: UserProfile, task: str, peers: List[Peer], top_k: int = 5
    ) -> Dict[str, any]:
        """
        两步检索策略：
        1. 综合检索：基于 focus_query + COT 的加权相似度召回候选
        2. LLM判断：使用LLM过滤出真正有用的记忆
        
        Args:
            user: 用户档案
            task: 当前任务/查询
            peers: 同伴列表（为了接口兼容保留，但不使用）
            top_k: 最终返回的记忆数量
            
        Returns:
            包含检索结果的字典，格式保持兼容
        """
        memories = self.store.list_memories()
        Q_i = self._encode_query(user.profile_text, task)

        # ============ 第一步：综合检索（focus_query + COT） ============
        recall_k = self.cfg.hybrid_recall_k
        candidate_memories = self._hybrid_retrieve_with_focus_and_cot(
            memories, Q_i, recall_k
        )

        # 如果没有候选，直接返回空
        if not candidate_memories:
            return {"items": [], "lambda": 0.0, "alpha": [], "peer": []}

        # ============ 第二步：LLM 有用性判断 ============
        # 从候选中选择前 top_k*2 个送入LLM判断（避免过滤后数量不足）
        llm_input_k = min(len(candidate_memories), top_k * 2)
        llm_candidates = candidate_memories[:llm_input_k]
        
        # 提取 focus_queries 用于LLM判断
        focus_queries_for_llm = [m.focus_query for m in llm_candidates]
        
        # 日志：送入LLM的记忆ID
        print("\n" + "="*80)
        print(f"🤖 [LLM判断阶段] 送入 {llm_input_k} 个候选记忆进行有用性判断")
        print("-"*80)
        try:
            ids_for_llm = [m.id for m in llm_candidates]
            for i, mem_id in enumerate(ids_for_llm, 1):
                print(f"  {i:2d}. ID: {mem_id:20s}")
        except Exception:
            pass
        print("="*80 + "\n")
        
        # 调用LLM批量判断
        useful_flags = self.llm.are_focus_queries_useful(task, focus_queries_for_llm)

        # 过滤出有用的记忆
        filtered_memories = []
        for i, (mem, is_useful) in enumerate(zip(llm_candidates, useful_flags)):
            if is_useful:
                filtered_memories.append(mem)

        # 日志：LLM判断结果
        print("\n" + "="*80)
        print(f"✅ [LLM判断结果] {len(filtered_memories)}/{llm_input_k} 个记忆被判定为有用")
        print("-"*80)
        for i, mem in enumerate(filtered_memories[:10], 1):
            focus_preview = mem.focus_query[:50].replace("\n", " ") if mem.focus_query else "无"
            print(f"  {i:2d}. ID: {mem.id:20s}")
            print(f"       └─ Focus: {focus_preview}")
        print("="*80 + "\n")

        # 返回最终的 top_k 结果
        final_k = min(top_k, len(filtered_memories))
        results = [
            {
                "rank": int(r + 1),
                "score": 1.0 - (r / max(len(filtered_memories), 1)),  # 简单的递减分数
                "memory": filtered_memories[r].to_dict(),
            }
            for r in range(final_k)
        ]
        
        # 返回格式保持兼容，但lambda/alpha/peer在新策略中不使用
        return {
            "items": results,
            "lambda": 0.0,  # 新策略不使用lambda
            "alpha": [],    # 新策略不使用alpha
            "peer": [],     # 新策略不使用peer
        }

    def build_prompt_blocks(
        self,
        items: List[Dict[str, any]],
        conversation_id: str = None,
        username: str = None,
    ) -> str:
        parts: List[str] = []
        selected_ids: List[str] = []
        for i, it in enumerate(items, start=1):
            mem = it["memory"]
            cot = mem.get("cot_text", "")
            focus_query = mem.get("focus_query", "")
            source_user_id = mem.get("source_user_id", "")
            kg = mem.get("meta", {}).get("kg", [])
            # print(cot)
            # print(kg)
            print("##############################################")
            print(f"🔍 build_prompt_blocks - Memory #{i}")
            print(f"  - ID: {mem.get('id', 'NO_ID_FOUND')}")
            print(f"  - Type of mem: {type(mem)}")
            print(f"  - Keys in mem: {list(mem.keys())}")
            print("##############################################")
            # Record selected memory id for final summary print
            memory_id = mem.get("id", "NO_ID_FOUND")
            selected_ids.append(memory_id)
            print(f"  - 添加记忆ID到selected_ids: {memory_id}")
            # Get the static profile of the memory creator
            creator_profile = ""
            if source_user_id:
                creator_user = self.store.get_user(source_user_id)
                if creator_user:
                    creator_profile = creator_user.profile_text

            # Build the memory block with focus_query and creator profile
            parts.append(f"### Memory #{i}")
            if focus_query:
                parts.append(f"**Focus Query:** {focus_query}")
            # if creator_profile:
            #     parts.append(f"**Created by:** {creator_profile}")
            # parts.append(f"**Content:** {cot}")
            parts.append("**KG:**")
            for e in kg:
                head = e.get("head", "?")
                rel = e.get("relation", "rel")
                tail = e.get("tail", "?")
                parts.append(f"- ({head}, {rel}, {tail})")
            parts.append("")  # Add empty line between memories
            with open("local/cotkg.txt", "a+") as f:
                f.write(cot)
                f.write("\n\n--- Knowledge Graph ---\n")
                for e in kg:
                    head = e.get("head", "?")
                    rel = e.get("relation", "rel")
                    tail = e.get("tail", "?")
                    f.write(f"- ({head}, {rel}, {tail})\n")
        # Final concise log of memory IDs added to the prompt
        if selected_ids:
            try:
                print(f"✅ 最终加入提示词的共享记忆ID: {', '.join(selected_ids)}")
            except Exception:
                # Fallback to avoid any unexpected printing errors
                print("✅ 最终加入提示词的共享记忆ID:", selected_ids)

        # 记忆ID的保存现在通过save_chat_conversation函数完成
        print("\n🔧 [build_prompt_blocks] 记忆ID将通过save_chat_conversation函数保存:")
        print(f"  - selected_ids: {selected_ids}")
        print(f"  - conversation_id: {conversation_id}")
        print(f"  - username: {username}")

        return "\n".join(parts)

    def _save_used_memories_to_conversation(
        self, conversation_id: str, memory_ids: List[str], username: str
    ) -> None:
        """保存对话中使用的共享记忆ID和focus_query"""
        import json
        import os

        try:
            print("\n🔧 [pipeline_retrieve] 开始保存使用的记忆ID:")
            print(f"  - 对话ID: {conversation_id}")
            print(f"  - 用户名: {username}")
            print(f"  - 记忆ID列表: {memory_ids}")

            # 构建对话文件路径
            # 从当前文件路径: /root/autodl-tmp/service/agent-share/sharememory_user/pipeline_retrieve.py
            # 需要到达: /root/autodl-tmp/service/agent-share/eval/memoryos_data
            # 当前文件: __file__ = /root/autodl-tmp/service/agent-share/sharememory_user/pipeline_retrieve.py
            # 上一级: os.path.dirname(__file__) = /root/autodl-tmp/service/agent-share/sharememory_user
            # 上两级: os.path.dirname(os.path.dirname(__file__)) = /root/autodl-tmp/service/agent-share
            # 目标: /root/autodl-tmp/service/agent-share/eval/memoryos_data
            MEMORYOS_DATA_DIR = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "eval", "memoryos_data"
            )
            conversation_file = os.path.join(
                MEMORYOS_DATA_DIR,
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

                # 获取所有记忆，用于查找focus_query
                all_memories = self.store.list_memories()
                memory_id_to_focus_query = {}
                for memory in all_memories:
                    memory_id_to_focus_query[memory.id] = memory.focus_query

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
                            f"✅ [pipeline_retrieve] 保存记忆ID: {memory_id}, focus_query: {focus_query[:50]}..."
                        )
                    else:
                        print(f"⚠️ [pipeline_retrieve] 记忆ID已存在，跳过: {memory_id}")

                # 保存更新后的对话数据
                with open(conversation_file, "w", encoding="utf-8") as f:
                    json.dump(conversation_data, f, ensure_ascii=False, indent=2)

                print(
                    f"✅ [pipeline_retrieve] 已保存使用的记忆ID和focus_query到对话: {conversation_id}"
                )
            else:
                print(f"⚠️ [pipeline_retrieve] 对话文件不存在: {conversation_file}")
        except Exception as e:
            print(f"⚠️ [pipeline_retrieve] 保存使用的记忆ID失败: {e}")
            import traceback

            traceback.print_exc()
