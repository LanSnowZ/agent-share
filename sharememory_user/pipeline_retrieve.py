from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .config import Config
from .embedding import Embedder
from .llm_qc import LLMQC
from .models import MemoryItem, UserProfile
from .storage import JsonStore
from .utils import entropy, js_divergence, l2_normalize, softmax


@dataclass
class Peer:
    user_id: str
    profile_text: str


class RetrievePipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.store = JsonStore(cfg)
        self.embed = Embedder(cfg)
        self._peer_embedding_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._cached_peers: List[Peer] = []
        self.llm = LLMQC(cfg)

    def precompute_peer_embeddings(self, peers: List[Peer]):
        """Precomputes and caches query and state vectors for a list of peers."""
        print("Pre-computing and caching peer embeddings for performance...")
        self._cached_peers = peers
        for p in tqdm(peers, desc="Caching Peer Embeddings"):
            if p.user_id not in self._peer_embedding_cache:
                Q_j = self._encode_query(p.profile_text, "")
                H_j = self._encode_state(p.profile_text)
                self._peer_embedding_cache[p.user_id] = {"Q_j": Q_j, "H_j": H_j}

    def get_cached_peers(self) -> List[Peer]:
        """Returns the list of peers whose embeddings have been cached."""
        return self._cached_peers

    def _encode_query(self, profile_text: str, task: str) -> np.ndarray:
        """
        Encode query by separately encoding profile and task, then fusing them.
        This prevents long profiles from diluting short task queries.
        """
        # If no task is provided, use profile only (for peer encoding)
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

    def _encode_state(self, profile_text: str) -> np.ndarray:
        text = profile_text or ""
        vec = np.array(self.embed.embed_text(text), dtype=np.float64)
        return l2_normalize(vec)

    def _stack_memory_matrix(self, memories: List[MemoryItem]) -> np.ndarray:
        if not memories:
            return np.zeros((0, self.cfg.embed_dimension), dtype=np.float64)
        mat = np.array([m.E_m for m in memories], dtype=np.float64)
        # assume already normalized; re-normalize just in case
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        return mat

    def step_a(self, Q_i: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = E @ Q_i  # (M,)
        alpha = softmax(scores)
        Z_B = (alpha[:, None] * E).sum(axis=0)
        return alpha, Z_B

    def step_b(self, Q_i: np.ndarray, peers: List[Peer], E: np.ndarray) -> np.ndarray:
        if not peers:
            M = E.shape[0]
            return np.ones(M) / max(1, M)
        alphas = []
        H_list = []

        # Check if all required peer embeddings are in the cache
        can_use_cache = all(p.user_id in self._peer_embedding_cache for p in peers)

        for p in peers:
            if can_use_cache:
                peer_embeds = self._peer_embedding_cache[p.user_id]
                Q_j = peer_embeds["Q_j"]
                H_j = peer_embeds["H_j"]
            else:
                # Fallback for safety, though with pre-computation it shouldn't be hit in eval
                Q_j = self._encode_query(p.profile_text, "")
                H_j = self._encode_state(p.profile_text)

            alpha_j, _ = self.step_a(Q_j, E)
            alphas.append(alpha_j)
            H_list.append(H_j)

        alphas_mat = np.stack(alphas, axis=0)  # (J, M)
        H = np.stack(H_list, axis=0)  # (J, d)
        # trust weights beta over peers
        beta_scores = H @ Q_i  # (J,)
        beta = softmax(beta_scores)
        p_peer = (beta[:, None] * alphas_mat).sum(axis=0)
        return p_peer

    def step_c(
        self, alpha_i: np.ndarray, p_peer: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        M = alpha_i.shape[0]
        H_self = entropy(alpha_i)
        H_peer = entropy(p_peer)
        H_max = np.log(max(M, 1))
        c_self = 1.0 - (H_self / (H_max + 1e-12))
        c_peer = 1.0 - (H_peer / (H_max + 1e-12))
        delta_c = c_self - c_peer
        s = 1.0 - js_divergence(alpha_i, p_peer) / (np.log(2) + 1e-12)
        z = self.cfg.kappa * (1.0 - s) * delta_c + self.cfg.bias_b
        lambda_i = 1.0 / (1.0 + np.exp(-z))
        tilde_alpha = lambda_i * alpha_i + (1.0 - lambda_i) * p_peer
        return tilde_alpha, float(lambda_i)

    def _pre_filter_memories_by_focus(
        self, memories: List[MemoryItem], query_vec: np.ndarray, top_k: int
    ) -> List[MemoryItem]:
        """Pre-filters memories based on cosine similarity between query and memory's focus_query."""
        if not memories or top_k <= 0:
            return []

        focus_queries = [m.focus_query for m in memories]
        focus_vectors = np.array(self.embed.embed_many(focus_queries), dtype=np.float64)
        focus_vectors = l2_normalize(focus_vectors)

        similarities = focus_vectors @ query_vec

        # Get top_k indices, ensuring we don't go out of bounds
        num_memories = len(memories)
        k = min(top_k, num_memories)

        # Get the indices of the top k similarities
        top_k_indices = np.argsort(-similarities)[:k]

        return [memories[i] for i in top_k_indices]

    def retrieve(
        self, user: UserProfile, task: str, peers: List[Peer], top_k: int = 5
    ) -> Dict[str, any]:
        memories = self.store.list_memories()
        Q_i = self._encode_query(user.profile_text, task)

        # Step 1: Pre-filter to get top 10 candidates based on focus_query
        pre_filter_k = 20
        candidate_memories = self._pre_filter_memories_by_focus(
            memories, Q_i, pre_filter_k
        )

        # If no candidates, return empty
        if not candidate_memories:
            return {"items": [], "lambda": 0.0, "alpha": [], "peer": []}

        # Step 2: Run the attention mechanism on the pre-filtered candidates
        E = self._stack_memory_matrix(candidate_memories)
        if E.shape[0] == 0:
            return {"items": [], "lambda": 0.0, "alpha": [], "peer": []}

        alpha_i, _ = self.step_a(Q_i, E)
        p_peer = self.step_b(Q_i, peers, E)
        tilde_alpha, lambda_i = self.step_c(alpha_i, p_peer)

        order = np.argsort(-tilde_alpha)

        # Final top_k selection from the candidates
        indices = order[:top_k]

        # LLM-based usefulness filtering (batch): collect all focus_queries, call LLM once, filter indices
        topk_focus_queries = [candidate_memories[idx].focus_query for idx in indices]
        # Log the memory IDs that are being sent to LLM usefulness judgement
        try:
            ids_for_llm = [candidate_memories[idx].id for idx in indices]
            print("\n🔎 [retrieve] 送入 LLM 有用性判断的记忆ID: " + ", ".join(ids_for_llm))
        except Exception:
            pass
        useful_flags = self.llm.are_focus_queries_useful(task, topk_focus_queries)

        filtered_indices: List[int] = []
        for i, idx in enumerate(indices):
            if i < len(useful_flags) and useful_flags[i]:
                filtered_indices.append(idx)

        results = [
            {
                "rank": int(r + 1),
                "score": float(tilde_alpha[idx]),
                "memory": candidate_memories[idx].to_dict(),
            }
            for r, idx in enumerate(filtered_indices)
        ]
        return {
            "items": results,
            "lambda": lambda_i,
            "alpha": alpha_i.tolist(),
            "peer": p_peer.tolist(),
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
