from __future__ import annotations

import time
import warnings
from typing import List, Optional

import numpy as np

from config import Config
from embedding import Embedder
from llm_qc import LLMQC
from models import MemoryItem, UserProfile
from storage import JsonStore


class IngestPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.store = JsonStore(cfg)
        self.embed = Embedder(cfg)
        self.qc = LLMQC(cfg)

    def ensure_user(self, user_id: str, profile_text: str) -> None:
        profile = UserProfile(user_id=user_id, profile_text=profile_text)
        self.store.add_user(profile)

    def ingest_dialog(self, source_user_id: str, raw_text: str) -> Optional[MemoryItem]:
        user_profile = self.store.get_user(source_user_id)
        if not user_profile:
            warnings.warn(f"User profile for {source_user_id} not found. Skipping ingestion.")
            return None

        # Build focus_query strictly from user questions within this dialog segment
        user_query_only = self.qc.extract_query_from_user_questions(raw_text)
        # Run QC on full text for include/scores/cot/kg
        qc = self.qc.evaluate(raw_text, source_user_id)
        # Override qc.query to ensure focus_query derives from user-only questions
        if user_query_only:
            qc.query = user_query_only
        if not qc.include:
            return None

        # --- New Merge Logic ---
        existing_memories = self.store.list_memories()
        if existing_memories:
            # 1. Retrieve top_k candidates based on query similarity
            new_query_vec = np.array(self.embed.embed_text(qc.query))
            
            candidates = []
            for mem in existing_memories:
                if mem.E_q:
                    score = np.dot(new_query_vec, np.array(mem.E_q))
                    candidates.append((score, mem))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_k_memories = [mem for _, mem in candidates[:self.cfg.merge_top_k]]

            if top_k_memories:
                # 2. LLM-based matching
                base_query = qc.query
                candidate_queries = [m.meta.get("focus_query", "") for m in top_k_memories]
                matches = self.qc.match_queries(base_query, candidate_queries)
                
                matched_memories = [mem for mem, is_match in zip(top_k_memories, matches) if is_match]

                if matched_memories:
                    # 3. Merge with the best match
                    target_memory = matched_memories[0]
                    old_fq_preview = (target_memory.focus_query or "")[:100]
                    new_fq_preview = (qc.query or "")[:100]
                    print("🔗 检测到同一话题链，合并到已有记忆:")
                    print(f"   ↳ target_id={target_memory.id}")
                    print(f"   ↳ old_focus_query={old_fq_preview}")
                    print(f"   ↳ new_focus_query={new_fq_preview}")
                    
                    # Merge raw dialog text first（以原始对话内容为主）
                    merged_raw = (target_memory.raw_text + "\n\n" + raw_text).strip() if target_memory.raw_text else raw_text

                    # Merge CoT and KG（作为附加信息保留）
                    merged_cot = self.qc.merge_cot(target_memory.cot_text, qc.cot)
                    merged_kg = self.qc.merge_kg(target_memory.meta.get("kg", []), qc.kg)
                    
                    # Merge focus_query (综合新旧两个 focus_query)
                    old_query = target_memory.focus_query or target_memory.meta.get("focus_query", "")
                    new_query = qc.query
                    merged_query = self.qc.merge_query(old_query, new_query)
                    
                    # Update target memory
                    target_memory.raw_text = merged_raw
                    target_memory.cot_text = merged_cot
                    target_memory.meta = target_memory.meta or {}
                    target_memory.meta.setdefault("contribution_score", 0)
                    target_memory.meta["kg"] = merged_kg
                    
                    # Update focus_query to the merged one
                    target_memory.focus_query = merged_query
                    target_memory.meta["focus_query"] = merged_query
                    
                    # Update timestamp to current time (treated as a new memory)
                    target_memory.created_at = time.time()
                    
                    # Update user metadata
                    merged_users = target_memory.meta.get("merged_users", [target_memory.source_user_id])
                    if source_user_id not in merged_users:
                        merged_users.append(source_user_id)
                    target_memory.meta["merged_users"] = merged_users
                    
                    # Re-embed using merged raw dialog text（检索向量来自原始对话）
                    target_memory.E_m = self.embed.embed_text(merged_raw)
                    
                    # Re-embed query vector with merged focus_query
                    target_memory.E_q = self.embed.embed_text(merged_query)
                    
                    # Print merged focus_query
                    merged_fq_preview = merged_query[:160].replace("\n", " ")
                    print(f"   ↳ merged_focus_query={merged_fq_preview}")
                    
                    self.store.update_memory(target_memory)
                    return target_memory

        # --- Fallback to creating a new memory ---
        print("🧩 未发现可合并的候选，创建新的对话链记忆")
        print(f"   ↳ focus_query={qc.query[:160]}")
        # 以原始对话文本生成检索向量
        E_m = self.embed.embed_text(raw_text)
        E_q = self.embed.embed_text(qc.query)

        meta_info = {
            "kg": qc.kg,
            "focus_query": qc.query,
            "contribution_score": 0,
            # "source_user_profile": user_profile.to_dict(),
        }

        item = MemoryItem(
            id=MemoryItem.build_id(),
            created_at=time.time(),
            source_user_id=source_user_id,
            raw_text=raw_text,
            cot_text=qc.cot,  # 作为附加字段保留（检索基于 raw_text）
            focus_query=qc.query,  # Assign focus_query here
            E_m=E_m,
            E_q=E_q,
            meta=meta_info,
            phi_m=None,
        )
        self.store.add_memories([item])
        return item

    def ingest_many(self, source_user_id: str, texts: List[str]) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        for t in texts:
            it = self.ingest_dialog(source_user_id, t)
            if it is not None:
                items.append(it)
        return items
