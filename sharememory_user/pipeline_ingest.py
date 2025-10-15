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

        qc = self.qc.evaluate(raw_text, source_user_id)
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
                    
                    # Merge CoT and KG
                    merged_cot = self.qc.merge_cot(target_memory.cot_text, qc.cot)
                    merged_kg = self.qc.merge_kg(target_memory.meta.get("kg", []), qc.kg)
                    
                    # Update target memory
                    target_memory.cot_text = merged_cot
                    target_memory.meta["kg"] = merged_kg
                    
                    # Update user metadata
                    merged_users = target_memory.meta.get("merged_users", [target_memory.source_user_id])
                    if source_user_id not in merged_users:
                        merged_users.append(source_user_id)
                    target_memory.meta["merged_users"] = merged_users
                    
                    # Re-embed merged content
                    target_memory.E_m = self.embed.embed_text(merged_cot)
                    
                    self.store.update_memory(target_memory)
                    return target_memory

        # --- Fallback to creating a new memory ---
        E_m = self.embed.embed_text(qc.cot)
        E_q = self.embed.embed_text(qc.query)

        meta_info = {
            "kg": qc.kg,
            "focus_query": qc.query,
            "source_user_profile": user_profile.to_dict(),
        }

        item = MemoryItem(
            id=MemoryItem.build_id(),
            created_at=time.time(),
            source_user_id=source_user_id,
            raw_text=raw_text,
            cot_text=qc.cot,
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
