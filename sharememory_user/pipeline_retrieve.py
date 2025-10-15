from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from config import Config
from embedding import Embedder
from models import MemoryItem, UserProfile
from storage import JsonStore
from .utils import l2_normalize, softmax, entropy, js_divergence


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
        profile_vec = np.array(self.embed.embed_text(profile_text or ""), dtype=np.float64)
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

    def step_c(self, alpha_i: np.ndarray, p_peer: np.ndarray) -> Tuple[np.ndarray, float]:
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

    def _pre_filter_memories_by_focus(self, memories: List[MemoryItem], query_vec: np.ndarray, top_k: int) -> List[MemoryItem]:
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

    def retrieve(self, user: UserProfile, task: str, peers: List[Peer], top_k: int = 3) -> Dict[str, any]:
        memories = self.store.list_memories()
        Q_i = self._encode_query(user.profile_text, task)

        # Step 1: Pre-filter to get top 10 candidates based on focus_query
        pre_filter_k = 10
        candidate_memories = self._pre_filter_memories_by_focus(memories, Q_i, pre_filter_k)
        
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
        
        results = [
            {
                "rank": int(r + 1),
                "score": float(tilde_alpha[idx]),
                "memory": candidate_memories[idx].to_dict(),
            }
            for r, idx in enumerate(indices)
        ]
        return {
            "items": results,
            "lambda": lambda_i,
            "alpha": alpha_i.tolist(),
            "peer": p_peer.tolist(),
        }

    def build_prompt_blocks(self, items: List[Dict[str, any]]) -> str:
        parts: List[str] = []
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
            if creator_profile:
                parts.append(f"**Created by:** {creator_profile}")
            # parts.append(f"**Content:** {cot}")
            parts.append("**KG:**")
            for e in kg:
                head = e.get("head", "?")
                rel = e.get("relation", "rel")
                tail = e.get("tail", "?")
                parts.append(f"- ({head}, {rel}, {tail})")
            parts.append("")  # Add empty line between memories
            with open('cotkg.txt', 'a+') as f:
                f.write(cot)
                f.write('\n\n--- Knowledge Graph ---\n')
                for e in kg:
                    head = e.get("head", "?")
                    rel = e.get("relation", "rel")
                    tail = e.get("tail", "?")
                    f.write(f"- ({head}, {rel}, {tail})\n")
        return "\n".join(parts)
