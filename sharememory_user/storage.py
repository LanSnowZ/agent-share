from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List

try:
    import orjson as json
except Exception:  # pragma: no cover
    import json  # type: ignore

from .config import Config, ensure_dirs
from .models import MemoryItem, UserProfile


class JsonStore:
    """Simple JSON file storage for users and memories."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        ensure_dirs(cfg)
        if not os.path.exists(cfg.memory_path):
            self._write_json(cfg.memory_path, {"memories": []})
        if not os.path.exists(cfg.users_path):
            self._write_json(cfg.users_path, {"users": []})

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            data = f.read()
        try:
            return json.loads(data)  # type: ignore[attr-defined]
        except Exception:
            # fallback to standard json
            import json as std_json

            with open(path, "r") as f:
                return std_json.load(f)

    @staticmethod
    def _write_json(path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                f.write(json.dumps(obj))  # type: ignore[attr-defined]
        except Exception:
            with open(path, "w") as f:
                json.dump(obj, f)

    def add_user(self, profile: UserProfile) -> None:
        data = self._read_json(self.cfg.users_path)
        users: List[Dict[str, Any]] = data.get("users", [])
        # replace if exists
        users = [u for u in users if u.get("user_id") != profile.user_id]
        users.append(profile.to_dict())
        self._write_json(self.cfg.users_path, {"users": users})

    def get_user(self, user_id: str) -> UserProfile | None:
        data = self._read_json(self.cfg.users_path)
        for u in data.get("users", []):
            if u.get("user_id") == user_id:
                return UserProfile(**u)
        return None

    def list_users(self) -> List[UserProfile]:
        data = self._read_json(self.cfg.users_path)
        return [UserProfile(**u) for u in data.get("users", [])]

    def add_memories(self, items: Iterable[MemoryItem]) -> None:
        data = self._read_json(self.cfg.memory_path)
        memories: List[Dict[str, Any]] = data.get("memories", [])
        new_items = [m.to_dict() for m in items]
        # Print concise logs for each new memory
        for m in new_items:
            mem_id = m.get("id", "<no-id>")
            fq = m.get("focus_query", "")
            src = m.get("source_user_id", "")
            print(f"🆕 新增共享记忆: id={mem_id}, source={src}")
            if fq:
                preview = fq[:160].replace("\n", " ")
                print(f"   ↳ focus_query: {preview}")
        memories.extend(new_items)
        self._write_json(self.cfg.memory_path, {"memories": memories})

    def update_memory(self, item: MemoryItem) -> bool:
        """Finds a memory by ID and updates it."""
        data = self._read_json(self.cfg.memory_path)
        memories: List[Dict[str, Any]] = data.get("memories", [])
        updated = False
        idx_to_move = None
        for i, mem in enumerate(memories):
            if mem.get("id") == item.id:
                memories[i] = item.to_dict()
                idx_to_move = i
                updated = True
                break
        if updated:
            # Move the updated memory to the front (most recently updated on top)
            if idx_to_move is not None and 0 <= idx_to_move < len(memories):
                mem_dict = memories.pop(idx_to_move)
                memories.insert(0, mem_dict)
                print(f"📌 已置顶更新的共享记忆: id={item.id}")
            self._write_json(self.cfg.memory_path, {"memories": memories})
        return updated

    def list_memories(self) -> List[MemoryItem]:
        print(f"\n{'=' * 80}")
        print("📂 正在读取记忆文件:")
        print(f"   文件路径: {self.cfg.memory_path}")
        print(f"   文件名: {os.path.basename(self.cfg.memory_path)}")
        print(f"{'=' * 80}")

        data = self._read_json(self.cfg.memory_path)
        memories_list = [MemoryItem(**m) for m in data.get("memories", [])]

        print(f"✅ 成功读取 {len(memories_list)} 条记忆")
        if memories_list:
            print(f"   - 第一条记忆 ID: {memories_list[0].id}")
            print(f"   - 最后一条记忆 ID: {memories_list[-1].id}")
        print(f"{'=' * 80}\n")

        return memories_list

    def clear_all(self) -> None:
        self._write_json(self.cfg.memory_path, {"memories": []})
        self._write_json(self.cfg.users_path, {"users": []})

    def stats(self) -> Dict[str, Any]:
        return {
            "num_users": len(self.list_users()),
            "num_memories": len(self.list_memories()),
            "updated_at": time.time(),
        }
