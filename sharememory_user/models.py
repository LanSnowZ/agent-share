from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time


@dataclass
class UserProfile:
    user_id: str
    profile_text: str
    # Structured dimensions grouped by three major categories -> { subDimensionCN: levelCN }
    profile_dimensions: Optional[Dict[str, Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KGEdge:
    head: str
    relation: str
    tail: str
    evidence: Optional[str] = None
    source_user_id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryItem:
    id: str
    created_at: float
    source_user_id: str
    raw_text: str
    cot_text: str
    focus_query: str  # Added field for pre-filtering
    E_m: List[float]
    meta: Dict[str, Any]
    phi_m: Optional[List[float]] = None
    E_q: Optional[List[float]] = None

    @staticmethod
    def build_id(prefix: str = "mem") -> str:
        return f"{prefix}_{int(time.time()*1000)}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QCResult:
    include: bool
    scores: Dict[str, float]
    cot: str
    kg: List[Dict[str, Any]]
    query: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
