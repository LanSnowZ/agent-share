from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Tuple

try:
    import orjson as json
except ImportError:
    import json  # type: ignore

from .config import Config
from .models import QCResult, KGEdge

# 直接定义 prompts，避免模块导入冲突
QC_SYSTEM_PROMPT = """你是一名严谨的助手，需要评估一段对话是否包含可复用、足够深入且聚焦的经验。

**评估维度（0.0 - 3.0）**：
1.  **reusability（可复用性）**：是否沉淀了可迁移的方法、模板或避坑清单？
2.  **depth（深度）**：是否包含非表面的推理、权衡或边界条件？
3.  **focus（聚焦度）**：是否围绕单一核心任务展开？

**纳入规则**：当 `reusability >= 2.0` 且 `depth >= 2.0` 且 `focus >= 1.5` 时判定为纳入。

**输出格式**：仅输出一个 JSON 对象，前后不得附加其它文本。
- 纳入：`{"include": true, "scores": {"reusability": <float>, "depth": <float>, "focus": <float>}}`
- 不纳入：`{"include": false, "scores": {"reusability": <float>, "depth": <float>, "focus": <float>}}`
"""

COT_EXTRACTION_SYSTEM_PROMPT = """你是技术讨论总结专家。
请从以下文本中提炼核心推理过程，产出简洁的结构化大纲（Chain-of-Thought 风格）。
强调可复用步骤，保持精炼，聚焦关键点。

**重要要求**：
- 不要泄露任何人物的身份信息（如姓名、用户名、具体身份等）
- 只提取抽象的经验、方法、步骤和知识
- 专注于技术内容本身，避免涉及个人隐私信息

**输出格式**：仅输出一个 JSON 对象。
`{"cot": "<string: 结构化的 COT 大纲>"}`
"""

KG_EXTRACTION_SYSTEM_PROMPT = """你是知识图谱专家。
请从以下文本中抽取有意义的实体与关系，并以三元组表示。
聚焦技术实体、工具与概念。

**重要要求**：
- 不要泄露任何人物的身份信息（如姓名、用户名、具体身份等）
- 只提取技术相关的实体和关系，避免涉及个人隐私信息
- 专注于技术内容本身，提取抽象的知识结构

**输出格式**：仅输出一个 JSON 对象，包含三元组列表。
`{"kg": [{"head": "<实体>", "relation": "<关系>", "tail": "<实体>"}]}`
"""

QUERY_EXTRACTION_SYSTEM_PROMPT = """你是查询归纳专家。
请分析以下文本，抽取其所针对的核心问题（或任务）。
问题需简洁，能概括主要意图。

**重要要求**：
- 不要泄露任何人物的身份信息（如姓名、用户名、具体身份等）
- 只提取抽象的技术问题或任务，避免涉及个人隐私信息
- 专注于技术内容本身，确保查询的通用性和可复用性

**输出格式**：仅输出一个 JSON 对象。
`{"query": "<string: 抽取出的查询>"}`
"""

QUERY_MATCHING_SYSTEM_PROMPT = """你是查询语义匹配专家。
判断哪些候选查询在语义上与基准查询一致，并返回布尔列表。

**输出格式**：仅输出一个 JSON 对象。
`{"matches": [<bool>, <bool>, ...]}`
"""

# Decide whether memories' focus_queries are genuinely useful to answer the user's current query (batch version)
FOCUS_USEFULNESS_BATCH_SYSTEM_PROMPT = """你是一名助手，判断哪些记忆的 focus_query 能切实帮助回答当前用户问题。

规则：
- 仅当某个 focus_query 直接指向同一任务，或可解决用户问题的关键子问题时，标记 useful=true；
- 若其泛泛而谈、无关或无法提升答案质量，则标记 useful=false；
- 请从严判定，对边缘匹配倾向于给出 false。

仅输出 JSON（顺序与输入 focus_queries 保持一致）：
{"useful": [<boolean>, <boolean>, ...]}
"""

COT_MERGING_SYSTEM_PROMPT = """你是结构化推理合并专家。
请将以下两份 Chain-of-Thought 总结合并为一份连贯、完整的总结。

**重要要求**：
- 不要泄露任何人物的身份信息（如姓名、用户名、具体身份等）
- 只提取和合并抽象的经验、方法、步骤和知识
- 专注于技术内容本身，避免涉及个人隐私信息
- 确保合并后的内容保持通用性和可复用性

**输出格式**：仅输出一个 JSON 对象。
`{"merged_cot": "<string: 合并后的总结>"}`
"""

KG_MERGING_SYSTEM_PROMPT = """你是知识图谱合并专家。
请将以下两份知识图谱列表进行合并，去重并整合信息。

**重要要求**：
- 不要泄露任何人物的身份信息（如姓名、用户名、具体身份等）
- 只合并技术相关的实体和关系，避免涉及个人隐私信息
- 专注于技术内容本身，确保合并后的知识图谱保持通用性

**输出格式**：仅输出一个 JSON 对象。
`{"merged_kg": [{"head": "<实体>", "relation": "<关系>", "tail": "<实体>"}]}`
"""

QUERY_MERGING_SYSTEM_PROMPT = """你是查询合并专家。
请将以下两个 focus_query 合并为一个新的、更全面的查询描述。
新的查询应该涵盖两个查询的核心意图，去重并保持简洁。

**重要要求**：
- 不要泄露任何人物的身份信息（如姓名、用户名、具体身份等）
- 只合并抽象的技术问题或任务，避免涉及个人隐私信息
- 专注于技术内容本身，确保合并后的查询保持通用性和可复用性

**输出格式**：仅输出一个 JSON 对象。
`{"merged_query": "<string: 合并后的查询>"}`
"""

def get_user_prompt(text: str) -> str:
    return f"**待评估文本：**\n{text}"

def get_query_matching_prompt(base_query: str, candidate_queries: List[str]) -> str:
    candidates = "\n".join([f"{i+1}. {q}" for i, q in enumerate(candidate_queries)])
    return f"**基准查询：**\n{base_query}\n\n**候选查询：**\n{candidates}\n\n哪些候选与基准查询匹配？"

def get_cot_merging_prompt(cot1: str, cot2: str) -> str:
    return f"**COT 1：**\n{cot1}\n\n**COT 2：**\n{cot2}\n\n请合并以上两份总结："

def get_kg_merging_prompt(kg1: List[Dict], kg2: List[Dict]) -> str:
    return f"**KG 1：**\n{json.dumps(kg1)}\n\n**KG 2：**\n{json.dumps(kg2)}\n\n请合并上述知识图谱："

def get_query_merging_prompt(query1: str, query2: str) -> str:
    return f"**Query 1：**\n{query1}\n\n**Query 2：**\n{query2}\n\n请合并以上两个查询："


class LLMQC:
    """LLM QC and post-processing with OpenAI client and heuristic fallback."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._provider = cfg.llm_provider
        self._client = None
        if self._provider == "openai" and cfg.openai_api_key and "YOUR_API_KEY" not in cfg.openai_api_key:
            try:
                from openai import OpenAI

                api_key = cfg.openai_api_key
                base_url = cfg.openai_api_base if cfg.openai_api_base and "YOUR_BASE_URL" not in cfg.openai_api_base else None
                self._client = OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                warnings.warn("OpenAI provider configured but 'openai' package not found. Falling back to heuristic.")
                self._provider = "none"
        else:
            self._provider = "none" # Fallback if key is placeholder

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self._client:
            raise ValueError("OpenAI client not initialized.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self._client.chat.completions.create(
            model=self.cfg.llm_model_name,
            messages=messages,  # type: ignore
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def evaluate(self, text: str, source_user_id: str) -> QCResult:
        if self._provider == "openai":
            try:
                user_prompt = get_user_prompt(text)
                
                # Step 1: Quality Check
                qc_data = self._call_llm(QC_SYSTEM_PROMPT, user_prompt)
                
                include = qc_data.get("include", False)
                scores = qc_data.get("scores", {})
                if "reusability" in scores: # Normalize key
                    scores["reuse"] = scores.pop("reusability")

                if not include:
                    return QCResult(include=False, scores=scores, cot="", kg=[], query="")

                # Steps 2, 3, 4: Extract COT, KG, and Query
                cot_data = self._call_llm(COT_EXTRACTION_SYSTEM_PROMPT, user_prompt)
                kg_data = self._call_llm(KG_EXTRACTION_SYSTEM_PROMPT, user_prompt)
                query_data = self._call_llm(QUERY_EXTRACTION_SYSTEM_PROMPT, user_prompt)

                return QCResult(
                    include=True,
                    scores=scores,
                    cot=cot_data.get("cot", ""),
                    kg=kg_data.get("kg", []),
                    query=query_data.get("query", ""),
                )
            except Exception as e:
                warnings.warn(f"OpenAI call sequence failed: {e}. Falling back to heuristic.")

        # Fallback to heuristic for any failure or if provider is "none"
        return self._run_heuristic(text, source_user_id)

    def extract_query_from_user_questions(self, text: str) -> str:
        """Extracts a concise query from user-only questions/turns within the dialog text.

        Heuristics to isolate user turns:
        - Prefer lines prefixed by role markers like "User:", "用户:", "问:", "Q:", "Question:".
        - Otherwise, prefer lines ending with a question mark ("?" or "？").
        Falls back to the original text if no user-only content is detected.
        """
        user_only = self._extract_user_turns(text)

        # LLM path
        if self._provider == "openai" and self._client:
            try:
                user_prompt = get_user_prompt(user_only)
                query_data = self._call_llm(QUERY_EXTRACTION_SYSTEM_PROMPT, user_prompt)
                query = query_data.get("query", "").strip()
                if query:
                    return query
            except Exception:
                # Fall through to heuristic
                pass

        # Heuristic fallback
        return self._build_query(user_only)

    def _extract_user_turns(self, text: str) -> str:
        """Best-effort extraction of user-only utterances/questions from a dialog transcript."""
        lines = [l for l in text.splitlines()]
        kept: List[str] = []
        # Updated regex to handle formats like "User (timestamp):" or "User:"
        role_prefix_re = re.compile(r"^(user|用户|question|问|提问|q)\s*(?:\([^)]*\))?\s*[:：]\s*", flags=re.I)
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # Role-labeled user turns
            m = role_prefix_re.match(s)
            if m:
                # Extract only the message content after the role prefix
                user_message = role_prefix_re.sub("", s).strip()
                if user_message:  # Only add non-empty messages
                    kept.append(user_message)
                continue
            # Question-like lines
            if s.endswith("?") or s.endswith("？"):
                kept.append(s)
        # If nothing matched, return original text as a conservative fallback
        if not kept:
            return text
        return "\n".join(kept)

    def match_queries(self, base_query: str, candidate_queries: List[str]) -> List[bool]:
        """Uses LLM to find which candidate queries match the base query."""
        if self._provider != "openai" or not self._client:
            return [False] * len(candidate_queries)
        try:
            user_prompt = get_query_matching_prompt(base_query, candidate_queries)
            response = self._call_llm(QUERY_MATCHING_SYSTEM_PROMPT, user_prompt)
            matches = response.get("matches", [])
            if isinstance(matches, list) and all(isinstance(m, bool) for m in matches):
                return matches
            return [False] * len(candidate_queries)
        except Exception as e:
            warnings.warn(f"OpenAI call for query matching failed: {e}. Defaulting to no matches.")
            return [False] * len(candidate_queries)

    def are_focus_queries_useful(self, user_query: str, focus_queries: List[str]) -> List[bool]:
        """Decides via LLM (in batch) whether each memory's focus_query is useful for answering user_query.

        Fallback heuristic if LLM unavailable: keyword overlap check for each focus_query.
        """
        if not focus_queries:
            return []
        
        # Heuristic fallback
        if self._provider != "openai" or not self._client:
            uq = (user_query or "").lower()
            uq_terms = set(t for t in re.split(r"[^a-z0-9_\-\u4e00-\u9fff]+", uq) if len(t) >= 3)
            results = []
            for fq in focus_queries:
                fq_lower = (fq or "").lower()
                if not fq_lower.strip():
                    results.append(False)
                    continue
                fq_terms = set(t for t in re.split(r"[^a-z0-9_\-\u4e00-\u9fff]+", fq_lower) if len(t) >= 3)
                results.append(len(uq_terms & fq_terms) > 0)
            return results

        # LLM batch path
        try:
            focus_list_str = "\n".join([f"{i+1}. {fq}" for i, fq in enumerate(focus_queries)])
            user_prompt = (
                f"User Query:\n---\n{user_query}\n---\n\n"
                f"Focus Queries (numbered):\n---\n{focus_list_str}\n---\n"
            )
            response = self._call_llm(FOCUS_USEFULNESS_BATCH_SYSTEM_PROMPT, user_prompt)
            useful_list = response.get("useful", [])
            # Validate response length and types
            if isinstance(useful_list, list) and len(useful_list) == len(focus_queries):
                return [bool(u) for u in useful_list]
            # Fallback if response malformed
            warnings.warn(f"LLM returned malformed useful list (expected {len(focus_queries)}, got {len(useful_list)}). Using heuristic.")
        except Exception as e:
            warnings.warn(f"OpenAI call for focus usefulness failed: {e}. Falling back to heuristic.")
        
        # Fallback heuristic on error
        uq = (user_query or "").lower()
        uq_terms = set(t for t in re.split(r"[^a-z0-9_\-\u4e00-\u9fff]+", uq) if len(t) >= 3)
        results = []
        for fq in focus_queries:
            fq_lower = (fq or "").lower()
            if not fq_lower.strip():
                results.append(False)
                continue
            fq_terms = set(t for t in re.split(r"[^a-z0-9_\-\u4e00-\u9fff]+", fq_lower) if len(t) >= 3)
            results.append(len(uq_terms & fq_terms) > 0)
        return results

    def merge_cot(self, cot1: str, cot2: str) -> str:
        """Merges two CoT strings using an LLM."""
        if self._provider != "openai" or not self._client:
            return cot1 + "\n\n" + cot2  # Fallback
        try:
            user_prompt = get_cot_merging_prompt(cot1, cot2)
            response = self._call_llm(COT_MERGING_SYSTEM_PROMPT, user_prompt)
            return response.get("merged_cot", "")
        except Exception as e:
            warnings.warn(f"OpenAI call for CoT merging failed: {e}. Defaulting to concatenation.")
            return cot1 + "\n\n" + cot2

    def merge_kg(self, kg1: List[Dict[str, Any]], kg2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges two KGs using an LLM."""
        if self._provider != "openai" or not self._client:
            return kg1 + kg2  # Fallback
        try:
            # 直接传对象，由 get_kg_merging_prompt 负责序列化
            user_prompt = get_kg_merging_prompt(kg1, kg2)
            response = self._call_llm(KG_MERGING_SYSTEM_PROMPT, user_prompt)
            return response.get("merged_kg", [])
        except Exception as e:
            warnings.warn(f"OpenAI call for KG merging failed: {e}. Defaulting to concatenation.")
            return kg1 + kg2

    def merge_query(self, query1: str, query2: str) -> str:
        """Merges two focus_query strings using an LLM."""
        if self._provider != "openai" or not self._client:
            return f"{query1}；{query2}"  # Fallback: simple concatenation with semicolon
        try:
            user_prompt = get_query_merging_prompt(query1, query2)
            response = self._call_llm(QUERY_MERGING_SYSTEM_PROMPT, user_prompt)
            return response.get("merged_query", "")
        except Exception as e:
            warnings.warn(f"OpenAI call for query merging failed: {e}. Defaulting to concatenation.")
            return f"{query1}；{query2}"

    def _run_heuristic(self, text: str, source_user_id: str) -> QCResult:
        reuse, depth, focus = self._heuristic_scores(text)
        include = (reuse >= 2.0) and (depth >= 2.0) and (focus >= 1.5)
        cot = self._build_cot(text) if include else ""
        kg = [e.to_dict() for e in self._extract_kg(text, source_user_id)] if include else []
        query = self._build_query(text) if include else ""
        return QCResult(
            include=include,
            scores={"reuse": reuse, "depth": depth, "focus": focus},
            cot=cot,
            kg=kg,
            query=query,
        )

    def _heuristic_scores(self, text: str) -> Tuple[float, float, float]:
        # reuse: presence of steps, bullets, patterns
        bullets = len(re.findall(r"^(?:\s*[-*]|\s*\d+\.)", text, flags=re.M))
        templates = len(re.findall(r"\b(步骤|template|pattern|checklist|清单|模板)\b", text, flags=re.I))
        reuse = min(3.0, 1.0 + 0.5 * bullets + 0.7 * templates)

        # depth: technical terms, tradeoffs, constraints
        keywords = len(re.findall(r"\b(latency|throughput|复杂度|边界|约束|trade[- ]?off|限制|反例)\b", text, flags=re.I))
        if len(text) > 800:
            keywords += 2
        depth = min(3.0, 0.5 + 0.6 * keywords)

        # focus: topic concentration via topic-change heuristics
        headings = len(re.findall(r"^#+\s", text, flags=re.M))
        sections = len(re.findall(r"\n\n+", text)) + headings
        focus_penalty = max(0, sections - 4) * 0.4
        focus = max(0.0, min(3.0, 2.5 - focus_penalty))
        return reuse, depth, focus

    def _build_cot(self, text: str) -> str:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        steps = []
        for line in lines:
            if re.match(r"^(?:[-*]|\d+\.)\s+", line):
                steps.append(line)
        if not steps:
            # fallback: sentence-based outline
            sentences = re.split(r"[。.!?]\s+", text)
            steps = [f"- {s.strip()}" for s in sentences if 10 <= len(s.strip()) <= 180][:8]
        cot = "\n".join(["- 前提/环境: 明确依赖、版本、限制", *steps[:10], "- 结论: 给出预期结果与验证方法"])
        return cot

    def _extract_kg(self, text: str, source_user_id: str) -> List[KGEdge]:
        edges: List[KGEdge] = []
        # naive extraction of "X -> Y", "X uses Y"
        for m in re.finditer(r"([\w\-/]+)\s*(?:->|→|使用|uses)\s*([\w\-/]+)", text, flags=re.I):
            edges.append(KGEdge(head=m.group(1), relation="relates_to", tail=m.group(2), evidence=m.group(0), source_user_id=source_user_id))
        return edges[:20]

    def _build_query(self, text: str) -> str:
        # pick title-like first sentence
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        return first_line[:140]
