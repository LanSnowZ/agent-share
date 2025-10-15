from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Tuple

try:
    import orjson as json
except ImportError:
    import json  # type: ignore

from config import Config
from models import QCResult, KGEdge

# 直接定义 prompts，避免模块导入冲突
QC_SYSTEM_PROMPT = """You are a meticulous assistant evaluating a conversation transcript to determine if it contains reusable, in-depth, and focused experience.

**Evaluation Criteria (score 0.0 to 3.0):**
1.  **reusability**: Does it summarize a transferable method, template, or pitfall list?
2.  **depth**: Does it contain non-superficial inference, trade-offs, or boundary conditions?
3.  **focus**: Is it centered around a single core task?

**Inclusion Logic**: A conversation is included if `reusability >= 2.0`, `depth >= 2.0`, and `focus >= 1.5`.

**Output Schema**: Respond with a single JSON object. Do not add any text before or after it.
- For inclusion: `{"include": true, "scores": {"reusability": <float>, "depth": <float>, "focus": <float>}}`
- For exclusion: `{"include": false, "scores": {"reusability": <float>, "depth": <float>, "focus": <float>}}`
"""

COT_EXTRACTION_SYSTEM_PROMPT = """You are an expert in summarizing technical discussions.
Analyze the following text and extract its core reasoning process into a concise, structured outline (Chain-of-Thought style).
Focus on reusable steps. Be concise and focus on the key points.

**Output Schema**: Respond with a single JSON object containing the summary.
`{"cot": "<string: The structured COT outline>"}`
"""

KG_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph expert.
Analyze the following text and extract meaningful entities and their relationships as knowledge graph triples.
Focus on technical entities, tools, and concepts.

**Output Schema**: Respond with a single JSON object containing a list of triples.
`{"kg": [{"head": "<entity>", "relation": "<relationship>", "tail": "<entity>"}]}`
"""

QUERY_EXTRACTION_SYSTEM_PROMPT = """You are a query formulation expert.
Analyze the following text and extract the core question or task it addresses.
The query should be concise and capture the main intent.

**Output Schema**: Respond with a single JSON object.
`{"query": "<string: The extracted query>"}`
"""

QUERY_MATCHING_SYSTEM_PROMPT = """You are a query matching expert.
Determine which candidate queries semantically match the base query.
Return a boolean list indicating matches.

**Output Schema**: Respond with a single JSON object.
`{"matches": [<bool>, <bool>, ...]}`
"""

COT_MERGING_SYSTEM_PROMPT = """You are an expert in merging structured reasoning.
Merge the following two Chain-of-Thought summaries into one coherent, comprehensive summary.

**Output Schema**: Respond with a single JSON object.
`{"merged_cot": "<string: The merged summary>"}`
"""

KG_MERGING_SYSTEM_PROMPT = """You are a knowledge graph merging expert.
Merge the following two knowledge graph lists, removing duplicates and consolidating information.

**Output Schema**: Respond with a single JSON object.
`{"merged_kg": [{"head": "<entity>", "relation": "<relationship>", "tail": "<entity>"}]}`
"""

def get_user_prompt(text: str) -> str:
    return f"**Text to Evaluate:**\n{text}"

def get_query_matching_prompt(base_query: str, candidate_queries: List[str]) -> str:
    candidates = "\n".join([f"{i+1}. {q}" for i, q in enumerate(candidate_queries)])
    return f"**Base Query:**\n{base_query}\n\n**Candidate Queries:**\n{candidates}\n\nWhich candidates match the base query?"

def get_cot_merging_prompt(cot1: str, cot2: str) -> str:
    return f"**COT 1:**\n{cot1}\n\n**COT 2:**\n{cot2}\n\nMerge these summaries:"

def get_kg_merging_prompt(kg1: List[Dict], kg2: List[Dict]) -> str:
    return f"**KG 1:**\n{json.dumps(kg1)}\n\n**KG 2:**\n{json.dumps(kg2)}\n\nMerge these knowledge graphs:"


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
            kg1_str = json.dumps(kg1)
            kg2_str = json.dumps(kg2)
            user_prompt = get_kg_merging_prompt(kg1_str, kg2_str)
            response = self._call_llm(KG_MERGING_SYSTEM_PROMPT, user_prompt)
            return response.get("merged_kg", [])
        except Exception as e:
            warnings.warn(f"OpenAI call for KG merging failed: {e}. Defaulting to concatenation.")
            return kg1 + kg2

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
