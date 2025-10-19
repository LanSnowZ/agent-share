"""
集中管理与 LLM 交互使用的提示词，遵循 OpenAI API 的消息格式。
每个提示词面向独立的 API 调用，互不依赖。
"""

# 1. 质量评估提示
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

# 2. COT 抽取提示
COT_EXTRACTION_SYSTEM_PROMPT = """你是技术讨论总结专家。
请从以下文本中提炼核心推理过程，产出简洁的结构化大纲（Chain-of-Thought 风格）。
强调可复用步骤，保持精炼，聚焦关键点。

**输出格式**：仅输出一个 JSON 对象。
`{"cot": "<string: 结构化的 COT 大纲>"}`
"""

# 3. KG 抽取提示
KG_EXTRACTION_SYSTEM_PROMPT = """你是知识图谱专家。
请从以下文本中抽取有意义的实体与关系，并以三元组表示。
聚焦技术实体、工具与概念。

**输出格式**：仅输出一个 JSON 对象，包含三元组列表。
`{"kg": [{"head": "<实体>", "relation": "<关系>", "tail": "<实体>"}]}`
"""

# 4. 查询抽取提示
QUERY_EXTRACTION_SYSTEM_PROMPT = """你是搜索意图理解专家。
请分析以下文本，归纳出一个该片段能够直接回答的、单一且聚焦的问题。
该问题需能代表清晰的用户查询。

**输出格式**：仅输出一个 JSON 对象。
`{"query": "<string: 聚焦的用户查询>"}`
"""


def get_user_prompt(text: str) -> str:
    """为各阶段统一格式化用户内容。"""
    return f"""
待分析的对话文本：
---
{text}
---
"""

# 5. 查询语义匹配提示
QUERY_MATCHING_SYSTEM_PROMPT = """你是一名语义查询匹配助手。
需要判断一组候选查询是否与基准查询在语义上完全一致（核心意图与主题一致，措辞可不同）。

**输出格式**：仅输出一个 JSON 对象。
`{"matches": [<boolean: 若候选与基准一致则为 true，否则为 false>]}`
布尔数组顺序必须与输入的候选查询顺序一致。
"""


def get_query_matching_prompt(base_query: str, candidate_queries: list[str]) -> str:
    candidates_str = "\n".join(f"- {q}" for q in candidate_queries)
    return f"""
基准查询：
---
{base_query}
---

候选查询：
---
{candidates_str}
---
"""


# 6. CoT 合并提示
COT_MERGING_SYSTEM_PROMPT = """你是一名将两份 Chain-of-Thought（COT）大纲合并为单一、完整且无冗余版本的助手。
请合并两份 COT 的步骤与推理，保持逻辑顺序，去重并综合相关点。

**输出格式**：仅输出一个 JSON 对象。
`{"merged_cot": "<string: 新的统一 COT 大纲>"}`
"""


def get_cot_merging_prompt(cot1: str, cot2: str) -> str:
    return f"""
CoT 1：
---
{cot1}
---

CoT 2：
---
{cot2}
---
"""


# 7. KG 合并提示
KG_MERGING_SYSTEM_PROMPT = """你是知识图谱专家。
请将两组三元组合并为单一、一致的知识图谱，消除重复并在冲突时选择更合理的关系。

**输出格式**：仅输出一个 JSON 对象，包含合并后的三元组列表。
`{"merged_kg": [{"head": "<实体>", "relation": "<关系>", "tail": "<实体>"}]}`
"""


def get_kg_merging_prompt(kg1: str, kg2: str) -> str:
    """`kg1` 与 `kg2` 传入 JSON 字符串。"""
    return f"""
KG 1：
---
{kg1}
---

KG 2：
---
{kg2}
---
"""
