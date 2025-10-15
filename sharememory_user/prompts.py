"""
Centralized prompts for LLM interactions, following OpenAI API format.
Each prompt is designed for a separate API call to ensure independence.
"""

# 1. Quality Check Prompt
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

# 2. COT Extraction Prompt
COT_EXTRACTION_SYSTEM_PROMPT = """You are an expert in summarizing technical discussions.
Analyze the following text and extract its core reasoning process into a concise, structured outline (Chain-of-Thought style).
Focus on reusable steps. Be concise and focus on the key points.

**Output Schema**: Respond with a single JSON object containing the summary.
`{"cot": "<string: The structured COT outline>"}`
"""

# 3. KG Extraction Prompt
KG_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph expert.
Analyze the following text and extract meaningful entities and their relationships as knowledge graph triples.
Focus on technical entities, tools, and concepts.

**Output Schema**: Respond with a single JSON object containing a list of triples.
`{"kg": [{"head": "<entity>", "relation": "<relationship>", "tail": "<entity>"}]}`
"""

# 4. Query Extraction Prompt
QUERY_EXTRACTION_SYSTEM_PROMPT = """You are an expert in search intent.
Analyze the following text and formulate a single, focused question that this conversation snippet directly answers.
The question should represent a clear user query.

**Output Schema**: Respond with a single JSON object containing the query.
`{"query": "<string: The focused user query>"}`
"""


def get_user_prompt(text: str) -> str:
    """Formats the user prompt for all stages, providing the raw text."""
    return f"""
Conversation Text to Analyze:
---
{text}
---
"""

# 5. Query Matching Prompt
QUERY_MATCHING_SYSTEM_PROMPT = """You are an AI assistant specializing in semantic query matching.
Your task is to determine if a set of candidate queries are asking the exact same question as a base query.
The queries are considered a match if they are semantically identical, focusing on the same core intent and subject matter, even if the wording is different.

**Output Schema**: Respond with a single JSON object.
`{"matches": [<boolean: true if a candidate query matches the base query, false otherwise>]}`
The list of booleans must be in the same order as the candidate queries provided in the user prompt.
"""


def get_query_matching_prompt(base_query: str, candidate_queries: list[str]) -> str:
    candidates_str = "\n".join(f"- {q}" for q in candidate_queries)
    return f"""
Base Query:
---
{base_query}
---

Candidate Queries:
---
{candidates_str}
---
"""


# 6. CoT Merging Prompt
COT_MERGING_SYSTEM_PROMPT = """You are an AI assistant that merges two Chain-of-Thought (CoT) outlines into a single, comprehensive, and non-redundant CoT.
Combine the steps and reasoning from both CoTs, keeping the logical flow. Eliminate duplicate information and synthesize related points.

**Output Schema**: Respond with a single JSON object.
`{"merged_cot": "<string: The new, unified CoT outline>"}`
"""


def get_cot_merging_prompt(cot1: str, cot2: str) -> str:
    return f"""
CoT 1:
---
{cot1}
---

CoT 2:
---
{cot2}
---
"""


# 7. KG Merging Prompt
KG_MERGING_SYSTEM_PROMPT = """You are a Knowledge Graph expert.
Your task is to merge two sets of knowledge graph triples into a single, consistent graph.
Eliminate duplicate triples and resolve any conflicting information by choosing the most plausible relationship.

**Output Schema**: Respond with a single JSON object containing the merged list of triples.
`{"merged_kg": [{"head": "<entity>", "relation": "<relationship>", "tail": "<entity>"}]}`
"""


def get_kg_merging_prompt(kg1: str, kg2: str) -> str:
    """`kg1` and `kg2` should be JSON strings."""
    return f"""
KG 1:
---
{kg1}
---

KG 2:
---
{kg2}
---
"""
