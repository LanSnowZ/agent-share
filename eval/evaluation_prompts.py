# -*- coding: utf-8 -*-
"""
端到端评测脚本使用的提示词。
"""

def get_rag_answer_prompt(user_query: str, retrieved_context: str, user_profile: str) -> str:
    """
    生成用于 RAG（带上下文）条件的提示词，包含用户画像以实现个性化回答。
    """
    return f"""你是一名乐于助人的 AI 助手。请基于提供的上下文回答用户的问题。
上下文来源于以往对话构成的共享知识库。
请综合上下文信息，给出全面且准确的回答。
若上下文不相关，请忽略上下文并基于自身知识作答。

**用户画像：**
---
{user_profile}
---

**共享记忆上下文：**
---
{retrieved_context}
---

**用户问题：**
---
{user_query}
---

重要：请根据用户画像、专业背景与水平调整表达。
回答需与该用户高度相关、恰当。

你的回答：
"""

def get_fusion_rag_prompt(user_query: str, shared_memory_context: str, personal_memory_context: str, user_profile: str) -> str:
    """
    生成用于 Fusion RAG（共享与个人记忆融合）条件的提示词，包含用户画像以实现个性化回答。
    """
    return f"""你是一名乐于助人的 AI 助手。请基于两类记忆源提供的上下文回答用户的问题。
上下文来自共享知识库与你对过往对话的个人记忆。
请综合两类来源信息，给出全面且准确的回答。
若上下文不相关，请忽略上下文并基于自身知识作答。

**用户画像：**
---
{user_profile}
---

**共享记忆上下文：**
---
{shared_memory_context}
---

**个人记忆上下文：**
---
{personal_memory_context}
---

**用户问题：**
---
{user_query}
---

重要：请结合用户画像、专业水平与背景定制回答。
需同时合理利用共享知识与个人记忆；若二者冲突，请优先采用与当前问题最相关的信息。

你的回答：
"""

def get_baseline_answer_prompt(user_query: str, user_profile: str) -> str:
    """
    生成用于基线条件（无上下文）的提示词，包含用户画像以实现个性化回答。
    """
    return f"""你的任务是基于自身知识回答用户问题。

**用户画像：**
---
{user_profile}
---

**用户问题：**
---
{user_query}
---

重要：请根据用户画像、专业水平与背景进行个性化作答。
回答需与该用户高度相关、恰当。

你的回答：
"""

def get_judge_prompt(user_query: str, user_profile: str, answer_a: str, answer_b: str) -> str:
    """
    生成用于评审模型的提示词：比较两份答案并给出分数与理由，且不允许平局。
    """
    return f"""你是一名评测专家。比较两份回答并判断哪一份更好。不得判定平局。

评估依据：
- **个性化**：回答是否贴合用户背景与专业水平？
- **追问预见性**：回答是否预见并覆盖潜在追问？
- **深度与洞见**：回答是否提供有价值的洞见？

**重要**：A 与 B 的分数必须不同。

**用户画像：**
---
{user_profile}
---

**用户问题：**
---
{user_query}
---

**答案 A：**
---
{answer_a}
---

**答案 B：**
---
{answer_b}
---

**必须严格遵循的输出格式：**

Justification: 说明为何某答案更好。
Score A: 
Score B: 
Winner: A 或 B"""
