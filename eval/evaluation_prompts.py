# -*- coding: utf-8 -*-
"""
Prompts used for the end-to-end evaluation script.
"""

def get_rag_answer_prompt(user_query: str, retrieved_context: str, user_profile: str) -> str:
    """
    Creates the prompt for the RAG (with context) condition.
    Includes the user's profile to personalize the response.
    """
    return f"""You are a helpful AI assistant. Your task is to answer the user's question based on the provided context.
The context is retrieved from a shared knowledge base of past conversations.
Synthesize the information from the context to provide a comprehensive and accurate answer.
If the context is not relevant, ignore it and answer based on your own knowledge.

**USER PROFILE:**
---
{user_profile}
---

**CONTEXT FROM SHARED MEMORY:**
---
{retrieved_context}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Your response should be relevant and appropriate for this specific user.

Your Answer:
"""

def get_fusion_rag_prompt(user_query: str, shared_memory_context: str, personal_memory_context: str, user_profile: str) -> str:
    """
    Creates the prompt for the Fusion RAG condition with both shared and personal memory.
    Includes the user's profile to personalize the response.
    """
    return f"""You are a helpful AI assistant. Your task is to answer the user's question based on the provided context from two memory sources.
The context is retrieved from both shared knowledge base and your personal memory of past conversations.
Synthesize the information from both sources to provide a comprehensive and accurate answer.
If the context is not relevant, ignore it and answer based on your own knowledge.

**USER PROFILE:**
---
{user_profile}
---

**CONTEXT FROM SHARED MEMORY:**
---
{shared_memory_context}
---

**CONTEXT FROM PERSONAL MEMORY:**
---
{personal_memory_context}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Use both shared knowledge and personal context to provide a response that is relevant and appropriate for this specific user.
If there are conflicts between shared and personal memory, prioritize the information that is most relevant to the user's current question.

Your Answer:
"""

def get_baseline_answer_prompt(user_query: str, user_profile: str) -> str:
    """
    Creates the prompt for the baseline (without context) condition.
    Includes the user's profile to personalize the response.
    """
    return f"""Your task is to answer the user's question based on your own knowledge.

**USER PROFILE:**
---
{user_profile}
---

**USER'S QUESTION:**
---
{user_query}
---

Important: Tailor your answer to the user's profile, expertise level, and professional background.
Your response should be relevant and appropriate for this specific user.

Your Answer:
"""

def get_judge_prompt(user_query: str, user_profile: str, answer_a: str, answer_b: str) -> str:
    """
    Creates a prompt for the judge model to provide scores and a justification, with no ties allowed.
    """
    return f"""You are an expert evaluator. Compare the two answers and determine which is better. You cannot declare a tie.

Evaluate based on:
- **Personalization**: Is the answer suited to the user's background and expertise?
- **Follow-up Question Addressing**: Does the answer anticipate potential follow-up questions?
- **Depth and Insight**: Does the answer provide valuable insights?

**IMPORTANT**: Scores for Answer A and Answer B must be different.

**USER PROFILE:**
---
{user_profile}
---

**USER'S QUESTION:**
---
{user_query}
---

**ANSWER A:**
---
{answer_a}
---

**ANSWER B:**
---
{answer_b}
---

**REQUIRED OUTPUT FORMAT (follow exactly):**

Justification: Answer xx provides xxxx.
Score A: 
Score B: 
Winner: xx"""
