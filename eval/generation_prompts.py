# -*- coding: utf-8 -*-
"""
Prompts used for the synthetic dataset generator.
"""

def get_dialogue_generation_prompt(topic: str, query: str, persona_name: str, persona_profile: str, follow_up_style: str) -> str:
    """
    Creates the system prompt for generating a multi-turn dialogue.
    """
    return f"""You are a powerful simulation engine for generating high-quality academic dialogues. Your task is to create a realistic, in-depth, 3-turn conversation about "{topic}".

**ROLES:**
*   **User**: You are a **{persona_name}**. Your background is: "{persona_profile}".
*   **AI**: You are a world-class AI research scientist, capable of explaining complex topics with clarity and depth.

**TASK:**
Generate a 3-turn dialogue starting with the user's initial query. The user's follow-up questions must reflect their specific persona and background.

*   **Turn 1**:
    *   **User**: As a {persona_name}, asks the initial question: "{query}"
    *   **AI**: Provides a comprehensive and accurate answer to the core question.

*   **Turn 2**:
    *   **User**: Based on the AI's first answer, the user asks a follow-up question that is highly characteristic of their persona. The question should follow this style: "{follow_up_style}".
    *   **AI**: Answers the nuanced follow-up, providing deeper technical details, theoretical background, or practical advice as needed.

*   **Turn 3**:
    *   **User**: Asks a final question to broaden the perspective, perhaps asking about limitations, future trends, or comparisons to alternative technologies.
    *   **AI**: Provides a concluding high-level summary, addressing the user's final question and wrapping up the discussion.

**OUTPUT INSTRUCTIONS:**
*   You MUST generate the full 3-turn dialogue.
*   The dialogue must be technically accurate and detailed.
*   Start each line with "User:" or "AI:".
*   Do NOT add any introductory or concluding text outside of the dialogue itself.
"""
