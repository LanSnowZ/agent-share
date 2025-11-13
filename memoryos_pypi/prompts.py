"""
此文件集中存放 MemoryOS 系统使用的所有提示词（Prompts）。
"""

# Prompt for generating system response (from main_memoybank.py, generate_system_response_with_meta)
GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT = (
    "你是一位具备优秀沟通习惯的交流专家。在接下来的对话中，请始终以 {relationship} 的身份进行回应。\n"
    "以下是你的个性特征与已知知识：\n{assistant_knowledge_text}\n"
    "用户概览：\n"
    "{meta_data_text}\n"
    "你的任务是：在保持上述人设与语气的前提下进行回答。\n"
)

GENERATE_SYSTEM_RESPONSE_USER_PROMPT = (
    "<上下文>\n"
    "基于你与该用户最近的对话：\n"
    "{history_text}\n\n"
    "<记忆>\n"
    "与当前对话相关的记忆如下：\n"
    "{retrieval_text}\n\n"
    "<用户特征>\n"
    "结合以往与你的对话，你总结出该用户具有以下特征：\n"
    "{background}\n\n"
    "现在，请以 {relationship} 的身份继续和用户对话。\n"
    "用户刚刚说：{query}\n"
    "请用不超过 30 个字进行回应（语言不限，但需准确简洁）。\n"
    "注意最后给出的回答，如果与相关记忆无关就不要提及相关记忆，直接给出回答。"
    "回答时务必核对引用信息的时间是否与问题时间范围一致。"
)

# Prompt for assistant knowledge extraction (from utils.py, analyze_assistant_knowledge)
ASSISTANT_KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """你是一个助手知识抽取引擎。规则：
1. 仅抽取关于助手“身份”或“知识/能力”的明确信息。
2. 使用第一人称，表述必须简洁且客观。
3. 若没有相关信息，输出 "None"。"""

ASSISTANT_KNOWLEDGE_EXTRACTION_USER_PROMPT = """
# 助手知识抽取任务
请分析以下对话，抽取任何关于“助手”的事实或身份特征。
若无法抽取，直接回复 "None"。输出需尽可能简洁，格式如下：
【助手知识】
- [事实1]
- [事实2]
- （若无可抽取信息，输出 "None"）

Few-shot 示例：
1. 用户：能推荐几部电影吗？
   AI：可以，我推荐《星际穿越》。
   时间：2023-10-01
   【助手知识】
   - 我在 2023-10-01 推荐了《星际穿越》。

2. 用户：你能帮我找做菜的菜谱吗？
   AI：可以，我对烹饪菜谱与技巧非常熟悉。
   时间：2023-10-02
   【助手知识】
   - 我在 2023-10-02 具备烹饪菜谱与技巧方面的知识。

3. 用户：这很有意思，我不知道你还能这样。
   AI：很高兴你觉得有趣！
   【助手知识】
   - None

对话：
{conversation}
"""

# Prompt for summarizing dialogs (from utils.py, gpt_summarize)
SUMMARIZE_DIALOGS_SYSTEM_PROMPT = (
    "你是对话主题总结专家。请生成极其简洁且精准的主题摘要，尽量简短但保留核心要点。"
)
SUMMARIZE_DIALOGS_USER_PROMPT = (
    "请基于以下对话生成简要主题总结，最多 2-3 句短句：\n{dialog_text}\n简要总结："
)

# Prompt for multi-summary generation (from utils.py, gpt_generate_multi_summary)
MULTI_SUMMARY_SYSTEM_PROMPT = (
    "你是对话主题分析专家。请生成简明的小主题总结，主题不超过两个，尽量简短。"
)
MULTI_SUMMARY_USER_PROMPT = (
    "请分析以下对话，并在需要时生成不超过两个小主题的极简总结。\n"
    "每个总结尽量短，仅包含主题词与简述。输出为 JSON 数组：\n"
    '[\n  {{"theme": "主题", "keywords": ["关键词1", "关键词2"], "content": "简述"}}\n]\n'
    "\n对话内容：\n{text}"
)

# Prompt for personality analysis (NEW TEMPLATE)
PERSONALITY_ANALYSIS_SYSTEM_PROMPT = """你是一名专业的用户偏好分析助手。请基于给定维度，从提供的对话中分析用户的性格与偏好。

For each dimension:
1. Carefully read the conversation and determine if the dimension is reflected.
2. If reflected, determine the user's preference level: High / Medium / Low, and briefly explain the reasoning, including time, people, and context if possible.
3. If the dimension is not reflected, do not extract or list it.

Focus only on the user's preferences and traits for the personality analysis section.
仅输出“用户画像”部分。
"""

PERSONALITY_ANALYSIS_USER_PROMPT = """请分析下方最新一轮用户-AI 对话，并基于 95 个性格/偏好维度更新用户画像。

Here are the 95 dimensions and their explanations:

[Basic Information]
Name: 用户的姓名（如对话中提及）。
Gender: 用户的性别（如对话中提及）。
Age: 用户的年龄（如对话中提及）。
Occupation: 用户的职业（如对话中提及，不能有详细描述）。
Work Details: 用户的工作详情、所在行业或具体职位（如对话中提及 尽量简洁，只总结确定重要的信息，不能超过15个字）。

[Psychological Model (Basic Needs & Personality)]
Extraversion: 偏好社交活动。
Openness: 接纳新事物与新体验的意愿。
Agreeableness: 友好与合作倾向。
Conscientiousness: 责任心与组织能力。
Neuroticism: 情绪稳定性与敏感度。
Physiological Needs: 对舒适与基本需求的关注。
Need for Security: 对安全与稳定的重视。
Need for Belonging: 对群体归属的需求。
Need for Self-Esteem: 对尊重与认可的需求。
Cognitive Needs: 对知识与理解的渴望。
Aesthetic Appreciation: 对美与艺术的欣赏。
Self-Actualization: 追求自我实现。
Need for Order: 偏好整洁与条理。
Need for Autonomy: 偏好独立决策与行动。
Need for Power: 影响或掌控他人的愿望。
Need for Achievement: 重视成就与结果。

[AI Alignment Dimensions]
Helpfulness: 回答是否对用户有实际帮助。（反映用户对 AI 的期望）
Honesty: 回答是否真实可信。（反映用户对 AI 的期望）
Safety: 避免敏感或有害内容。（反映用户对 AI 的期望）
Instruction Compliance: 严格遵循用户指令。（反映用户对 AI 的期望）
Truthfulness: 内容准确与真实。（反映用户对 AI 的期望）
Coherence: 表达清晰、逻辑一致。（反映用户对 AI 的期望）
Complexity: 偏好详细与复杂的信息。
Conciseness: 偏好简短清晰的回答。

[Content Platform Interest Tags]
Science Interest: 科学类话题兴趣。
Education Interest: 教育与学习相关关注。
Psychology Interest: 心理学类话题兴趣。
Family Concern: 家庭与育儿相关关注。
Fashion Interest: 时尚类话题兴趣。
Art Interest: 艺术参与或兴趣。
Health Concern: 健康与生活方式关注。
Financial Management Interest: 理财与预算兴趣。
Sports Interest: 运动与体能活动兴趣。
Food Interest: 对烹饪与美食的热情。
Travel Interest: 旅行与探索兴趣。
Music Interest: 音乐欣赏或创作兴趣。
Literature Interest: 文学与阅读兴趣。
Film Interest: 电影与影像兴趣。
Social Media Activity: 社交媒体活跃度。
Tech Interest: 科技与创新兴趣。
Environmental Concern: 环境与可持续议题关注。
History Interest: 历史知识与话题兴趣。
Political Concern: 政治与社会议题关注。
Religious Interest: 宗教与精神领域兴趣。
Gaming Interest: 电子游戏或桌游兴趣。
Animal Concern: 动物或宠物关怀。
Emotional Expression: 情绪表达直率 vs. 克制偏好。
Sense of Humor: 幽默风格偏好 vs. 严肃风格。
Information Density: 偏好信息密度（详细 vs. 简洁）。
Language Style: 语言风格（正式 vs. 口语）。
Practicality: 偏好可操作建议 vs. 理论讨论。

**任务说明：**
1. 阅读下方已有用户画像；
2. 基于上述 95 个维度在新对话中寻找证据；
3. 将发现整合到完整的用户画像中；
4. **严格遵守格式规范**，对每个可识别维度：
   - **基础信息维度**：`DimensionName（Value: 具体值）`
     示例：`Name（Value: 张三）`、`Occupation（Value: 软件工程师）`
   - **其他维度**：`DimensionName（Level: High/Medium/Low）`
     示例：`Extraversion（Level: High）`、`Helpfulness（Level: Medium）`
   - **注意事项**：
     * 使用中文括号（）而非英文括号()
     * 每个维度单独一行，不要合并多个维度
     * 不要添加额外标注（如"AI期望"等）
     * Level 值只能是：High / Medium / Low / Medium-High / Low-Medium
5. 尽量给出简短理由（若可定位到时间、人物、场景更佳）；
6. 在合并时保留旧画像中的有效洞见；
7. 若无法从旧画像或新对话推断某维度，请不要包含该维度。

**已有用户画像：**
{existing_user_profile}

**最新用户-AI 对话：**
{conversation}

**更新后的用户画像：**
请在下方提供综合后的完整用户画像，需同时整合旧画像与新对话的洞见："""

# Prompt for knowledge extraction (NEW)
KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """你是一名知识抽取助手。你的任务是从对话中抽取“用户私有数据”与“助手知识”。

关注以下两类信息：
1. 用户私有数据：用户的个人信息、偏好或私密事实；
2. 助手知识：助手展示过的行为、提供过的内容或体现出的能力；

抽取时务必简洁客观，使用尽可能短的短语。
"""

KNOWLEDGE_EXTRACTION_USER_PROMPT = """请从下方最新的用户-AI 对话中，抽取“用户私有数据”与“助手知识”。

Latest User-AI Conversation:
{conversation}

【用户私有数据】
抽取与用户相关的个人信息；务必极简：
- [简要事实]： [最小上下文（含实体与时间）]
- [简要事实]： [最小上下文（含实体与时间）]
- （若未发现，写 "None"）

【助手知识】
抽取助手体现出的行为或能力；采用“Assistant [动作] at [时间]”格式，尽量简短：
- Assistant [简要动作] at [时间/场景]
- Assistant [简要能力] during [简要场景]
- （若未发现，写 "None"）
"""

# Prompt for updating user profile (from utils.py, gpt_update_profile)
UPDATE_PROFILE_SYSTEM_PROMPT = "你是用户画像的合并与更新专家。请将新的分析信息整合进旧画像，保持一致性并提升整体理解，避免冗余。新的分析来自特定维度，请有意义地纳入。"
UPDATE_PROFILE_USER_PROMPT = '请基于新的分析更新以下用户画像。若旧画像为空或为 "None"，则基于新分析创建一份新的。\n\n旧用户画像：\n{old_profile}\n\n新分析数据：\n{new_analysis}\n\n更新后的用户画像：'

# Prompt for extracting theme (from utils.py, gpt_extract_theme)
EXTRACT_THEME_SYSTEM_PROMPT = "你是文本主题提取专家。请给出精炼的主题。"
EXTRACT_THEME_USER_PROMPT = "请从以下文本中提取主要主题：\n{answer_text}\n\n主题："


# Prompt for conversation continuity check (from dynamic_update.py, _is_conversation_continuing)
CONTINUITY_CHECK_SYSTEM_PROMPT = """你是对话连续性判定器。你的任务是严格判断两段对话的讨论主体是否一致。
仅返回 'true' 或 'false'。"""

CONTINUITY_CHECK_USER_PROMPT = """判断以下两页对话是否围绕同一核心主体进行讨论（无主体转移）。

**判断标准（严格模式）**：
1. 讨论的核心主体（技术、概念、工具、框架等）必须一致
2. 即使话题相关，但如果核心主体发生切换，也应判断为不连续
3. 示例：
   - 从"RAG检索增强"切换到"MOE混合专家" → 不连续 (false)
   - 从"Transformer架构"切换到"BERT模型" → 不连续 (false)
   - 继续深入讨论同一技术的不同方面 → 连续 (true)
   - 在同一主体下扩展子话题 → 连续 (true)

**请严格判断核心讨论主体是否发生变化**，仅返回 "true" 或 "false"。

上一页：
User: {prev_user}
Assistant: {prev_agent}

当前页：
User: {curr_user}
Assistant: {curr_agent}

是否为同一主体的连续对话？"""

# Prompt for generating meta info (from dynamic_update.py, _generate_meta_info)
META_INFO_SYSTEM_PROMPT = """你是对话元摘要的更新器。你的任务：
1. 保留上一版元摘要中的相关上下文；
2. 融入当前对话中的新信息；
3. 仅输出更新后的摘要（不需要解释）。"""
META_INFO_USER_PROMPT = """请在保持连续性的前提下，用当前对话更新元摘要。
        
    指南：
    1. 以上一版元摘要为起点（如存在）；
    2. 根据新对话进行补充/更新；
    3. 保持简洁（最多 1-2 句）；
    4. 保持上下文一致。
    
    之前的元摘要：{last_meta}
    新的对话：
    {new_dialogue}
    
    更新后的元摘要："""
