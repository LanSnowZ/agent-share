
# PERSONAS = {
#     "algorithm_engineer": {
#         "name": "Algorithm Engineer",
#         "profile_text": "Name: Algorithm Engineer. Gender: flexible. Occupation: Implements machine learning algorithms in production code and prototypes; responsible for debugging, profiling, and optimizing. Personality: pragmatic, detail-oriented, hands-on, enjoys measurable improvements. Language style: concise, technical, often includes pseudocode or code pointers. Likes: clean architectures, reproducible experiments, profiling, benchmarks, vectorization. Dislikes: vague advice, untested claims, inefficient loops, missing docs.",
#         "follow_up_style": "Ask for concrete implementations, code-level steps, and measurable performance outcomes."
#     },
#     "theorist": {
#         "name": "Theoretical Researcher",
#         "profile_text": "Name: Theoretical Researcher. Gender: flexible. Occupation: Studies mathematical foundations, assumptions, and limits of learning algorithms. Personality: analytical, rigorous, enjoys proofs and derivations, skeptical of heuristics. Language style: formal and precise with definitions, lemmas, and caveats. Likes: elegant objectives, convergence guarantees, sample complexity bounds. Dislikes: hand-wavy fixes, hidden assumptions, irreproducible results.",
#         "follow_up_style": "Ask for assumptions, guarantees, explicit trade-offs, and formal problem statements."
#     },
#     "student": {
#         "name": "Student",
#         "profile_text": "Name: Student. Gender: flexible. Occupation: Learner of ML/LLMs with partial background knowledge. Personality: curious, enthusiastic, sometimes overwhelmed, eager to build small demos. Language style: informal-to-technical with lots of clarifying questions. Likes: step-by-step guides, small examples, sanity checks. Dislikes: unexplained jargon, too much math at once, massive compute needs.",
#         "follow_up_style": "Ask for beginner-friendly explanations, analogies, and minimal working examples."
#     },
#     "teacher": {
#         "name": "Teacher",
#         "profile_text": "Name: Teacher. Gender: flexible. Occupation: Teaches ML/LLMs and designs pedagogy, labs, and assessments. Personality: patient, structured, outcome-focused, values clarity and scaffolding. Language style: clear, analogy-rich, avoids unnecessary jargon. Likes: concept maps, worked examples, formative assessment. Dislikes: confusing leaps, unmotivated notation, skipping prerequisites.",
#         "follow_up_style": "Ask for lesson-plan framing, analogies, checks for understanding, and assessment strategies."
#     },
#     "business_stakeholder": {
#         "name": "Business Stakeholder",
#         "profile_text": "Name: Business Stakeholder. Gender: flexible. Occupation: Owns product or business outcomes; wants AI benefits without deep technical details. Personality: pragmatic, ROI-driven, risk-aware, prefers plain language and timelines. Language style: concise, focused on outcomes, costs, and risk. Likes: impact metrics, reliability, compliance, time-to-value. Dislikes: technical jargon, unpredictable costs, fragile pipelines.",
#         "follow_up_style": "Ask for ROI, risk/benefit trade-offs, timelines, resourcing, and success criteria."
#     }
# }

# TOPICS = {
#     "deep_research": {
#         "name": "Deep Research Techniques for LLMs",
#         "seed_queries": [
#             {"id": "deep-001", "query": "Give a high-level definition of deep research with LLMs and contrast it with shallow Q&A.", "personas": ["student", "teacher"]},
#             {"id": "deep-002", "query": "Provide pseudocode for an iterative research loop with retrieval, critique, and synthesis stages.", "personas": ["algorithm_engineer"]},
#             {"id": "deep-003", "query": "State assumptions and a formal model for iterative self-refinement in research agents.", "personas": ["theorist"]},
#             {"id": "deep-004", "query": "Explain how deep research can help a non-technical team produce accurate market briefs.", "personas": ["business_stakeholder"]},
#             {"id": "deep-005", "query": "Design a lecture outline introducing deep research to first-year students.", "personas": ["teacher"]},
#             {"id": "deep-006", "query": "Compare Tree-of-Thought, Self-Ask, and Plan-and-Solve for long-horizon investigations.", "personas": ["theorist", "teacher"]},
#             {"id": "deep-007", "query": "Implement a scratchpad memory and reasoning trace store for multi-session research.", "personas": ["algorithm_engineer"]},
#             {"id": "deep-008", "query": "What common pitfalls occur when beginners try deep research for the first time?", "personas": ["student", "teacher"]},
#             {"id": "deep-009", "query": "Propose stopping criteria and success metrics for a deep research workflow.", "personas": ["theorist", "business_stakeholder"]},
#             {"id": "deep-010", "query": "Explain how research agents should cite and cross-verify sources to reduce hallucinations.", "personas": ["teacher", "algorithm_engineer"]},
#             {"id": "deep-011", "query": "Show code to integrate web browsing tools with retrieval and note-taking for research.", "personas": ["algorithm_engineer"]},
#             {"id": "deep-012", "query": "Provide a formal argument for why self-consistency sampling can stabilize conclusions.", "personas": ["theorist"]},
#             {"id": "deep-013", "query": "How should students keep track of claims, evidence, and counterevidence?", "personas": ["student", "teacher"]},
#             {"id": "deep-014", "query": "Define a rubric to grade depth, novelty, and groundedness of research outputs.", "personas": ["teacher", "theorist"]},
#             {"id": "deep-015", "query": "Explain compliance and audit needs for research in regulated industries.", "personas": ["business_stakeholder"]},
#             {"id": "deep-016", "query": "Show an inference-time cost model for multi-step research and how to optimize it.", "personas": ["algorithm_engineer", "business_stakeholder"]},
#             {"id": "deep-017", "query": "Design checkpoints that prevent exploration loops and dead-ends.", "personas": ["theorist", "algorithm_engineer"]},
#             {"id": "deep-018", "query": "Create a classroom activity to practice evidence triangulation.", "personas": ["teacher", "student"]},
#             {"id": "deep-019", "query": "Formally discuss failure modes when source documents disagree.", "personas": ["theorist"]},
#             {"id": "deep-020", "query": "Show code for caching intermediate artifacts between research sessions.", "personas": ["algorithm_engineer"]},
#             {"id": "deep-021", "query": "How do we teach novices to distinguish correlation from causation in research claims?", "personas": ["teacher", "student"]},
#             {"id": "deep-022", "query": "Propose a lightweight governance model for approving research reports.", "personas": ["business_stakeholder"]},
#             {"id": "deep-023", "query": "Analyze token budgeting strategies for long-horizon research prompts.", "personas": ["algorithm_engineer", "theorist"]},
#             {"id": "deep-024", "query": "Give a gentle analogy that explains iterative refinement to newcomers.", "personas": ["teacher", "student"]},
#             {"id": "deep-025", "query": "Define KPIs that demonstrate ROI of deep research in a product team.", "personas": ["business_stakeholder"]}
#         ],
#         "eval_queries": [
#             {"id": "deep-eval-001", "query": "Design a reproducible, auditable deep research pipeline with explicit provenance and latency budgets.", "persona": "algorithm_engineer"},
#             {"id": "deep-eval-002", "query": "Provide a formal definition of research ‘depth’ and propose measurable proxies for evaluation.", "persona": "theorist"},
#             {"id": "deep-eval-003", "query": "Create a 45-minute lesson plan that teaches evidence triangulation and bias mitigation to beginners.", "persona": "teacher"},
#             {"id": "deep-eval-004", "query": "Explain how a deep research assistant would lower costs and risks for a non-technical team.", "persona": "business_stakeholder"}
#         ]
#     },
#     "rag": {
#         "name": "Retrieval-Augmented Generation (RAG)",
#         "seed_queries": [
#             {"id": "rag-001", "query": "Give a beginner-friendly definition of RAG and its benefits.", "personas": ["student", "teacher"]},
#             {"id": "rag-002", "query": "Write code to build a FAISS/HNSW index and perform top-k retrieval.", "personas": ["algorithm_engineer"]},
#             {"id": "rag-003", "query": "State conditions under which retrieval reduces hallucinations in a formal sense.", "personas": ["theorist"]},
#             {"id": "rag-004", "query": "Explain how RAG keeps product knowledge fresh without constant fine-tuning.", "personas": ["business_stakeholder"]},
#             {"id": "rag-005", "query": "Create a classroom analogy showing how a librarian (retriever) helps a writer (generator).", "personas": ["teacher"]},
#             {"id": "rag-006", "query": "Compare dense, sparse, and hybrid retrieval under domain shift.", "personas": ["theorist", "algorithm_engineer"]},
#             {"id": "rag-007", "query": "Show chunking strategies and their impact on recall and latency.", "personas": ["algorithm_engineer", "student"]},
#             {"id": "rag-008", "query": "Explain typical failure modes when retrieved passages are off-topic.", "personas": ["teacher", "student"]},
#             {"id": "rag-009", "query": "Define evaluation metrics that separately assess retriever and generator quality.", "personas": ["theorist", "teacher"]},
#             {"id": "rag-010", "query": "Explain business KPIs for a RAG-based customer support assistant.", "personas": ["business_stakeholder"]},
#             {"id": "rag-011", "query": "Provide pseudocode for query rewriting and multi-hop retrieval.", "personas": ["algorithm_engineer"]},
#             {"id": "rag-012", "query": "Give a theoretical account of why fusion-in-decoder can improve grounding.", "personas": ["theorist"]},
#             {"id": "rag-013", "query": "Design a classroom demo comparing grounded vs non-grounded answers.", "personas": ["teacher", "student"]},
#             {"id": "rag-014", "query": "Show strategies to refresh embeddings and indices in evolving corpora.", "personas": ["algorithm_engineer"]},
#             {"id": "rag-015", "query": "Compare cost and risk of RAG versus frequent full-model retraining.", "personas": ["business_stakeholder"]},
#             {"id": "rag-016", "query": "Teach the difference between parametric and retrieved knowledge to beginners.", "personas": ["teacher"]},
#             {"id": "rag-017", "query": "Integrate retrieved snippets into prompts with salience ordering.", "personas": ["algorithm_engineer", "student"]},
#             {"id": "rag-018", "query": "Formally analyze contradictions among retrieved sources and tie-breaking strategies.", "personas": ["theorist"]},
#             {"id": "rag-019", "query": "Advise students on critically evaluating evidence quality and recency.", "personas": ["teacher", "student"]},
#             {"id": "rag-020", "query": "List business risks of stale embeddings and how to mitigate them.", "personas": ["business_stakeholder"]},
#             {"id": "rag-021", "query": "Profile latency budgets across retriever, reranker, and generator.", "personas": ["algorithm_engineer"]},
#             {"id": "rag-022", "query": "Craft a plain-language explanation of RAG for executives.", "personas": ["teacher", "business_stakeholder"]},
#             {"id": "rag-023", "query": "Why can sparse retrievers fail on semantic queries? Provide intuition.", "personas": ["theorist", "student"]},
#             {"id": "rag-024", "query": "Implement a cache for retrieval results with privacy controls.", "personas": ["algorithm_engineer"]},
#             {"id": "rag-025", "query": "Explain where RAG adds the most value in market intelligence workflows.", "personas": ["business_stakeholder"]}
#         ],
#         "eval_queries": [
#             {"id": "rag-eval-001", "query": "Propose an embedding refresh and index-rebuild strategy for a fast-changing corpus with strict latency SLAs.", "persona": "algorithm_engineer"},
#             {"id": "rag-eval-002", "query": "Explain to beginners how grounding reduces hallucinations while noting its limitations.", "persona": "teacher"},
#             {"id": "rag-eval-003", "query": "Provide a formal model of retrieval-augmented knowledge fusion and its failure modes.", "persona": "theorist"},
#             {"id": "rag-eval-004", "query": "Show how a RAG assistant lowers support costs and risk compared to periodic fine-tuning.", "persona": "business_stakeholder"}
#         ]
#     },
#     "moe": {
#         "name": "Mixture of Experts (MoE)",
#         "seed_queries": [
#             {"id": "moe-001", "query": "Define Mixture of Experts in plain terms for newcomers.", "personas": ["student", "teacher"]},
#             {"id": "moe-002", "query": "Provide pseudocode for top-k gating and expert routing.", "personas": ["algorithm_engineer"]},
#             {"id": "moe-003", "query": "Give theoretical motivation for sparse activation and conditional computation.", "personas": ["theorist"]},
#             {"id": "moe-004", "query": "Explain the business case for MoE and when it reduces inference costs.", "personas": ["business_stakeholder"]},
#             {"id": "moe-005", "query": "Create a classroom analogy that explains token-to-expert routing.", "personas": ["teacher"]},
#             {"id": "moe-006", "query": "Describe training instabilities such as expert collapse and balancing losses.", "personas": ["theorist", "algorithm_engineer"]},
#             {"id": "moe-007", "query": "Show kernel-level considerations for dispatch/combine operations.", "personas": ["algorithm_engineer"]},
#             {"id": "moe-008", "query": "Explain mathematically why certain experts dominate routing.", "personas": ["theorist"]},
#             {"id": "moe-009", "query": "List misconceptions beginners have about MoE and correct them.", "personas": ["teacher", "student"]},
#             {"id": "moe-010", "query": "Quantify FLOPs and memory trade-offs of sparse vs dense models.", "personas": ["algorithm_engineer", "business_stakeholder"]},
#             {"id": "moe-011", "query": "Formalize load-balancing objectives and their gradients.", "personas": ["theorist"]},
#             {"id": "moe-012", "query": "Design telemetry to monitor expert utilization and skew in production.", "personas": ["algorithm_engineer"]},
#             {"id": "moe-013", "query": "Discuss business risks when experts collapse or drift.", "personas": ["business_stakeholder"]},
#             {"id": "moe-014", "query": "Draft a teaching diagram that explains gating and capacity factors.", "personas": ["teacher"]},
#             {"id": "moe-015", "query": "Can a sparse MoE be distilled into a dense model without severe loss?", "personas": ["theorist", "algorithm_engineer"]},
#             {"id": "moe-016", "query": "Provide code to add an auxiliary loss that balances routing.", "personas": ["algorithm_engineer"]},
#             {"id": "moe-017", "query": "Design a classroom activity to visualize token-to-expert assignment.", "personas": ["teacher"]},
#             {"id": "moe-018", "query": "Identify industries where MoE yields outsized gains.", "personas": ["business_stakeholder"]},
#             {"id": "moe-019", "query": "Explain why MoE can be harder to fine-tune than dense networks.", "personas": ["theorist", "algorithm_engineer"]},
#             {"id": "moe-020", "query": "Give a starter project for students to tinker with small MoE layers.", "personas": ["student"]},
#             {"id": "moe-021", "query": "Discuss theoretical stability of routing under distribution shift.", "personas": ["theorist"]},
#             {"id": "moe-022", "query": "Create a plain-language MoE explainer for executives.", "personas": ["teacher", "business_stakeholder"]},
#             {"id": "moe-023", "query": "List debugging signals to catch failing experts early.", "personas": ["algorithm_engineer"]},
#             {"id": "moe-024", "query": "Compare ROI of MoE vs scaling a dense baseline for a new product.", "personas": ["business_stakeholder"]},
#             {"id": "moe-025", "query": "Propose assessment questions to verify student understanding of MoE.", "personas": ["teacher"]}
#         ],
#         "eval_queries": [
#             {"id": "moe-eval-001", "query": "Propose a monitoring plan to detect expert collapse and routing skew in production.", "persona": "algorithm_engineer"},
#             {"id": "moe-eval-002", "query": "Formally explain how sparse activation reduces compute without severe quality loss.", "persona": "theorist"},
#             {"id": "moe-eval-003", "query": "Design a hands-on classroom activity to teach gating and capacity limits.", "persona": "teacher"},
#             {"id": "moe-eval-004", "query": "Explain MoE’s business ROI and risks in plain language with concrete scenarios.", "persona": "business_stakeholder"}
#         ]
#     },
#     "multi_agent_systems": {
#         "name": "Multi-Agent Systems with LLMs",
#         "seed_queries": [
#             {"id": "mas-001", "query": "Define a multi-agent system in simple terms and give a motivating example.", "personas": ["student", "teacher"]},
#             {"id": "mas-002", "query": "Provide code architecture for a planner–researcher–writer triad of agents.", "personas": ["algorithm_engineer"]},
#             {"id": "mas-003", "query": "State a formal model for message passing and belief updates between agents.", "personas": ["theorist"]},
#             {"id": "mas-004", "query": "Explain business advantages and risks of orchestrating multiple agents.", "personas": ["business_stakeholder"]},
#             {"id": "mas-005", "query": "Create an analogy-based explanation of coordinator vs decentralized agents.", "personas": ["teacher"]},
#             {"id": "mas-006", "query": "Compare star, ring, and mesh topologies for agent communication.", "personas": ["theorist", "algorithm_engineer"]},
#             {"id": "mas-007", "query": "Show how to serialize and log agent messages for audits.", "personas": ["algorithm_engineer"]},
#             {"id": "mas-008", "query": "List common beginner mistakes when building agent teams.", "personas": ["teacher", "student"]},
#             {"id": "mas-009", "query": "Propose evaluation metrics that capture interaction quality, not just final outcomes.", "personas": ["teacher", "theorist"]},
#             {"id": "mas-010", "query": "Describe business cost models for running N agents versus a single stronger model.", "personas": ["business_stakeholder"]},
#             {"id": "mas-011", "query": "Provide pseudocode for turn-taking and arbitration among agents.", "personas": ["algorithm_engineer"]},
#             {"id": "mas-012", "query": "Formally analyze error amplification when agents echo each other’s mistakes.", "personas": ["theorist"]},
#             {"id": "mas-013", "query": "Design a classroom demo where agents critique and refine each other’s solutions.", "personas": ["teacher", "student"]},
#             {"id": "mas-014", "query": "Show safeguards that limit tools and permissions per agent role.", "personas": ["algorithm_engineer"]},
#             {"id": "mas-015", "query": "Discuss theoretical benefits of specialization and division of labor among agents.", "personas": ["theorist"]},
#             {"id": "mas-016", "query": "Schedule multiple agents on limited GPUs while meeting latency SLAs.", "personas": ["algorithm_engineer", "business_stakeholder"]},
#             {"id": "mas-017", "query": "Create prompts that encourage agents to ask clarifying questions to peers.", "personas": ["teacher", "student"]},
#             {"id": "mas-018", "query": "Propose logging granularity balancing privacy with debuggability.", "personas": ["algorithm_engineer", "business_stakeholder"]},
#             {"id": "mas-019", "query": "Formally reason about consensus mechanisms to resolve conflicts.", "personas": ["theorist"]},
#             {"id": "mas-020", "query": "Implement a shared blackboard memory with access control and TTLs.", "personas": ["algorithm_engineer"]},
#             {"id": "mas-021", "query": "Teach newcomers why role drift is dangerous and how to prevent it.", "personas": ["teacher", "student"]},
#             {"id": "mas-022", "query": "Explain escalation-to-human policies and triggers for safety.", "personas": ["teacher", "business_stakeholder"]},
#             {"id": "mas-023", "query": "Give a minimal benchmark for collaborative coding among agents.", "personas": ["student", "algorithm_engineer"]},
#             {"id": "mas-024", "query": "Compare decentralized consensus vs coordinator arbitration in practice.", "personas": ["theorist", "algorithm_engineer"]},
#             {"id": "mas-025", "query": "List business-ready use cases where multi-agent is worth the added complexity.", "personas": ["business_stakeholder"]}
#         ],
#         "eval_queries": [
#             {"id": "mas-eval-001", "query": "Create prompts that encourage agents to ask clarifying questions to peers", "persona": "algorithm_engineer"},
#             {"id": "mas-eval-002", "query": "Design interaction protocols that reduce error amplification and promote peer correction.", "persona": "teacher"},
#             {"id": "mas-eval-003", "query": "Formally reason about consensus mechanisms to resolve conflicts.", "persona": "theorist"},
#             {"id": "mas-eval-004", "query": "List business-ready use cases where multi-agent is worth the added complexity", "persona": "business_stakeholder"}
#         ]
#     }
# }
PERSONAS = {
    "algorithm_engineer": {
        "name": "Algorithm Engineer",
        "profile_text": "Name: Algorithm Engineer. Gender: flexible. Occupation: Implements machine learning algorithms in production code and prototypes; responsible for debugging, profiling, and optimizing. Personality: pragmatic, detail-oriented, hands-on, enjoys measurable improvements. Language style: concise, technical, often includes pseudocode or code pointers. Likes: clean architectures, reproducible experiments, profiling, benchmarks, vectorization. Dislikes: vague advice, untested claims, inefficient loops, missing docs.",
        "follow_up_style": "Ask for concrete implementations, code-level steps, and measurable performance outcomes."
    },
    "student2": {
        "name": "Student 2",
        "profile_text": "Name: Student 2. Gender: flexible. Occupation: A graduate student focusing on applying ML models. Has some practical coding experience but wants to deepen their understanding of model tuning and evaluation. Personality: curious, pragmatic, goal-oriented. Language style: technical and inquisitive, often asking about best practices. Likes: code examples, summaries of common pitfalls, practical tuning tips. Dislikes: overly abstract theories, projects that aren't hands-on.",
        "follow_up_style": "Ask for code examples, common pitfalls, and practical tuning tips."
    },
    "student": {
        "name": "Student",
        "profile_text": "Name: Student. Gender: flexible. Occupation: A beginner in ML/LLMs with incomplete background knowledge. Personality: curious, enthusiastic, sometimes overwhelmed, eager to build small demos. Language style: informal to technical, with lots of clarifying questions. Likes: step-by-step guides, small examples, sanity checks. Dislikes: unexplained jargon, too much math at once, massive compute needs.",
        "follow_up_style": "Ask for beginner-friendly explanations, analogies, and minimal working examples."
    },
    "teacher": {
        "name": "Teacher",
        "profile_text": "Name: Teacher. Gender: flexible. Occupation: Teaches ML/LLMs and designs pedagogy, labs, and assessments. Personality: patient, structured, outcome-focused, values clarity and scaffolding. Language style: clear, analogy-rich, avoids unnecessary jargon. Likes: concept maps, worked examples, formative assessment. Dislikes: confusing leaps, unmotivated notation, skipping prerequisites.",
        "follow_up_style": "Ask for lesson-plan framing, analogies, checks for understanding, and assessment strategies."
    },
    "business_stakeholder": {
        "name": "Business Stakeholder",
        "profile_text": "Name: Business Stakeholder. Gender: flexible. Occupation: Owns product or business outcomes; wants AI benefits without deep technical details. Personality: pragmatic, ROI-driven, risk-aware, prefers plain language and timelines. Language style: concise, focused on outcomes, costs, and risk. Likes: impact metrics, reliability, compliance, time-to-value. Dislikes: technical jargon, unpredictable costs, fragile pipelines.",
        "follow_up_style": "Ask for ROI, risk/benefit trade-offs, timelines, resourcing, and success criteria."
    }
}

TOPICS = {
    "rag": {
        "name": "Retrieval-Augmented Generation (RAG)",
        "seed_queries": [
            {"id": "rag-001", "query": "What is RAG? Why is it important?", "personas": ["student", "teacher"]},
            {"id": "rag-002", "query": "What are the main components of a RAG system?", "personas": ["student", "algorithm_engineer"]},
            {"id": "rag-003", "query": "How do you choose a good embedding model for a RAG system?", "personas": ["algorithm_engineer", "student2"]},
            {"id": "rag-004", "query": "How does the text chunking strategy affect RAG performance?", "personas": ["algorithm_engineer", "student2"]},
            {"id": "rag-005", "query": "What is the difference between using RAG and fine-tuning a model?", "personas": ["student", "teacher", "student2"]},
            {"id": "rag-006", "query": "What are some common problems when building a RAG system?", "personas": ["student", "algorithm_engineer"]},
            {"id": "rag-007", "query": "Explain the business value of RAG in applications like customer service.", "personas": ["business_stakeholder"]},
            {"id": "rag-008", "query": "What is a vector database? Why does RAG need it?", "personas": ["student", "teacher"]},
            {"id": "rag-009", "query": "What is the approximate cost of running a RAG system in production?", "personas": ["business_stakeholder"]},
            {"id": "rag-010", "query": "How do you evaluate the quality of a RAG system?", "personas": ["algorithm_engineer", "student2"]},
            {"id": "rag-011", "query": "What benefits can RAG bring to a company with a large internal knowledge base?", "personas": ["business_stakeholder"]},
            {"id": "rag-012", "query": "Explain RAG to a beginner using a simple analogy.", "personas": ["teacher", "student"]}
        ],
        "eval_queries": [
            {"id": "rag-eval-001", "query": "Design a complete RAG pipeline for a company's internal knowledge base. Detail your choice of embedding model, vector database, and chunking strategy, and explain which Key Performance Indicators (KPIs) you would monitor.", "persona": "algorithm_engineer"},
            {"id": "rag-eval-002", "query": "Create a lesson plan for a one-hour class that explains the fundamental differences between RAG and supervised fine-tuning, including when to choose one over the other. Provide a vivid analogy to help students understand.", "persona": "teacher"},
            {"id": "rag-eval-003", "query": "Our company wants to build a customer support chatbot using our existing product documentation. Present the business case for using a RAG solution, including estimated costs, potential ROI, and the main risks compared to using a generic chatbot service.", "persona": "business_stakeholder"}
        ]
    },
    "classic_ml": {
        "name": "Classic Machine Learning",
        "seed_queries": [
            {"id": "ml-001", "query": "Explain the difference between supervised and unsupervised learning with examples.", "personas": ["student", "teacher"]},
            {"id": "ml-002", "query": "What is overfitting? How can you prevent a model from overfitting?", "personas": ["student", "student2", "algorithm_engineer"]},
            {"id": "ml-003", "query": "How does a decision tree model work?", "personas": ["student", "teacher"]},
            {"id": "ml-004", "query": "What's the difference between Logistic Regression and Support Vector Machines (SVMs)? When should you use one over the other?", "personas": ["student2", "algorithm_engineer"]},
            {"id": "ml-005", "query": "What is feature engineering? Give a few common techniques.", "personas": ["algorithm_engineer", "student2"]},
            {"id": "ml-006", "query": "How do you evaluate a classification model? What are precision and recall?", "personas": ["student", "student2"]},
            {"id": "ml-007", "query": "Explain the concept of the Bias-Variance Tradeoff.", "personas": ["teacher", "student2"]},
            {"id": "ml-008", "query": "In business decisions, which is more important: a model's accuracy or its interpretability?", "personas": ["business_stakeholder"]},
            {"id": "ml-009", "query": "What is the K-Nearest Neighbors (KNN) algorithm?", "personas": ["student", "teacher"]},
            {"id": "ml-010", "query": "What is the difference between Gradient Boosting and Random Forest?", "personas": ["algorithm_engineer", "student2"]},
            {"id": "ml-011", "query": "What is cross-validation and why is it important?", "personas": ["student", "student2"]},
            {"id": "ml-012", "query": "What is the most critical step when starting a new machine learning project?", "personas": ["business_stakeholder", "algorithm_engineer"]}
        ],
        "eval_queries": [
            {"id": "ml-eval-001", "query": "Given a customer churn dataset, describe the end-to-end process of building and evaluating a predictive model. Explain how you would handle feature engineering, prevent overfitting, and choose between logistic regression and a gradient boosting model.", "persona": "algorithm_engineer"},
            {"id": "ml-eval-002", "query": "You are teaching an introductory ML course. Prepare a simple, step-by-step guide for building a decision tree classifier. Use a small, intuitive dataset (e.g., predicting if a fruit is an apple or an orange) and use this example to explain the bias-variance tradeoff.", "persona": "teacher"},
            {"id": "ml-eval-003", "query": "I'm a new data science student. Walk me through the process of evaluating a classification model. Explain what a confusion matrix is, and describe precision, recall, and F1-score in a simple, easy-to-understand way.", "persona": "student"}
        ]
    },
    "llm_finetuning": {
        "name": "Large Language Model Fine-tuning",
        "seed_queries": [
            {"id": "ft-001", "query": "What does it mean to fine-tune a large language model?", "personas": ["student", "teacher"]},
            {"id": "ft-002", "query": "What is LoRA and why is it a popular fine-tuning method?", "personas": ["algorithm_engineer", "student2"]},
            {"id": "ft-003", "query": "How do you prepare a dataset for supervised fine-tuning (SFT)?", "personas": ["student", "algorithm_engineer"]},
            {"id": "ft-004", "query": "What is catastrophic forgetting and how can you avoid it during fine-tuning?", "personas": ["student2", "algorithm_engineer"]},
            {"id": "ft-005", "query": "What are the business trade-offs between using a foundation model's API and fine-tuning your own model?", "personas": ["business_stakeholder"]},
            {"id": "ft-006", "query": "What kind of hardware do you need to fine-tune a 7-billion parameter model?", "personas": ["student", "student2"]},
            {"id": "ft-007", "query": "How do you evaluate a fine-tuned LLM to see if it's better than the base model?", "personas": ["algorithm_engineer", "student2"]},
            {"id": "ft-008", "query": "What are the risks of fine-tuning on low-quality data?", "personas": ["business_stakeholder", "teacher"]},
            {"id": "ft-009", "query": "What is the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT)?", "personas": ["student", "teacher", "student2"]},
            {"id": "ft-010", "query": "Explain what RLHF (Reinforcement Learning from Human Feedback) is.", "personas": ["student2", "teacher"]},
            {"id": "ft-011", "query": "How much time and budget are required to fine-tune a model?", "personas": ["business_stakeholder"]},
            {"id": "ft-012", "query": "What are the most common mistakes beginners make when fine-tuning a model?", "personas": ["student", "teacher"]}
        ],
        "eval_queries": [
            {"id": "ft-eval-001", "query": "Provide a complete, reproducible code plan for fine-tuning a Llama 3 8B model on a custom instruction dataset using LoRA. Specify the data format, key training hyperparameters, and the evaluation strategy you would use to confirm the model has improved.", "persona": "algorithm_engineer"},
            {"id": "ft-eval-002", "query": "As a business leader, I need to decide whether my team should fine-tune an open-source model or use a commercial API like GPT-4. Create a decision framework that outlines the key factors, including costs, data privacy risks, maintenance overhead, and time-to-market.", "persona": "business_stakeholder"},
            {"id": "ft-eval-003", "query": "I'm a student trying to fine-tune my first LLM. Explain the process from start to finish. What are the most common mistakes beginners make, like data formatting errors or catastrophic forgetting, and how can I avoid them?", "persona": "student2"}
        ]
    }
}