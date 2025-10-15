# -*- coding: utf-8 -*-
import os
import sys
import time
import warnings
from typing import Any, Dict, List
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup Project Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import orjson as json
except ImportError:
    import json  # type: ignore

try:
    from openai import OpenAI
    from tqdm import tqdm
except ImportError:
    print("Dependencies not found. Please run: pip install openai tqdm")
    sys.exit(1)

from sharememory_user.config import Config
from sharememory_user.pipeline_retrieve import RetrievePipeline, Peer
from sharememory_user.storage import JsonStore
from eval.generation_config import PERSONAS, TOPICS
from eval.evaluation_prompts import get_rag_answer_prompt, get_baseline_answer_prompt, get_judge_prompt, get_fusion_rag_prompt

# Import memoryos for personal memory
import sys
memoryos_path = os.path.join(project_root, "memoryos-pypi")
sys.path.insert(0, memoryos_path)
from memoryos import Memoryos

class EndToEndEvaluator:
    """
    Performs an end-to-end evaluation of the RAG system by comparing its answers
    to a baseline model without retrieval augmentation, and uses a judge model
    to evaluate the quality of answers.
    """
    def __init__(self, cfg: Config, max_workers: int = 4, use_personal_memory: bool = False):
        self.cfg = cfg
        self.retrieve_pipeline = RetrievePipeline(cfg)
        self.store = JsonStore(cfg)
        self.use_personal_memory = use_personal_memory
        if cfg.openai_api_key and "YOUR_API_KEY" not in cfg.openai_api_key:
            # Add a timeout to the client for robustness
            self.llm_client = OpenAI(
                api_key=cfg.openai_api_key, 
                base_url=cfg.openai_api_base,
                timeout=30.0,  # 30-second timeout for API calls
                max_retries=2, # Retry up to 2 times
            )
        else:
            raise ValueError("OpenAI API key is not configured. Please check 'sharememory_user/config.py'.")
        self.results = []
        self.results_lock = threading.Lock()
        self.max_workers = max_workers
        
        # Initialize memoryos instances for each persona
        self.memoryos_instances = {}
        if self.use_personal_memory:
            print("🚀 Personal memory (MemoryOS) is ENABLED.")
            self._init_memoryos_instances(use_precomputed=True)  # Use precomputed memory by default
        else:
            print("⚪️ Personal memory (MemoryOS) is DISABLED.")
        
        # Judge model configuration - can be adjusted as needed
        self.judge_config = {
            "model": "gpt-4o-mini",  # 默认使用强大的 GPT-4 作为裁判
            "temperature": 0.1,      # Lower temperature for more consistent justifications
            "max_tokens": 2500        # Allow space for justification
        }
        
        # Statistics for overall evaluation
        self.rag_wins = 0
        self.baseline_wins = 0
        self.total_evaluations = 0
        self.stats_lock = threading.Lock()
        
    def _init_memoryos_instances(self, use_precomputed=True):
        """Initialize memoryos instances for each persona with personal memory data.
        
        Args:
            use_precomputed (bool): If True, try to load pre-computed memory data.
                                  If False, process personal memory data from scratch.
        """
        # Create data storage directory for memoryos
        memoryos_data_dir = os.path.join(os.path.dirname(__file__), "memoryos_data")
        os.makedirs(memoryos_data_dir, exist_ok=True)
        
        # Load personal memory data
        personal_memory_file = os.path.join(os.path.dirname(__file__), "personal_memory_data.json")
        if not os.path.exists(personal_memory_file):
            print(f"⚠️ Personal memory data file not found: {personal_memory_file}")
            return
            
        # Use standard json for reading files, as orjson doesn't have load() method
        import json as std_json
        with open(personal_memory_file, 'r', encoding='utf-8') as f:
            personal_memory_data = std_json.load(f)
        
        # Initialize memoryos for each persona
        for persona_id in PERSONAS.keys():
            persona_data_dir = os.path.join(memoryos_data_dir, persona_id)
            
            # Check if precomputed data exists
            precomputed_exists = (
                use_precomputed and 
                os.path.exists(persona_data_dir) and 
                any(f.endswith('.json') for f in os.listdir(persona_data_dir) if os.path.isfile(os.path.join(persona_data_dir, f)))
            )
            
            try:
                # Initialize MemoryOS instance following the standard approach
                print(f"🧠 Initializing MemoryOS for {persona_id}...")
                os.makedirs(persona_data_dir, exist_ok=True)
                
                # Use MemoryOS exactly as designed - let it handle everything internally
                memoryos_instance = Memoryos(
                    user_id=persona_id,
                    openai_api_key=self.cfg.openai_api_key,
                    data_storage_path=persona_data_dir,
                    openai_base_url=self.cfg.openai_api_base,
                    llm_model=self.cfg.llm_model_name,
                    assistant_id="evaluation_assistant",
                    # Use MemoryOS default settings - don't override its designed behavior
                    short_term_capacity=3,
                    mid_term_capacity=2000,
                    long_term_knowledge_capacity=100,
                    retrieval_queue_capacity=3,
                    mid_term_heat_threshold=8,  # Use normal threshold - let MemoryOS work as designed
                    mid_term_similarity_threshold=0.7,
                    embedding_model_name=self.cfg.embed_model_name if hasattr(self.cfg, 'embed_model_name') else "all-MiniLM-L6-v2"
                )

                # If pre-computed data does NOT exist, process and add it now.
                # Otherwise, MemoryOS will have loaded the existing data automatically.
                if not precomputed_exists:
                    print(f"  No precomputed data found for {persona_id}. Processing from source...")
                    # Add personal conversations using MemoryOS standard method
                    if persona_id in personal_memory_data:
                        conversations = personal_memory_data[persona_id]
                        print(f"  📚 Processing {len(conversations)} conversations for MemoryOS...")
                        
                        total_turns_added = 0
                        for i, conversation in enumerate(conversations, 1):
                            turn_pairs = self._format_conversation_for_memoryos(conversation)
                            print(f"    Processing conversation {i}: {len(turn_pairs)} turn pairs")
                            
                            for user_input, agent_response in turn_pairs:
                                # Use MemoryOS standard add_memory method - let it handle all the processing
                                memoryos_instance.add_memory(user_input, agent_response)
                                total_turns_added += 1
                        
                        print(f"  ✅ Added {total_turns_added} individual turns across {len(conversations)} conversations")
                else:
                    print(f"  ✅ Found precomputed data for {persona_id}. Loading existing memory.")
                
                print(f"✅ MemoryOS ready for {persona_id}")
                
                self.memoryos_instances[persona_id] = memoryos_instance
                
            except Exception as e:
                print(f"⚠️ Failed to initialize memoryos for {persona_id}: {e}")
                self.memoryos_instances[persona_id] = None
    
    def _format_conversation_for_memoryos(self, conversation):
        """Format conversation data for memoryos ingestion - return individual turn pairs."""
        turns = conversation.get("turns", [])
        
        # Process turns in pairs (user + agent)
        turn_pairs = []
        i = 0
        while i < len(turns) - 1:
            current_turn = turns[i]
            next_turn = turns[i + 1]
            
            # Check if current turn is from persona and next is from AI/colleague
            current_role = current_turn.get("role", "")
            next_role = next_turn.get("role", "")
            
            if (current_role in ["student", "algorithm_engineer", "theorist", "teacher", "business_stakeholder"] and
                next_role in ["ai", "colleague", "teacher"]):
                
                user_input = current_turn.get("content", "")
                agent_response = next_turn.get("content", "")
                turn_pairs.append((user_input, agent_response))
                i += 2  # Skip both turns as they form a pair
            else:
                i += 1  # Move to next turn
        
        # If no pairs found, try to extract meaningful content
        if not turn_pairs:
            user_messages = []
            agent_messages = []
            
            for turn in turns:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                
                if role in ["student", "algorithm_engineer", "theorist", "teacher", "business_stakeholder"]:
                    user_messages.append(content)
                else:
                    agent_messages.append(content)
            
            # Create one pair from all messages
            user_input = " ".join(user_messages) if user_messages else "General discussion"
            agent_response = " ".join(agent_messages) if agent_messages else "Acknowledgment and response"
            turn_pairs.append((user_input, agent_response))
        
        return turn_pairs
    
    def _format_memoryos_retrieval_result(self, memoryos_result):
        """Format MemoryOS retrieval result exactly as provided by the standard retrieval method."""
        if not memoryos_result:
            return ""
            
        formatted_context = ""
        
        # Short-term memory (recent conversations)
        if 'short_term_queue' in memoryos_result and memoryos_result['short_term_queue']:
            formatted_context += "SHORT-TERM MEMORY (Recent Interactions):\n"
            for i, item in enumerate(memoryos_result['short_term_queue'], 1):
                user_input = item.get('user_input', '')
                agent_response = item.get('agent_response', '')
                timestamp = item.get('timestamp', '')
                formatted_context += f"{i}. [{timestamp}] User: {user_input}\n"
                formatted_context += f"   Agent: {agent_response}\n\n"
        
        # Mid-term memory (processed conversations/pages)
        if 'mid_term_pages' in memoryos_result and memoryos_result['mid_term_pages']:
            formatted_context += "MID-TERM MEMORY (Processed Conversations):\n"
            for i, page in enumerate(memoryos_result['mid_term_pages'], 1):
                content = page.get('content', '')
                if content:
                    formatted_context += f"{i}. {content}\n\n"
        
        # User long-term knowledge
        if 'user_knowledge' in memoryos_result and memoryos_result['user_knowledge']:
            formatted_context += "LONG-TERM KNOWLEDGE (Personal Insights):\n"
            for i, knowledge in enumerate(memoryos_result['user_knowledge'], 1):
                knowledge_text = knowledge.get('knowledge', '') or knowledge.get('content', '')
                if knowledge_text:
                    formatted_context += f"{i}. {knowledge_text}\n\n"
        
        # Assistant long-term knowledge
        if 'assistant_knowledge' in memoryos_result and memoryos_result['assistant_knowledge']:
            formatted_context += "ASSISTANT KNOWLEDGE (Domain Expertise):\n"
            for i, knowledge in enumerate(memoryos_result['assistant_knowledge'], 1):
                knowledge_text = knowledge.get('knowledge', '') or knowledge.get('content', '')
                if knowledge_text:
                    formatted_context += f"{i}. {knowledge_text}\n\n"
        
        return formatted_context.strip()
        
    def set_judge_model(self, model_name: str):
        """Allow users to change the judge model."""
        self.judge_config["model"] = model_name
        print(f"Judge model set to: {model_name}")

    def _generate_answer(self, prompt: str) -> str:
        """Calls the LLM to generate a final answer based on a prompt."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.cfg.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=10000,
            )
            return response.choices[0].message.content or "Error: No content returned."
        except Exception as e:
            warnings.warn(f"LLM call for answer generation failed: {e}")
            return f"Error: API call failed. {e}"
            
    def _evaluate_answers(self, query: str, user_profile: str, answer_rag: str, answer_baseline: str) -> Dict:
        """
        Use the judge model to choose the better answer, providing scores and justification.
        Returns a dictionary containing 'winner', 'justification', and scores.
        This method NO LONGER modifies class state, making it thread-safe.
        """
        try:
            # Randomize the order of answers (A/B) to avoid positional bias
            import random
            if random.random() > 0.5:
                answer_a, answer_b = answer_rag, answer_baseline
                is_rag_a = True
            else:
                answer_a, answer_b = answer_baseline, answer_rag
                is_rag_a = False
                
            judge_prompt = get_judge_prompt(query, user_profile, answer_a, answer_b)
            
            response = self.llm_client.chat.completions.create(
                model=self.judge_config["model"],
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=self.judge_config["temperature"],
                max_tokens=self.judge_config["max_tokens"],
            )
            
            evaluation_text = response.choices[0].message.content.strip()
            
            # DEBUG: Print the actual response to understand the format
            print(f"\n=== DEBUG: Judge Response ===")
            print(f"Raw response: {evaluation_text}")
            print("=" * 40)
            
            # Parse justification, scores, and winner from the response
            import re
            justification_match = re.search(r"Justification:\s*(.*?)(?=\s*Score A:)", evaluation_text, re.DOTALL | re.IGNORECASE)
            # Match the exact format we specified in the prompt
            score_a_match = re.search(r"Score A:\s*(\d+(?:\.\d+)?)", evaluation_text, re.IGNORECASE)
            score_b_match = re.search(r"Score B:\s*(\d+(?:\.\d+)?)", evaluation_text, re.IGNORECASE)

            # DEBUG: Show what the regex found
            print(f"DEBUG: justification_match = {justification_match}")
            print(f"DEBUG: score_a_match = {score_a_match}")
            print(f"DEBUG: score_b_match = {score_b_match}")
            if score_a_match:
                print(f"DEBUG: score_a_match.group(1) = '{score_a_match.group(1)}'")
            if score_b_match:
                print(f"DEBUG: score_b_match.group(1) = '{score_b_match.group(1)}'")

            justification = justification_match.group(1).strip() if justification_match else "No justification provided."
            score_a = float(score_a_match.group(1)) if score_a_match else 0.0
            score_b = float(score_b_match.group(1)) if score_b_match else 0.0

            print(f"DEBUG: Parsed score_a = {score_a}, score_b = {score_b}")
            print("=" * 40)

            # Assign scores correctly based on the randomization
            score_rag = score_a if is_rag_a else score_b
            score_baseline = score_b if is_rag_a else score_a

            # Determine winner based on scores for maximum reliability
            if score_rag > score_baseline:
                winner = "RAG"
            elif score_baseline > score_rag:
                winner = "Baseline"
            else:
                # This case should not happen based on the prompt, but as a fallback, we randomly choose.
                winner = "RAG" if random.random() > 0.5 else "Baseline"
                justification += " (Note: Scores were tied, winner chosen randomly as a fallback.)"

            return {"winner": winner, "justification": justification, "score_rag": score_rag, "score_baseline": score_baseline}
            
        except Exception as e:
            warnings.warn(f"Judge evaluation failed: {e}")
            return {"winner": "Error", "justification": f"Evaluation failed due to an error: {str(e)}", "score_rag": 0.0, "score_baseline": 0.0}

    def _process_single_task(self, task: Dict) -> Dict:
        """
        Processes a single evaluation task from retrieval to judgment.
        This method is designed to be executed in a separate thread.
        """
        user_id = task['persona']
        user_profile = self.store.get_user(user_id)
        if not user_profile:
            return {
                "status": "skipped",
                "message": f"⚠️ User '{user_id}' for eval query '{task['id']}' not found. Skipping."
            }
        
        # --- Prepare User Profile & Personal Memory ---
        base_profile_text = user_profile.profile_text
        personal_memory_context = ""
        memoryos_retrieval_result = {}

        # 1. Retrieve personal memory first to get the long-term profile
        if self.use_personal_memory and user_id in self.memoryos_instances and self.memoryos_instances[user_id] is not None:
            try:
                memoryos_instance = self.memoryos_instances[user_id]
                # Use MemoryOS standard retriever.retrieve_context method to get all memory layers
                memoryos_retrieval_result = memoryos_instance.retriever.retrieve_context(
                    user_query=task['query'], user_id=user_id,
                    segment_similarity_threshold=0.1, page_similarity_threshold=0.1,
                    knowledge_threshold=0.01, top_k_sessions=5, top_k_knowledge=2
                )
                
                # 2. Get the long-term user profile directly from MemoryOS
                long_term_profile = memoryos_instance.long_term_memory.get_raw_user_profile(user_id)
                if long_term_profile and long_term_profile != "None":
                    enhanced_profile_text = f"{base_profile_text}\n\n**Long-term User Profile Insights (from MemoryOS):**\n{long_term_profile}"
                else:
                    enhanced_profile_text = base_profile_text
                
                # 3. Format the rest of the personal memory for the context prompt
                # We exclude user_knowledge from here to avoid redundancy with the profile
                context_result = memoryos_retrieval_result.copy()
                context_result.pop('user_knowledge', None)
                personal_memory_context = self._format_memoryos_retrieval_result(context_result)
                print(f"🧠 Retrieved and formatted personal memory for {user_id}: {len(personal_memory_context)} chars")

            except Exception as e:
                print(f"⚠️ Failed to retrieve or process personal memory for {user_id}: {e}")
                personal_memory_context = ""
                enhanced_profile_text = base_profile_text
        else:
            enhanced_profile_text = base_profile_text

        # Create an enhanced UserProfile object for the retrieval pipeline
        from sharememory_user.models import UserProfile
        enhanced_user_profile = UserProfile(user_id=user_id, profile_text=enhanced_profile_text)
        
        # --- RAG Pipeline with Shared Memory ---
        peers = self.retrieve_pipeline.get_cached_peers()
        retrieval_result = self.retrieve_pipeline.retrieve(
            user=enhanced_user_profile, task=task['query'], peers=peers, top_k=3
        )
        shared_memory_context = self.retrieve_pipeline.build_prompt_blocks(retrieval_result['items'])
        
        # Conditionally create the prompt based on whether personal memory is used
        if self.use_personal_memory and personal_memory_context:
            rag_prompt = get_fusion_rag_prompt(
                task['query'], 
                shared_memory_context, 
                personal_memory_context,
                enhanced_profile_text
            )
        else:
            rag_prompt = get_rag_answer_prompt(
                task['query'],
                shared_memory_context,
                enhanced_profile_text
            )
        rag_answer = self._generate_answer(rag_prompt)

        # --- Baseline (No RAG) Pipeline ---
        baseline_prompt = get_baseline_answer_prompt(task['query'], enhanced_profile_text)
        baseline_answer = self._generate_answer(baseline_prompt)

        # --- Judge Evaluation (now thread-safe) ---
        winner_result = self._evaluate_answers(
            task['query'], enhanced_profile_text, rag_answer, baseline_answer
        )

        # --- Result Assembly ---
        clean_retrieved_items = []
        for item in retrieval_result['items']:
            # Create a clean copy without E_m vector data, and include focus_query
            clean_item = {
                "rank": item["rank"],
                "score": item["score"],
                "memory": {
                    "id": item["memory"]["id"],
                    "created_at": item["memory"]["created_at"],
                    "source_user_id": item["memory"]["source_user_id"],
                    "raw_text": item["memory"]["raw_text"],
                    "cot_text": item["memory"].get("cot_text", ""),
                    "focus_query": item["memory"].get("focus_query", ""), # Add focus_query
                    "kg": item["memory"].get("meta", {}).get("kg", []) # Also save the knowledge graph
                }
            }
            clean_retrieved_items.append(clean_item)
            
        result_data = {
            "eval_query_id": task['id'], "user_query": task['query'],
            "persona_id": user_id, "user_profile": enhanced_profile_text,
            "retrieved_context": {
                "shared_memory": clean_retrieved_items,
                "personal_memory_available": self.use_personal_memory and bool(personal_memory_context),
                "personal_memory_preview": personal_memory_context[:200] + "..." if len(personal_memory_context) > 200 else personal_memory_context
            },
            "answer_with_rag": rag_answer, "answer_baseline": baseline_answer,
            "evaluation": {
                "winner": winner_result["winner"],
                "justification": winner_result["justification"],
                "score_rag": winner_result["score_rag"],
                "score_baseline": winner_result["score_baseline"]
            }
        }
        
        return {"status": "success", "data": result_data}

    def run(self):
        """Main loop to run evaluation queries in parallel using a thread pool."""
        print("--- End-to-End RAG Evaluation (Multi-threaded) ---")
        print(f"🔧 Using judge model: {self.judge_config['model']}")
        print(f"🧵 Using {self.max_workers} worker threads")
        
        all_users = self.store.list_users()
        if not all_users:
            print("❌ No users found in storage. Please generate the dataset first.")
            return

        # Pre-compute embeddings for all peers to optimize performance
        all_peers = [Peer(user_id=u.user_id, profile_text=u.profile_text) for u in all_users]
        self.retrieve_pipeline.precompute_peer_embeddings(all_peers)
            
        eval_tasks = [
            eval_query for topic_data in TOPICS.values() 
            for eval_query in topic_data.get('eval_queries', [])
        ]
        print(f"🔹 Found {len(eval_tasks)} evaluation queries to run.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self._process_single_task, task): task for task in eval_tasks}
            
            with tqdm(total=len(eval_tasks), desc="Running Evaluation") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    pbar.update(1)

                    if result["status"] == "success":
                        with self.results_lock:
                            self.results.append(result["data"])
                        
                        winner = result["data"]["evaluation"]["winner"]
                        score_rag = result["data"]["evaluation"]["score_rag"]
                        score_baseline = result["data"]["evaluation"]["score_baseline"]
                        
                        with self.stats_lock:
                            self.total_evaluations += 1
                            if winner == "RAG":
                                self.rag_wins += 1
                            elif winner == "Baseline":
                                self.baseline_wins += 1
                            # No more ties

                        winner_symbol = "🏆" if winner == "RAG" else "❌"
                        tqdm.write(f"🧠 Query '{result['data']['eval_query_id']}': {winner_symbol} Winner: {winner} (RAG: {score_rag:.1f} vs Baseline: {score_baseline:.1f})")
                        tqdm.write(f"  💬 Justification: {result['data']['evaluation']['justification']}\n")
                    else:
                        tqdm.write(result["message"])

        # Save the detailed results
        output_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
        with open(output_path, "wb") as f:
            if hasattr(json, "OPT_INDENT_2"):
                # orjson is used, which returns bytes
                f.write(json.dumps(self.results, option=json.OPT_INDENT_2))
            else:
                # standard json is used, which returns str, so we need to encode.
                f.write(json.dumps(self.results, indent=2).encode("utf-8"))

        # Calculate and display summary statistics
        win_rate = (self.rag_wins / self.total_evaluations) * 100 if self.total_evaluations > 0 else 0
        
        print("\n--- Evaluation Complete ---")
        print(f"✅ Generated and saved {len(self.results)} evaluation comparisons.")
        print(f"✅ Results saved to: {output_path}")
        print("\n📊 Summary Statistics:")
        print(f"   - Total Evaluations: {self.total_evaluations}")
        print(f"   - RAG Wins: {self.rag_wins} ({win_rate:.1f}%)")
        print(f"   - Baseline Wins: {self.baseline_wins} ({100-win_rate:.1f}%)")
        


def main():
    """Main function to set up and run the evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation of the RAG system with judge comparison")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                      help="Model to use for judging answers (e.g., gpt-4-turbo, gpt-3.5-turbo)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads (default: 4)")
    parser.add_argument("--use-personal-memory", action="store_true", 
                      help="Enable the use of personal memory from MemoryOS.")
    args = parser.parse_args()
    
    config = Config()
    evaluator = EndToEndEvaluator(config, max_workers=args.workers, use_personal_memory=args.use_personal_memory)
    
    # Allow customization of the judge model
    if args.judge_model:
        evaluator.set_judge_model(args.judge_model)
        
    evaluator.run()


if __name__ == "__main__":
    main()
