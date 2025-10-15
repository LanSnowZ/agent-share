# -*- coding: utf-8 -*-
import os
import sys
import time
import warnings
from typing import Any, Dict, List
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# --- Setup Project Path ---
# This ensures that the script can find and import modules from the main 'sharememory_user' package.
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
from sharememory_user.pipeline_ingest import IngestPipeline
from eval.generation_config import PERSONAS, TOPICS
# The line below is no longer needed as prompts are now generated dynamically.
# from eval.generation_prompts import get_dialogue_generation_prompt

class DatasetGenerator:
    """
    Generates a synthetic dataset of conversations and ingests them into the memory store.
    Also creates a ground truth file for future evaluation.
    Enhanced with multi-threading and retry mechanisms for improved performance and reliability.
    """
    def __init__(self, cfg: Config, max_workers: int = 4):
        self.cfg = cfg
        self.ingest_pipeline = IngestPipeline(cfg)
        if cfg.openai_api_key and "YOUR_API_KEY" not in cfg.openai_api_key:
            self.llm_client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_api_base)
        else:
            raise ValueError("OpenAI API key is not configured. Please check 'sharememory_user/config.py' or environment variables.")
        self.ground_truth = []
        self.ground_truth_lock = threading.Lock()
        self.max_workers = max_workers

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.5, max_retries: int = 3) -> str:
        """Generic LLM caller with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.cfg.llm_model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1500,  # Adjusted for single turns
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                if attempt < max_retries - 1:
                    warnings.warn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    warnings.warn(f"LLM call failed after {max_retries} attempts: {e}")
                    return ""

    def _generate_conversation_turn_by_turn(
        self, topic: Dict, query_info: Dict, persona_id: str, num_turns: int = 3
    ) -> str:
        """Generates a dialogue by simulating a turn-by-turn conversation between two AI agents."""
        persona = PERSONAS[persona_id]

        ai_expert_system_prompt = (
            f"You are a world-class AI research scientist, capable of explaining complex topics "
            f"like '{topic['name']}' with clarity and depth. Provide comprehensive and accurate answers."
        )
        user_persona_system_prompt = (
            f"You are role-playing as a '{persona['name']}'.\n"
            f"Your background is: '{persona['profile_text']}'.\n"
            f"Your task is to ask insightful questions about '{topic['name']}' based on the ongoing conversation.\n"
            f"Your follow-up questions must be highly characteristic of your persona and follow this style: '{persona['follow_up_style']}'.\n"
            f"IMPORTANT: Only output the user's question itself, without any preamble like 'User:' or quotes."
        )

        conversation_log: List[str] = []
        messages: List[Dict[str, str]] = []

        # --- Turn 1 ---
        current_query = query_info["query"]
        conversation_log.append(f"User: {current_query}")
        
        # AI Expert answers the initial query
        messages = [
            {"role": "system", "content": ai_expert_system_prompt},
            {"role": "user", "content": current_query},
        ]
        ai_answer = self._call_llm(messages)
        if not ai_answer: return ""  # Stop if AI fails to answer
        
        conversation_log.append(f"AI: {ai_answer}")
        messages.append({"role": "assistant", "content": ai_answer})

        # --- Subsequent Turns ---
        for _ in range(1, num_turns):
            # Persona User asks a follow-up question
            messages[0] = {"role": "system", "content": user_persona_system_prompt}
            
            user_instruction = "Based on the conversation history above, generate your next single follow-up question now."
            messages.append({"role": "user", "content": user_instruction})

            user_follow_up = self._call_llm(messages, temperature=0.7)
            messages.pop()  # Remove the temporary instruction
            if not user_follow_up: break  # Stop if user fails to ask

            conversation_log.append(f"User: {user_follow_up}")
            messages.append({"role": "user", "content": user_follow_up})

            # AI Expert answers the follow-up
            messages[0] = {"role": "system", "content": ai_expert_system_prompt}
            ai_follow_up_answer = self._call_llm(messages)
            if not ai_follow_up_answer: break  # Stop if AI fails to answer

            conversation_log.append(f"AI: {ai_follow_up_answer}")
            messages.append({"role": "assistant", "content": ai_follow_up_answer})
            # Remove sleep from here as we're using threading

        return "\n".join(conversation_log)

    def _process_single_task(self, task: tuple) -> Dict[str, Any]:
        """Process a single generation task. Returns result dict for thread safety."""
        topic, query, persona_id = task
        
        try:
            dialogue_text = self._generate_conversation_turn_by_turn(topic, query, persona_id)
            
            if not dialogue_text or len(dialogue_text) < 150:
                return {
                    "success": False,
                    "message": f"⚠️ Skipping empty or short dialogue for query '{query['id']}' by '{persona_id}'.",
                    "query_id": query['id'],
                    "persona_id": persona_id
                }

            # Ingest the generated dialogue into the memory store
            memory_item = self.ingest_pipeline.ingest_dialog(
                source_user_id=persona_id,
                raw_text=dialogue_text
            )

            if memory_item:
                ground_truth_entry = {
                    "query_id": query['id'],
                    "query_text": query['query'],
                    "topic": topic['name'],
                    "persona_id": persona_id,
                    "generated_memory_id": memory_item.id,
                }
                return {
                    "success": True,
                    "message": f"✔️ Ingested memory {memory_item.id} for query '{query['id']}' by '{persona_id}'.",
                    "ground_truth_entry": ground_truth_entry,
                    "query_id": query['id'],
                    "persona_id": persona_id
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ QC rejected dialogue for query '{query['id']}' by '{persona_id}'.",
                    "query_id": query['id'],
                    "persona_id": persona_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Error processing query '{query['id']}' by '{persona_id}': {str(e)}",
                "query_id": query['id'],
                "persona_id": persona_id
            }

    def run(self):
        """Main loop to generate and ingest all data using multi-threading."""
        print("--- Synthetic Dataset Generation (Multi-threaded) ---")
        print(f"🧵 Using {self.max_workers} worker threads")
        
        # 1. Ensure all personas are registered as users in the system
        for persona_id, persona_data in PERSONAS.items():
            self.ingest_pipeline.ensure_user(user_id=persona_id, profile_text=persona_data['profile_text'])
        print(f"✅ Registered {len(PERSONAS)} user personas.")

        # 2. Assemble all generation tasks based on the config
        generation_tasks = []
        for topic_id, topic_data in TOPICS.items():
            for query_info in topic_data['seed_queries']:
                for persona_id in query_info['personas']:
                    generation_tasks.append((topic_data, query_info, persona_id))
        
        print(f"🔹 Found {len(generation_tasks)} dialogues to generate.")

        # 3. Execute generation and ingestion using ThreadPoolExecutor
        successful_tasks = 0
        failed_tasks = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(self._process_single_task, task): task for task in generation_tasks}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(generation_tasks), desc="Generating Dialogues") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Log the result
                    tqdm.write(result["message"])
                    
                    # Handle successful results
                    if result["success"] and "ground_truth_entry" in result:
                        with self.ground_truth_lock:
                            self.ground_truth.append(result["ground_truth_entry"])
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
                    
                    # Small delay to be nice to the API
                    time.sleep(0.1)

        # 4. Save the ground truth file for future evaluation
        output_path = os.path.join(os.path.dirname(__file__), "ground_truth_and_eval_queries.json")
        with open(output_path, "wb") as f:
            # Use orjson for efficient writing if available
            if hasattr(json, "OPT_INDENT_2"):
                # orjson is used, which returns bytes
                f.write(json.dumps(self.ground_truth, option=json.OPT_INDENT_2))
            else:
                # standard json is used, which returns str, so we need to encode.
                f.write(json.dumps(self.ground_truth, indent=2).encode("utf-8"))
        
        print("\n--- Generation Complete ---")
        print(f"✅ Successfully generated and ingested {successful_tasks} memories.")
        print(f"❌ Failed to generate {failed_tasks} memories.")
        print(f"📊 Success rate: {(successful_tasks / len(generation_tasks)) * 100:.1f}%")
        print(f"✅ Ground truth map saved to: {output_path}")


def main():
    """Main function to set up and run the generator."""
    print("Initializing multi-threaded dataset generator...")
    
    # Parse command line arguments for thread count
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic dataset with multi-threading")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    args = parser.parse_args()
    
    # Load configuration from the main project
    config = Config()
    
    # It's a good practice to clear previous data before generating a new set
    print("⚠️  This will clear all existing data in 'data/memory.json' and 'data/users.json'.")
    confirm = input("    Are you sure you want to continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    from sharememory_user.storage import JsonStore
    store = JsonStore(config)
    store.clear_all()
    print("🔹 Cleared existing memory and user data.")
    
    generator = DatasetGenerator(config, max_workers=args.workers)
    generator.run()


if __name__ == "__main__":
    main()