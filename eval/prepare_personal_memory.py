#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare personal memory data for evaluation by pre-processing conversations
and storing them in memoryos format.
"""
import os
import sys
import json

# Setup Project Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import memoryos for personal memory
memoryos_path = os.path.join(project_root, "memoryos-pypi")
sys.path.insert(0, memoryos_path)
from memoryos import Memoryos

from sharememory_user.config import Config
from eval.generation_config import PERSONAS

def format_conversation_for_memoryos(conversation):
    """Format conversation data for memoryos ingestion."""
    turns = conversation.get("turns", [])
    
    # Separate user and agent messages
    user_messages = []
    agent_messages = []
    
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        
        if role in ["student", "algorithm_engineer", "theorist", "teacher", "business_stakeholder"]:
            user_messages.append(content)
        else:
            agent_messages.append(content)
    
    # Join messages
    user_input = " ".join(user_messages) if user_messages else "General discussion"
    agent_response = " ".join(agent_messages) if agent_messages else "Acknowledgment and response"
    
    return user_input, agent_response

def prepare_personal_memory():
    """Pre-process personal memory data for all personas."""
    print("🚀 Preparing Personal Memory Data for Evaluation")
    
    config = Config()
    
    # Create data storage directory for memoryos
    memoryos_data_dir = os.path.join(os.path.dirname(__file__), "memoryos_data")
    os.makedirs(memoryos_data_dir, exist_ok=True)
    
    # Load personal memory data
    personal_memory_file = os.path.join(os.path.dirname(__file__), "personal_memory_data.json")
    if not os.path.exists(personal_memory_file):
        print(f"❌ Personal memory data file not found: {personal_memory_file}")
        return False
        
    # Use standard json for reading files
    import json as std_json
    with open(personal_memory_file, 'r', encoding='utf-8') as f:
        personal_memory_data = std_json.load(f)
    
    # Process each persona
    for persona_id in PERSONAS.keys():
        print(f"\n📝 Processing persona: {persona_id}")
        persona_data_dir = os.path.join(memoryos_data_dir, persona_id)
        os.makedirs(persona_data_dir, exist_ok=True)
        
        try:
            # Initialize memoryos instance
            memoryos_instance = Memoryos(
                user_id=persona_id,
                openai_api_key=config.openai_api_key,
                data_storage_path=persona_data_dir,
                openai_base_url=config.openai_api_base,
                llm_model=config.llm_model_name,
                mid_term_heat_threshold=5.0,  # Normal threshold for processing
                embedding_model_name=config.embed_model_name if hasattr(config, 'embed_model_name') else "all-MiniLM-L6-v2"
            )
            
            # Load personal conversations
            if persona_id in personal_memory_data:
                conversations = personal_memory_data[persona_id]
                print(f"  📚 Loading {len(conversations)} conversations...")
                
                for i, conversation in enumerate(conversations, 1):
                    print(f"    Processing conversation {i}/{len(conversations)}...")
                    user_input, agent_response = format_conversation_for_memoryos(conversation)
                    memoryos_instance.add_memory(user_input, agent_response)
                
                # Trigger analysis for knowledge extraction
                print(f"  🧠 Triggering memory analysis...")
                memoryos_instance.force_mid_term_analysis()
                
                print(f"✅ Successfully processed {len(conversations)} conversations for {persona_id}")
            else:
                print(f"⚠️ No personal memory data found for {persona_id}")
                
        except Exception as e:
            print(f"❌ Failed to process {persona_id}: {e}")
            return False
    
    print(f"\n🎉 Personal memory preparation completed!")
    print(f"📁 Data stored in: {memoryos_data_dir}")
    return True

def main():
    """Main function."""
    success = prepare_personal_memory()
    if success:
        print("\n✨ Personal memory data is ready for evaluation!")
        print("💡 You can now run the evaluation with pre-processed memory data.")
    else:
        print("\n❌ Personal memory preparation failed. Please check the errors above.")

if __name__ == "__main__":
    main()
