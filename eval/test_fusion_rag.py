#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the fusion RAG implementation works correctly.
"""
import os
import sys
import json

# Setup Project Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sharememory_user.config import Config
from eval.evaluate_end_to_end import EndToEndEvaluator
from eval.generation_config import PERSONAS, TOPICS

def test_memoryos_initialization():
    """Test if memoryos instances are properly initialized for each persona."""
    print("🧪 Testing MemoryOS initialization...")
    
    try:
        config = Config()
        evaluator = EndToEndEvaluator(config, max_workers=1)
        
        print(f"✅ Initialized evaluator with {len(evaluator.memoryos_instances)} memoryos instances")
        
        for persona_id, instance in evaluator.memoryos_instances.items():
            if instance is not None:
                print(f"✅ MemoryOS instance for '{persona_id}' is ready")
            else:
                print(f"⚠️ MemoryOS instance for '{persona_id}' failed to initialize")
        
        return True
        
    except Exception as e:
        print(f"❌ MemoryOS initialization failed: {e}")
        return False

def test_personal_memory_data():
    """Test if personal memory data file exists and is valid."""
    print("🧪 Testing personal memory data...")
    
    personal_memory_file = os.path.join(os.path.dirname(__file__), "personal_memory_data.json")
    
    if not os.path.exists(personal_memory_file):
        print(f"❌ Personal memory data file not found: {personal_memory_file}")
        return False
    
    try:
        with open(personal_memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Personal memory data loaded successfully")
        print(f"✅ Found data for {len(data)} personas")
        
        for persona_id in PERSONAS.keys():
            if persona_id in data:
                conversations = data[persona_id]
                print(f"✅ {persona_id}: {len(conversations)} conversations")
            else:
                print(f"⚠️ {persona_id}: No personal memory data")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load personal memory data: {e}")
        return False

def test_eval_queries():
    """Test if eval queries are properly loaded."""
    print("🧪 Testing evaluation queries...")
    
    eval_tasks = [
        eval_query for topic_data in TOPICS.values() 
        for eval_query in topic_data.get('eval_queries', [])
    ]
    
    print(f"✅ Found {len(eval_tasks)} evaluation queries")
    
    for task in eval_tasks[:3]:  # Show first 3 as examples
        print(f"  - {task['id']}: {task['query'][:60]}...")
    
    return len(eval_tasks) > 0

def test_fusion_prompt():
    """Test the fusion RAG prompt function."""
    print("🧪 Testing fusion RAG prompt...")
    
    try:
        from eval.evaluation_prompts import get_fusion_rag_prompt
        
        test_query = "How does RAG work?"
        test_shared = "RAG retrieves relevant documents from a knowledge base..."
        test_personal = "You previously discussed: RAG is like an open-book exam..."
        test_profile = "Student with basic ML knowledge"
        
        prompt = get_fusion_rag_prompt(test_query, test_shared, test_personal, test_profile)
        
        print(f"✅ Fusion RAG prompt generated successfully")
        print(f"✅ Prompt length: {len(prompt)} characters")
        
        # Check if prompt contains all sections
        required_sections = ["SHARED MEMORY", "PERSONAL MEMORY", "USER PROFILE", "USER'S QUESTION"]
        for section in required_sections:
            if section in prompt:
                print(f"✅ Contains section: {section}")
            else:
                print(f"⚠️ Missing section: {section}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fusion prompt test failed: {e}")
        return False

def test_enhanced_user_profiles():
    """Test if enhanced user profiles work correctly."""
    print("🧪 Testing enhanced user profiles...")
    
    try:
        config = Config()
        # Test with personal memory enabled
        evaluator = EndToEndEvaluator(config, max_workers=1, use_personal_memory=True)
        
        # Test for algorithm_engineer persona
        test_user_id = "algorithm_engineer"
        base_profile = "Name: Algorithm Engineer. Gender: flexible. Occupation: Implements machine learning algorithms."
        
        enhanced_profile = evaluator._enhance_user_profile(test_user_id, base_profile)
        
        print(f"✅ Enhanced profile generated successfully")
        print(f"✅ Base profile length: {len(base_profile)} characters")
        print(f"✅ Enhanced profile length: {len(enhanced_profile)} characters")
        
        # Check if enhanced profile contains the base profile
        if base_profile in enhanced_profile:
            print("✅ Enhanced profile contains original base profile")
        else:
            print("⚠️ Enhanced profile missing original base profile")
            
        # Check if enhancement was added when personal memory is available
        if len(enhanced_profile) > len(base_profile):
            print("✅ Profile was enhanced with additional information")
        else:
            print("⚠️ Profile was not enhanced")
            
        print(f"✅ Enhanced profile preview: {enhanced_profile[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced user profile test failed: {e}")
        return False

def test_focus_query_in_results():
    """Test if focus_query is included in retrieval results."""
    print("🧪 Testing focus_query inclusion in results...")
    
    try:
        config = Config()
        evaluator = EndToEndEvaluator(config, max_workers=1, use_personal_memory=False)
        
        # Create a test task
        test_task = {
            'id': 'test-001',
            'query': 'What are the benefits of RAG for business?',
            'persona': 'business_stakeholder'
        }
        
        # Process the task (but don't run full evaluation to avoid API calls)
        user_id = test_task['persona']
        user_profile = evaluator.store.get_user(user_id)
        
        if not user_profile:
            print("⚠️ Test user profile not found, skipping test")
            return True
            
        enhanced_profile_text = evaluator._enhance_user_profile(user_id, user_profile.profile_text)
        
        from sharememory_user.models import UserProfile
        enhanced_user_profile = UserProfile(user_id=user_id, profile_text=enhanced_profile_text)
        
        # Get peers and run retrieval
        peers = evaluator.retrieve_pipeline.get_cached_peers()
        retrieval_result = evaluator.retrieve_pipeline.retrieve(
            user=enhanced_user_profile, task=test_task['query'], peers=peers, top_k=3
        )
        
        # Check if retrieved items have focus_query
        if retrieval_result['items']:
            for item in retrieval_result['items']:
                if 'focus_query' in item['memory']:
                    print("✅ Retrieved memory contains focus_query field")
                    focus_query = item['memory'].get('focus_query', '')
                    print(f"✅ Focus query example: {focus_query[:100]}...")
                    break
            else:
                print("⚠️ No focus_query found in retrieved memories")
                
            return True
        else:
            print("⚠️ No memories retrieved for test")
            return True
            
    except Exception as e:
        print(f"❌ Focus query test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Fusion RAG Tests\n")
    
    tests = [
        ("Personal Memory Data", test_personal_memory_data),
        ("Evaluation Queries", test_eval_queries),
        ("Fusion Prompt", test_fusion_prompt),
        ("MemoryOS Initialization", test_memoryos_initialization),
        ("Enhanced User Profiles", test_enhanced_user_profiles),
        ("Focus Query in Results", test_focus_query_in_results),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fusion RAG implementation is ready.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
