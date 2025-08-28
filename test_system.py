#!/usr/bin/env python3
"""
Test script for the Deep Agent System
Tests basic functionality without requiring full database connection
"""

import sys
import os
import logging
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from config import settings
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from agents.base_agent import BaseAgent
        print("✅ BaseAgent imported successfully")
    except Exception as e:
        print(f"❌ BaseAgent import failed: {e}")
        return False
    
    try:
        from agents.planner_agent import PlannerAgent
        print("✅ PlannerAgent imported successfully")
    except Exception as e:
        print(f"❌ PlannerAgent import failed: {e}")
        return False
    
    try:
        from agents.query_agent import QueryAgent
        print("✅ QueryAgent imported successfully")
    except Exception as e:
        print(f"❌ QueryAgent import failed: {e}")
        return False
    
    try:
        from agents.data_quality_agent import DataQualityAgent
        print("✅ DataQualityAgent imported successfully")
    except Exception as e:
        print(f"❌ DataQualityAgent import failed: {e}")
        return False
    
    try:
        from agents.dml_agent import DMLAgent
        print("✅ DMLAgent imported successfully")
    except Exception as e:
        print(f"❌ DMLAgent import failed: {e}")
        return False
    
    try:
        from agents.deep_agent import DeepAgent
        print("✅ DeepAgent imported successfully")
    except Exception as e:
        print(f"❌ DeepAgent import failed: {e}")
        return False
    
    return True

def test_agent_creation():
    """Test that agents can be created (with mocked dependencies)"""
    print("\n🔍 Testing agent creation...")
    
    # Mock AWS and Redis dependencies
    with patch('boto3.Session'), patch('redis.Redis'):
        try:
            from agents.planner_agent import PlannerAgent
            planner = PlannerAgent()
            print("✅ PlannerAgent created successfully")
        except Exception as e:
            print(f"❌ PlannerAgent creation failed: {e}")
            return False
        
        try:
            from agents.query_agent import QueryAgent
            query_agent = QueryAgent()
            print("✅ QueryAgent created successfully")
        except Exception as e:
            print(f"❌ QueryAgent creation failed: {e}")
            return False
        
        try:
            from agents.data_quality_agent import DataQualityAgent
            dq_agent = DataQualityAgent()
            print("✅ DataQualityAgent created successfully")
        except Exception as e:
            print(f"❌ DataQualityAgent creation failed: {e}")
            return False
        
        try:
            from agents.dml_agent import DMLAgent
            dml_agent = DMLAgent()
            print("✅ DMLAgent created successfully")
        except Exception as e:
            print(f"❌ DMLAgent creation failed: {e}")
            return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\n🔍 Testing configuration loading...")
    
    try:
        from config import settings
        
        # Check that settings object exists
        if hasattr(settings, 'ORACLE_HOST'):
            print("✅ Configuration loaded successfully")
            print(f"   Oracle Host: {settings.ORACLE_HOST}")
            print(f"   AWS Region: {settings.AWS_REGION}")
            print(f"   Data Directory: {settings.DATA_DIR}")
            return True
        else:
            print("❌ Configuration not properly loaded")
            return False
            
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_directory_creation():
    """Test that required directories can be created"""
    print("\n🔍 Testing directory creation...")
    
    try:
        from config import create_directories
        create_directories()
        
        # Check if directories exist
        from config import settings
        import os
        
        for directory in [settings.DATA_DIR, settings.QUERY_CACHE_DIR, settings.AGENT_MEMORY_DIR]:
            if os.path.exists(directory):
                print(f"✅ Directory created: {directory}")
            else:
                print(f"❌ Directory not created: {directory}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        return False

def test_agent_methods():
    """Test that agents have required methods"""
    print("\n🔍 Testing agent methods...")
    
    # Mock dependencies
    with patch('boto3.Session'), patch('redis.Redis'):
        try:
            from agents.base_agent import BaseAgent
            
            # Test that BaseAgent is abstract
            try:
                BaseAgent("test")
                print("❌ BaseAgent should be abstract")
                return False
            except TypeError:
                print("✅ BaseAgent is properly abstract")
            
            # Test that subclasses can be instantiated
            from agents.planner_agent import PlannerAgent
            planner = PlannerAgent()
            
            # Check required methods
            required_methods = ['process', 'store_memory', 'retrieve_memory', 'clear_memory']
            for method in required_methods:
                if hasattr(planner, method):
                    print(f"✅ Method exists: {method}")
                else:
                    print(f"❌ Missing method: {method}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Agent method testing failed: {e}")
            return False

def main():
    """Run all tests"""
    print("🧪 Deep Agent System - System Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Configuration Loading", test_config_loading),
        ("Directory Creation", test_directory_creation),
        ("Agent Methods", test_agent_methods)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 