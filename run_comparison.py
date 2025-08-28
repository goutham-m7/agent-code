#!/usr/bin/env python3
"""
Quick Comparison Runner
Runs all schema managers and shows results in a simple format
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🎯 Schema Manager Comparison")
    print("=" * 50)
    print()
    
    print("📋 Available Solutions:")
    print("1. Schema Chunking - Simple entity extraction")
    print("2. Hierarchical Schema - Multiple detail levels")
    print("3. Intent-Based - Query classification")
    print("4. Vector Search - Semantic similarity")
    print("5. Progressive Loading - Adaptive complexity")
    print()
    
    print("🚀 Running Tests...")
    print()
    
    try:
        # Run the comprehensive test
        from test_schema_managers import main as run_tests
        result = run_tests()
        
        if result == 0:
            print("✅ All tests completed successfully!")
        else:
            print("⚠️  Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        print("\n💡 Try installing dependencies first:")
        print("   pip install -r requirements.txt")
        return 1
    
    print()
    print("🔍 For detailed analysis, check:")
    print("   • test_schema_managers.py - Individual manager tests")
    print("   • integrate_schema_managers.py - Integrated approach")
    print("   • schema_managers/README.md - Complete documentation")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 