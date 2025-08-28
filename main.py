#!/usr/bin/env python3
"""
Deep Agent System - Main Application
A comprehensive deep agent system for Oracle database operations with AWS Bedrock integration.

This system implements the four key components:
1. Planning Tool - coordinates complex operations
2. Sub Agents - specialized agents for different tasks
3. File System Access - stores and retrieves data, queries, and results
4. Detailed Prompts - comprehensive prompts for each agent type
"""

import os
import sys
import logging
import json
from typing import Dict, Any
from agents.deep_agent import DeepAgent
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner"""
    print("=" * 80)
    print("🤖 Deep Agent System - Oracle Database + AWS Bedrock Integration")
    print("=" * 80)
    print("Four Key Components:")
    print("1. 📋 Planning Tool - Intelligent task planning and coordination")
    print("2. 🔧 Sub Agents - Specialized agents for different operations")
    print("3. 💾 File System Access - Persistent storage and caching")
    print("4. 🎯 Detailed Prompts - AI-powered decision making")
    print("=" * 80)

def check_environment():
    """Check if required environment variables are set"""
    required_vars = [
        'ORACLE_HOST', 'ORACLE_USER', 'ORACLE_PASSWORD',
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(settings, var, None):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
    
    print("✅ Environment configuration verified")
    return True

def interactive_mode(deep_agent: DeepAgent):
    """Run the deep agent in interactive mode"""
    print("\n🚀 Starting Interactive Mode")
    print("Type 'help' for available commands, 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🤖 Deep Agent > ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
            
            elif user_input.lower() == 'status':
                status = deep_agent.get_system_status()
                print(f"📊 System Status: {status['overall_status']}")
                print(f"🔌 Database: {status['database']['status']}")
                print(f"🤖 Agents: {len(status['agents'])} active")
            
            elif user_input.lower() == 'history':
                queries = deep_agent.get_query_history()
                executions = deep_agent.get_execution_history()
                print(f"📝 Query History: {len(queries)} queries")
                print(f"⚡ Execution History: {len(executions)} executions")
            
            elif user_input.lower() == 'clear':
                result = deep_agent.clear_cache()
                print(f"🧹 Cache cleared: {result['status']}")
            
            elif user_input.lower().startswith('quality'):
                # Data quality analysis
                parts = user_input.split()
                if len(parts) > 1:
                    table_name = parts[1]
                    print(f"🔍 Analyzing data quality for table: {table_name}")
                    result = deep_agent.data_quality_agent.analyze_data_quality(table_name)
                    print(f"📊 Quality Score: {result.get('overall_quality_score', 'N/A')}")
                    print(f"💡 Recommendations: {len(result.get('recommendations', []))}")
                else:
                    print("❌ Please specify a table name: quality <table_name>")
            
            elif user_input.lower().startswith('query'):
                # Direct query execution
                parts = user_input.split(' ', 1)
                if len(parts) > 1:
                    query = parts[1]
                    print(f"🔍 Executing query: {query}")
                    result = deep_agent.query_agent.process({"query": query})
                    if "error" in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        print(f"✅ Query executed successfully")
                        print(f"📊 Rows returned: {result.get('execution_result', {}).get('row_count', 0)}")
                else:
                    print("❌ Please provide a query: query <your_query>")
            
            elif user_input:
                # Process user query through deep agent
                print(f"🤔 Processing: {user_input}")
                result = deep_agent.process_user_query(user_input)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("✅ Query processed successfully!")
                    print(f"📋 Plan Complexity: {result.get('execution_summary', {}).get('plan_complexity', 'unknown')}")
                    print(f"🤖 Agents Used: {', '.join(result.get('execution_summary', {}).get('sub_agents_used', []))}")
                    
                    # Show recommendations
                    recommendations = result.get('recommendations', [])
                    if recommendations:
                        print("\n💡 Recommendations:")
                        for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                            print(f"   {i}. {rec}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Interactive mode error: {e}")

def print_help():
    """Print available commands"""
    print("\n📚 Available Commands:")
    print("  help                    - Show this help message")
    print("  status                  - Show system status")
    print("  history                 - Show query and execution history")
    print("  clear                   - Clear all caches")
    print("  quality <table_name>    - Analyze data quality for a table")
    print("  query <your_query>     - Execute a direct query")
    print("  <natural language>      - Process natural language query through deep agent")
    print("  quit/exit/q            - Exit the application")

def demo_mode(deep_agent: DeepAgent):
    """Run the deep agent in demo mode with predefined examples"""
    print("\n🎬 Starting Demo Mode")
    print("Running predefined examples to demonstrate capabilities")
    print("-" * 50)
    
    demo_queries = [
        "Show me all tables in the database",
        "What is the data quality of the CUSTOMERS table?",
        "Find all orders from last month",
        "Analyze the structure of the PRODUCTS table"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Demo {i}: {query}")
        print("-" * 30)
        
        try:
            result = deep_agent.process_user_query(query)
            
            if "error" in result:
                print(f"❌ Demo {i} failed: {result['error']}")
            else:
                print(f"✅ Demo {i} completed successfully")
                summary = result.get('execution_summary', {})
                print(f"   Complexity: {summary.get('plan_complexity', 'unknown')}")
                print(f"   Agents: {', '.join(summary.get('sub_agents_used', []))}")
        
        except Exception as e:
            print(f"❌ Demo {i} error: {e}")
        
        print()
    
    print("🎬 Demo mode completed!")

def main():
    """Main application entry point"""
    try:
        print_banner()
        
        # Check environment configuration
        if not check_environment():
            sys.exit(1)
        
        print("\n🔧 Initializing Deep Agent System...")
        
        # Initialize the deep agent
        deep_agent = DeepAgent()
        
        print("✅ Deep Agent System initialized successfully!")
        
        # Check system status
        status = deep_agent.get_system_status()
        print(f"📊 System Status: {status['overall_status']}")
        
        # Determine mode
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            
            if mode == "demo":
                demo_mode(deep_agent)
            elif mode == "interactive":
                interactive_mode(deep_agent)
            else:
                print(f"❌ Unknown mode: {mode}")
                print("Available modes: demo, interactive")
                sys.exit(1)
        else:
            # Default to interactive mode
            print("\n💡 Starting in interactive mode (use 'demo' or 'interactive' as argument)")
            interactive_mode(deep_agent)
    
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 