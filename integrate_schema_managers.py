#!/usr/bin/env python3
"""
Integration Script for Schema Managers
Shows how to use all schema managers together in your deep agent system
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchemaManagerOrchestrator:
    """Orchestrates multiple schema managers for optimal performance"""
    
    def __init__(self):
        self.managers = {}
        self.manager_weights = {}
        self.performance_history = {}
        
        # Initialize all available schema managers
        self._initialize_managers()
        
    def _initialize_managers(self):
        """Initialize all available schema managers"""
        try:
            # Schema Chunking Manager
            from schema_managers.schema_chunking import SchemaChunkingManager
            self.managers["chunking"] = SchemaChunkingManager(max_tokens_per_query=5000)
            self.manager_weights["chunking"] = 1.0
            
            logger.info("✅ Schema Chunking Manager initialized")
        except Exception as e:
            logger.warning(f"❌ Schema Chunking Manager failed: {e}")
        
        try:
            # Hierarchical Schema Manager
            from schema_managers.hierarchical_schema import HierarchicalSchemaManager
            self.managers["hierarchical"] = HierarchicalSchemaManager()
            self.manager_weights["hierarchical"] = 1.0
            
            logger.info("✅ Hierarchical Schema Manager initialized")
        except Exception as e:
            logger.warning(f"❌ Hierarchical Schema Manager failed: {e}")
        
        try:
            # Intent-Based Schema Manager
            from schema_managers.intent_based_schema import IntentBasedSchemaSelector
            self.managers["intent"] = IntentBasedSchemaSelector()
            self.manager_weights["intent"] = 1.0
            
            logger.info("✅ Intent-Based Schema Manager initialized")
        except Exception as e:
            logger.warning(f"❌ Intent-Based Schema Manager failed: {e}")
        
        try:
            # Vector Schema Manager
            from schema_managers.vector_schema import VectorSchemaStore
            self.managers["vector"] = VectorSchemaStore()
            self.manager_weights["vector"] = 1.0
            
            logger.info("✅ Vector Schema Manager initialized")
        except Exception as e:
            logger.warning(f"❌ Vector Schema Manager failed: {e}")
        
        try:
            # Progressive Schema Manager
            from schema_managers.progressive_schema import ProgressiveSchemaLoader
            self.managers["progressive"] = ProgressiveSchemaLoader()
            self.manager_weights["progressive"] = 1.0
            
            logger.info("✅ Progressive Schema Manager initialized")
        except Exception as e:
            logger.warning(f"❌ Progressive Schema Manager failed: {e}")
        
        try:
            # LLM Dynamic Schema Manager
            from schema_managers.llm_dynamic_schema import LLMDynamicSchemaManager, LLMProvider
            self.managers["llm_dynamic"] = LLMDynamicSchemaManager(
                llm_provider=LLMProvider.ANTHROPIC,
                max_tokens_per_query=5000
            )
            self.manager_weights["llm_dynamic"] = 1.5  # Higher weight for LLM-powered manager
            
            logger.info("✅ LLM Dynamic Schema Manager initialized")
        except Exception as e:
            logger.warning(f"❌ LLM Dynamic Schema Manager failed: {e}")
        
        logger.info(f"Initialized {len(self.managers)} schema managers")
    
    def get_optimal_schema(self, user_query: str, available_tables: List[str], 
                           full_schema: Dict[str, Any], max_tokens: int = 5000) -> Dict[str, Any]:
        """Get optimal schema using the best performing manager"""
        try:
            # Analyze query characteristics
            query_analysis = self._analyze_query_characteristics(user_query)
            
            # Select best manager based on query characteristics
            selected_manager = self._select_best_manager(query_analysis, user_query)
            
            logger.info(f"Selected manager: {selected_manager}")
            
            # Get schema using selected manager
            start_time = time.time()
            
            if selected_manager == "chunking":
                result = self.managers["chunking"].get_relevant_schema(user_query, available_tables, full_schema)
            elif selected_manager == "hierarchical":
                result = self.managers["hierarchical"].get_adaptive_schema(user_query, full_schema, max_tokens)
            elif selected_manager == "intent":
                result = self.managers["intent"].select_schema_by_intent(user_query, available_tables, full_schema)
            elif selected_manager == "vector":
                result = self.managers["vector"].find_relevant_schema(user_query, top_k=5, similarity_threshold=0.3)
            elif selected_manager == "progressive":
                result = self.managers["progressive"].load_schema_progressively(user_query, max_tokens=max_tokens)
            elif selected_manager == "llm_dynamic":
                result = self.managers["llm_dynamic"].get_adaptive_schema(user_query, available_tables, full_schema)
            else:
                # Fallback to chunking manager
                result = self.managers["chunking"].get_relevant_schema(user_query, available_tables, full_schema)
            
            execution_time = time.time() - start_time
            
            # Update performance history
            self._update_performance_history(selected_manager, execution_time, result)
            
            # Add metadata
            if isinstance(result, dict):
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["selected_manager"] = selected_manager
                result["metadata"]["execution_time"] = execution_time
                result["metadata"]["query_analysis"] = query_analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get optimal schema: {e}")
            # Fallback to basic schema
            return self._get_fallback_schema(user_query, available_tables, full_schema)
    
    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics to determine best manager"""
        query_lower = query.lower()
        
        characteristics = {
            "is_count_query": any(word in query_lower for word in ["count", "how many", "total"]),
            "is_analysis_query": any(word in query_lower for word in ["analyze", "analysis", "examine"]),
            "is_comparison_query": any(word in query_lower for word in ["compare", "difference", "versus"]),
            "is_search_query": any(word in query_lower for word in ["find", "search", "locate"]),
            "is_medicaid_query": any(word in query_lower for word in ["medicaid", "ura", "rebate", "amp", "bp"]),
            "is_domain_specific": any(word in query_lower for word in ["customer", "financial", "healthcare", "ecommerce", "manufacturing"]),
            "query_length": len(query),
            "has_complex_syntax": any(char in query for char in ["(", ")", "JOIN", "GROUP BY", "ORDER BY"])
        }
        
        return characteristics
    
    def _select_best_manager(self, query_analysis: Dict[str, Any], query: str) -> str:
        """Select the best manager based on query analysis"""
        # Enhanced rule-based selection with LLM consideration
        
        if query_analysis["is_domain_specific"] and "llm_dynamic" in self.managers:
            # Domain-specific queries benefit from LLM analysis
            return "llm_dynamic"
        
        elif query_analysis["is_medicaid_query"]:
            # Medicaid queries often benefit from intent-based selection
            return "intent"
        
        elif query_analysis["is_count_query"] or query_analysis["is_search_query"]:
            # Simple queries benefit from chunking
            return "chunking"
        
        elif query_analysis["is_analysis_query"]:
            # Analysis queries benefit from hierarchical approach
            return "hierarchical"
        
        elif query_analysis["is_comparison_query"]:
            # Comparison queries benefit from vector similarity
            return "vector"
        
        elif query_analysis["has_complex_syntax"]:
            # Complex queries benefit from progressive loading
            return "progressive"
        
        elif "llm_dynamic" in self.managers:
            # Default to LLM dynamic for unknown query types
            return "llm_dynamic"
        else:
            # Fallback to chunking
            return "chunking"
    
    def _update_performance_history(self, manager: str, execution_time: float, result: Dict[str, Any]):
        """Update performance history for manager selection"""
        if manager not in self.performance_history:
            self.performance_history[manager] = []
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(execution_time, result)
        
        self.performance_history[manager].append({
            "execution_time": execution_time,
            "performance_score": performance_score,
            "timestamp": time.time()
        })
        
        # Keep only last 100 entries
        if len(self.performance_history[manager]) > 100:
            self.performance_history[manager] = self.performance_history[manager][-100:]
    
    def _calculate_performance_score(self, execution_time: float, result: Dict[str, Any]) -> float:
        """Calculate performance score based on execution time and result quality"""
        # Base score (faster is better)
        time_score = max(0, 1.0 - (execution_time / 10.0))  # Normalize to 0-1
        
        # Quality score based on result metadata
        quality_score = 0.5  # Default
        
        if isinstance(result, dict) and "metadata" in result:
            metadata = result["metadata"]
            
            # Token efficiency
            if "efficiency" in metadata:
                try:
                    efficiency_str = metadata["efficiency"]
                    efficiency = float(efficiency_str.replace("%", "")) / 100.0
                    quality_score = efficiency
                except:
                    pass
            
            # Tables included
            if "tables_included" in metadata:
                table_count = len(metadata["tables_included"])
                if table_count > 0:
                    quality_score = min(1.0, table_count / 10.0)  # More tables = better
            
            # Domain analysis (for LLM dynamic manager)
            if "domain_analysis" in metadata:
                quality_score += 0.2  # Bonus for domain understanding
        
        # Combine scores (70% time, 30% quality)
        final_score = (0.7 * time_score) + (0.3 * quality_score)
        
        return final_score
    
    def _get_fallback_schema(self, user_query: str, available_tables: List[str], 
                             full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback schema when all managers fail"""
        # Return minimal schema for first few tables
        fallback_tables = available_tables[:3]
        fallback_schema = {}
        
        for table in fallback_tables:
            if table in full_schema:
                fallback_schema[table] = {
                    "table_name": table,
                    "description": full_schema[table].get("description", ""),
                    "columns": full_schema[table].get("columns", [])[:5]  # Limit columns
                }
        
        return {
            "schema": fallback_schema,
            "metadata": {
                "selected_manager": "fallback",
                "execution_time": 0.0,
                "fallback": True,
                "tables_included": fallback_tables
            }
        }
    
    def get_manager_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all managers"""
        summary = {}
        
        for manager, history in self.performance_history.items():
            if history:
                execution_times = [entry["execution_time"] for entry in history]
                performance_scores = [entry["performance_score"] for entry in history]
                
                summary[manager] = {
                    "total_queries": len(history),
                    "average_execution_time": sum(execution_times) / len(execution_times),
                    "average_performance_score": sum(performance_scores) / len(performance_scores),
                    "best_performance_score": max(performance_scores),
                    "worst_performance_score": min(performance_scores)
                }
        
        return summary
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for schema manager usage"""
        recommendations = []
        
        performance_summary = self.get_manager_performance_summary()
        
        if not performance_summary:
            return ["No performance data available yet"]
        
        # Find best performing manager
        best_manager = max(performance_summary.keys(), 
                         key=lambda x: performance_summary[x]["average_performance_score"])
        
        recommendations.append(f"Best performing manager: {best_manager}")
        
        # Find fastest manager
        fastest_manager = min(performance_summary.keys(),
                            key=lambda x: performance_summary[x]["average_execution_time"])
        
        recommendations.append(f"Fastest manager: {fastest_manager}")
        
        # Performance improvement suggestions
        for manager, perf in performance_summary.items():
            if perf["average_performance_score"] < 0.5:
                recommendations.append(f"Consider optimizing {manager} manager (score: {perf['average_performance_score']:.2f})")
        
        # LLM-specific recommendations
        if "llm_dynamic" in self.managers:
            recommendations.append("LLM Dynamic Manager available for domain-adaptive schema selection")
        
        return recommendations

def demo_integration():
    """Demonstrate the integrated schema managers"""
    print("🚀 Schema Manager Integration Demo")
    print("=" * 50)
    
    # Create mock schema
    from test_schema_managers import create_mock_schema
    mock_schema = create_mock_schema()
    available_tables = list(mock_schema.keys())
    
    # Initialize orchestrator
    orchestrator = SchemaManagerOrchestrator()
    
    # Test queries
    test_queries = [
        "How many drugs are in the DRUGS table?",
        "Analyze the data quality of the PRICING table for URA calculations",
        "Find discrepancies between calculated and provided URA amounts",
        "Compare AMP vs BP pricing data across different quarters",
        "Show me all tables related to Medicaid drug rebates",
        "Analyze customer data for marketing campaigns",
        "Find financial transaction patterns",
        "Compare different business domains"
    ]
    
    print(f"\n📊 Testing {len(test_queries)} queries with integrated schema managers...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            result = orchestrator.get_optimal_schema(query, available_tables, mock_schema)
            
            if "metadata" in result:
                metadata = result["metadata"]
                print(f"✅ Manager: {metadata.get('selected_manager', 'unknown')}")
                print(f"⏱️  Time: {metadata.get('execution_time', 0):.4f}s")
                print(f"📋 Tables: {len(metadata.get('tables_included', []))}")
                
                if "efficiency" in metadata:
                    print(f"💾 Efficiency: {metadata['efficiency']}")
                
                if "domain" in metadata:
                    print(f"🌍 Domain: {metadata['domain']}")
                
                if "query_intent" in metadata:
                    intent = metadata["query_intent"]
                    if isinstance(intent, dict):
                        print(f"🎯 Intent: {intent.get('intent', 'unknown')}")
                        print(f"📊 Complexity: {intent.get('complexity', 'unknown')}")
            else:
                print(f"✅ Result: {type(result).__name__}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show performance summary
    print("\n" + "=" * 50)
    print("📈 Performance Summary")
    print("=" * 50)
    
    performance_summary = orchestrator.get_manager_performance_summary()
    for manager, perf in performance_summary.items():
        print(f"\n{manager.upper()}:")
        print(f"  Queries: {perf['total_queries']}")
        print(f"  Avg Time: {perf['average_execution_time']:.4f}s")
        print(f"  Avg Score: {perf['average_performance_score']:.2f}")
    
    # Show recommendations
    print("\n💡 Recommendations:")
    recommendations = orchestrator.get_recommendations()
    for rec in recommendations:
        print(f"  • {rec}")

if __name__ == "__main__":
    import time
    demo_integration() 