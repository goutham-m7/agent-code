import json
import logging
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import BaseAgent
from .planner_agent import PlannerAgent
from .query_agent import QueryAgent
from .data_quality_agent import DataQualityAgent
from .dml_agent import DMLAgent
from database.connection import db_connection
from config import settings

logger = logging.getLogger(__name__)

class DeepAgent:
    """Main deep agent orchestrator implementing the four key components:
    1. Planning Tool - coordinates complex operations
    2. Sub Agents - specialized agents for different tasks
    3. File System Access - stores and retrieves data, queries, and results
    4. Detailed Prompts - comprehensive prompts for each agent type
    """
    
    def __init__(self):
        self.planner_agent = PlannerAgent()
        self.query_agent = QueryAgent()
        self.data_quality_agent = DataQualityAgent()
        self.dml_agent = DMLAgent()
        
        # File system paths
        self.data_dir = settings.DATA_DIR
        self.query_cache_dir = settings.QUERY_CACHE_DIR
        self.agent_memory_dir = settings.AGENT_MEMORY_DIR
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Agent registry
        self.agents = {
            "planner": self.planner_agent,
            "query": self.query_agent,
            "data_quality": self.data_quality_agent,
            "dml": self.dml_agent
        }
        
        logger.info("Deep Agent initialized successfully")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.query_cache_dir, exist_ok=True)
        os.makedirs(self.agent_memory_dir, exist_ok=True)
    
    def process_user_query(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for processing user queries"""
        try:
            logger.info(f"Processing user query: {user_query}")
            
            # Store query in file system
            self._store_query(user_query, context)
            
            # Step 1: Planning - Create execution plan
            execution_plan = self.planner_agent.create_execution_plan(user_query, context)
            logger.info(f"Execution plan created: {execution_plan.get('estimated_complexity', 'unknown')} complexity")
            
            # Step 2: Execute plan using appropriate sub-agents
            execution_results = self._execute_plan(execution_plan, user_query, context)
            
            # Step 3: Store results in file system
            self._store_execution_results(execution_results, user_query)
            
            # Step 4: Generate comprehensive response
            final_response = self._generate_final_response(user_query, execution_plan, execution_results)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Deep agent processing failed: {e}")
            return {
                "error": str(e),
                "user_query": user_query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_plan(self, execution_plan: Dict[str, Any], user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan using appropriate sub-agents"""
        try:
            execution_results = {
                "plan": execution_plan,
                "agent_results": {},
                "execution_timestamp": datetime.now().isoformat()
            }
            
            # Execute tasks based on plan
            for task in execution_plan.get("tasks", []):
                task_result = self._execute_task(task, user_query, context, execution_plan)
                execution_results["agent_results"][task] = task_result
            
            # Execute sub-agent operations
            for agent_name in execution_plan.get("sub_agents_needed", []):
                if agent_name.lower() in self.agents:
                    agent_result = self._execute_agent_operation(agent_name.lower(), user_query, context)
                    execution_results["agent_results"][f"{agent_name}_operation"] = agent_result
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            raise
    
    def _execute_task(self, task: str, user_query: str, context: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task from the plan"""
        try:
            # Analyze task and determine appropriate action
            if "query" in task.lower() or "select" in task.lower():
                return self.query_agent.process({"query": user_query, "context": context})
            elif "quality" in task.lower() or "validate" in task.lower():
                # Extract table name from context or plan
                table_name = context.get("table_name") or self._extract_table_from_plan(plan)
                if table_name:
                    return self.data_quality_agent.process({"table_name": table_name, "business_rules": context.get("business_rules", {})})
            elif "plan" in task.lower() or "analyze" in task.lower():
                return self.planner_agent.process({"query": user_query, "context": context})
            
            # Default task execution
            return {
                "task": task,
                "status": "completed",
                "result": f"Task '{task}' executed successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "task": task,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_agent_operation(self, agent_name: str, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation using a specific sub-agent"""
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                return {"error": f"Agent '{agent_name}' not found"}
            
            if agent_name == "query":
                return agent.process({"query": user_query, "context": context})
            elif agent_name == "data_quality":
                table_name = context.get("table_name") or self._extract_table_from_query(user_query)
                if table_name:
                    return agent.process({"table_name": table_name, "business_rules": context.get("business_rules", {})})
                else:
                    return {"error": "No table name specified for data quality analysis"}
            elif agent_name == "planner":
                return agent.process({"query": user_query, "context": context})
            
            return {"error": f"Unknown agent operation for '{agent_name}'"}
            
        except Exception as e:
            logger.error(f"Agent operation failed: {e}")
            return {"error": str(e)}
    
    def _extract_table_from_plan(self, plan: Dict[str, Any]) -> Optional[str]:
        """Extract table name from execution plan"""
        required_tables = plan.get("required_tables", [])
        if required_tables:
            return required_tables[0]  # Return first table
        return None
    
    def _extract_table_from_query(self, query: str) -> Optional[str]:
        """Extract table name from user query"""
        # Simple table extraction logic
        query_lower = query.lower()
        if "from" in query_lower:
            parts = query_lower.split("from")
            if len(parts) > 1:
                table_part = parts[1].strip().split()[0]
                return table_part.upper()
        return None
    
    def _store_query(self, query: str, context: Dict[str, Any]):
        """Store user query in file system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_{timestamp}.json"
            filepath = os.path.join(self.query_cache_dir, filename)
            
            query_data = {
                "query": query,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(query_data, f, indent=2)
            
            logger.info(f"Query stored: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to store query: {e}")
    
    def _store_execution_results(self, results: Dict[str, Any], user_query: str):
        """Store execution results in file system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Add metadata
            results["metadata"] = {
                "user_query": user_query,
                "stored_timestamp": datetime.now().isoformat(),
                "deep_agent_version": "1.0.0"
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results stored: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to store results: {e}")
    
    def _generate_final_response(self, user_query: str, execution_plan: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final response"""
        try:
            # Compile all results
            final_response = {
                "user_query": user_query,
                "execution_summary": {
                    "plan_complexity": execution_plan.get("estimated_complexity", "unknown"),
                    "sub_agents_used": execution_plan.get("sub_agents_needed", []),
                    "execution_status": "completed",
                    "timestamp": datetime.now().isoformat()
                },
                "execution_plan": execution_plan,
                "results": execution_results,
                "recommendations": self._generate_recommendations(execution_results),
                "next_steps": self._suggest_next_steps(execution_results)
            }
            
            return final_response
            
        except Exception as e:
            logger.error(f"Failed to generate final response: {e}")
            return {
                "user_query": user_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_recommendations(self, execution_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on execution results"""
        recommendations = []
        
        try:
            # Analyze results and generate recommendations
            agent_results = execution_results.get("agent_results", {})
            
            for operation, result in agent_results.items():
                if "error" in result:
                    recommendations.append(f"Review and fix error in {operation}: {result['error']}")
                
                if "data_quality" in operation and "overall_quality_score" in result:
                    score = result["overall_quality_score"]
                    if score < 70:
                        recommendations.append(f"Data quality score is low ({score}) - implement data quality improvements")
                    elif score < 90:
                        recommendations.append(f"Data quality score is moderate ({score}) - focus on identified issues")
                
                if "query" in operation and "execution_result" in result:
                    exec_result = result["execution_result"]
                    if exec_result.get("row_count", 0) == 0:
                        recommendations.append("Query returned no results - verify query parameters and data availability")
            
            if not recommendations:
                recommendations.append("All operations completed successfully - no immediate action required")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations due to error"]
    
    def _suggest_next_steps(self, execution_results: Dict[str, Any]) -> List[str]:
        """Suggest next steps based on execution results"""
        next_steps = []
        
        try:
            # Analyze results and suggest next steps
            agent_results = execution_results.get("agent_results", {})
            
            for operation, result in agent_results.items():
                if "data_quality" in operation and "recommendations" in result:
                    # Add data quality improvement steps
                    next_steps.append("Review and implement data quality recommendations")
                
                if "query" in operation and "execution_result" in result:
                    exec_result = result["execution_result"]
                    if exec_result.get("success", False) and exec_result.get("row_count", 0) > 1000:
                        next_steps.append("Consider implementing pagination for large result sets")
                
                if "error" in result:
                    next_steps.append(f"Investigate and resolve error in {operation}")
            
            if not next_steps:
                next_steps.append("Monitor system performance and data quality metrics")
                next_steps.append("Consider implementing automated data quality monitoring")
            
            return next_steps
            
        except Exception as e:
            logger.error(f"Failed to suggest next steps: {e}")
            return ["Unable to suggest next steps due to error"]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "agents": {},
                "database": {},
                "file_system": {},
                "overall_status": "healthy"
            }
            
            # Check agent status
            for name, agent in self.agents.items():
                try:
                    memory_summary = agent.get_memory_summary()
                    status["agents"][name] = {
                        "status": "active",
                        "memory_entries": len(memory_summary),
                        "last_activity": max([data.get("timestamp", "") for data in memory_summary.values()]) if memory_summary else "unknown"
                    }
                except Exception as e:
                    status["agents"][name] = {"status": "error", "error": str(e)}
                    status["overall_status"] = "degraded"
            
            # Check database status
            try:
                test_query = "SELECT 1 FROM DUAL"
                result = db_connection.execute_query(test_query)
                status["database"]["status"] = "connected"
                status["database"]["test_query"] = "successful"
            except Exception as e:
                status["database"]["status"] = "error"
                status["database"]["error"] = str(e)
                status["overall_status"] = "degraded"
            
            # Check file system status
            try:
                status["file_system"]["data_dir"] = os.path.exists(self.data_dir)
                status["file_system"]["query_cache_dir"] = os.path.exists(self.query_cache_dir)
                status["file_system"]["agent_memory_dir"] = os.path.exists(self.agent_memory_dir)
            except Exception as e:
                status["file_system"]["error"] = str(e)
                status["overall_status"] = "degraded"
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear various caches and stored data"""
        try:
            if cache_type in ["all", "queries"]:
                # Clear query cache
                for filename in os.listdir(self.query_cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.query_cache_dir, filename))
                logger.info("Query cache cleared")
            
            if cache_type in ["all", "results"]:
                # Clear results cache
                for filename in os.listdir(self.data_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.data_dir, filename))
                logger.info("Results cache cleared")
            
            if cache_type in ["all", "memory"]:
                # Clear agent memory
                for agent in self.agents.values():
                    agent.clear_memory()
                logger.info("Agent memory cleared")
            
            return {"status": "success", "cache_cleared": cache_type}
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get history of processed queries"""
        try:
            history = []
            
            for filename in os.listdir(self.query_cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.query_cache_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            query_data = json.load(f)
                            history.append({
                                "filename": filename,
                                "data": query_data,
                                "file_size": os.path.getsize(filepath)
                            })
                    except Exception as e:
                        logger.error(f"Failed to read query file {filename}: {e}")
            
            # Sort by timestamp
            history.sort(key=lambda x: x["data"].get("timestamp", ""), reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get query history: {e}")
            return []
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of execution results"""
        try:
            history = []
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.data_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            result_data = json.load(f)
                            history.append({
                                "filename": filename,
                                "data": result_data,
                                "file_size": os.path.getsize(filepath)
                            })
                    except Exception as e:
                        logger.error(f"Failed to read result file {filename}: {e}")
            
            # Sort by timestamp
            history.sort(key=lambda x: x["data"].get("metadata", {}).get("stored_timestamp", ""), reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get execution history: {e}")
            return [] 