import json
import logging
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """Agent responsible for planning and coordinating complex database operations"""
    
    def __init__(self):
        super().__init__("Planner")
        self.system_prompt = self._get_planner_system_prompt()
    
    def _get_planner_system_prompt(self) -> str:
        """Get the system prompt for the planner agent"""
        return """You are a Database Query Planning Agent. Your role is to:

1. Analyze user queries and break them down into logical steps
2. Determine which database tables and schemas are relevant
3. Plan the sequence of operations needed
4. Identify potential data quality issues or discrepancies
5. Coordinate with specialized sub-agents for execution

You should:
- Think step by step about complex queries
- Consider database schema and relationships
- Plan for data validation and quality checks
- Break down complex operations into manageable tasks
- Provide clear instructions for sub-agents

Always respond with a structured plan that includes:
- Task breakdown
- Required tables/schemas
- Data quality considerations
- Execution steps
- Expected outcomes"""

    def create_execution_plan(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an execution plan for the user query"""
        try:
            # Store the query in memory
            self.store_memory("last_query", user_query)
            
            # Create planning prompt
            planning_prompt = f"""
User Query: {user_query}

Context: {context or 'No additional context provided'}

Please create a detailed execution plan for this database operation. Consider:
1. What tables and schemas might be involved?
2. What type of operation is this (SELECT, INSERT, UPDATE, DELETE, analysis)?
3. What are the potential data quality issues to check?
4. What steps are needed to complete this operation?
5. What sub-agents should be involved?

Provide your response as a structured plan.
"""
            
            # Get plan from Bedrock
            plan_response = self.invoke_bedrock(planning_prompt, self.system_prompt)
            
            # Parse and structure the plan
            execution_plan = self._parse_plan_response(plan_response, user_query)
            
            # Store the plan in memory
            self.store_memory("last_execution_plan", execution_plan)
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {e}")
            raise
    
    def _parse_plan_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse the AI response into a structured execution plan"""
        try:
            # Try to extract structured information from the response
            plan = {
                "original_query": original_query,
                "timestamp": self._get_current_timestamp(),
                "tasks": [],
                "required_tables": [],
                "data_quality_checks": [],
                "sub_agents_needed": [],
                "execution_steps": [],
                "estimated_complexity": "medium",
                "raw_response": response
            }
            
            # Extract tasks from the response
            if "task" in response.lower() or "step" in response.lower():
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and any(keyword in line.lower() for keyword in ['task', 'step', 'operation']):
                        plan["tasks"].append(line)
            
            # Extract table mentions
            if "table" in response.lower():
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if "table" in line.lower() and any(word.isupper() for word in line.split()):
                        # Look for potential table names (usually uppercase in Oracle)
                        words = line.split()
                        for word in words:
                            if word.isupper() and len(word) > 2:
                                plan["required_tables"].append(word)
            
            # Determine sub-agents needed based on query type
            if any(keyword in original_query.lower() for keyword in ['select', 'query', 'find', 'get']):
                plan["sub_agents_needed"].append("QueryAgent")
            if any(keyword in original_query.lower() for keyword in ['insert', 'update', 'delete', 'modify']):
                plan["sub_agents_needed"].append("DMLAgent")
            if any(keyword in original_query.lower() for keyword in ['analyze', 'check', 'validate', 'quality']):
                plan["sub_agents_needed"].append("DataQualityAgent")
            
            # Estimate complexity
            if len(plan["tasks"]) > 5 or len(plan["required_tables"]) > 3:
                plan["estimated_complexity"] = "high"
            elif len(plan["tasks"]) > 2 or len(plan["required_tables"]) > 1:
                plan["estimated_complexity"] = "medium"
            else:
                plan["estimated_complexity"] = "low"
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to parse plan response: {e}")
            # Return a basic plan if parsing fails
            return {
                "original_query": original_query,
                "timestamp": self._get_current_timestamp(),
                "tasks": ["Execute query based on AI-generated plan"],
                "required_tables": [],
                "data_quality_checks": [],
                "sub_agents_needed": ["QueryAgent"],
                "execution_steps": ["1. Analyze query", "2. Execute with appropriate agent"],
                "estimated_complexity": "medium",
                "raw_response": response
            }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and refine the execution plan"""
        try:
            validation_prompt = f"""
Please validate this execution plan:

{json.dumps(plan, indent=2)}

Identify any issues, missing steps, or improvements needed. Consider:
1. Are all necessary tables included?
2. Are the data quality checks comprehensive?
3. Are the right sub-agents selected?
4. Is the complexity estimation accurate?
5. Are there any potential risks or edge cases?

Provide your validation feedback and any suggested improvements.
"""
            
            validation_response = self.invoke_bedrock(validation_prompt, self.system_prompt)
            
            # Update plan with validation feedback
            plan["validation_feedback"] = validation_response
            plan["is_validated"] = True
            plan["validation_timestamp"] = self._get_current_timestamp()
            
            # Store validated plan
            self.store_memory("validated_plan", plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to validate plan: {e}")
            plan["validation_feedback"] = "Validation failed due to error"
            plan["is_validated"] = False
            return plan
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return execution plan"""
        user_query = input_data.get("query", "")
        context = input_data.get("context", {})
        
        # Create initial plan
        plan = self.create_execution_plan(user_query, context)
        
        # Validate the plan
        validated_plan = self.validate_plan(plan)
        
        return validated_plan
    
    def get_plan_history(self) -> List[Dict[str, Any]]:
        """Get history of recent execution plans"""
        memory_summary = self.get_memory_summary()
        plans = []
        
        for key, data in memory_summary.items():
            if "execution_plan" in key or "validated_plan" in key:
                plans.append({
                    "key": key,
                    "data": data.get("value"),
                    "timestamp": data.get("timestamp")
                })
        
        return sorted(plans, key=lambda x: x["timestamp"], reverse=True) 