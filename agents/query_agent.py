import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent
from database.connection import db_connection

logger = logging.getLogger(__name__)

class QueryAgent(BaseAgent):
    """Agent responsible for intelligent query generation and execution"""
    
    def __init__(self):
        super().__init__("Query")
        self.system_prompt = self._get_query_system_prompt()
        self.schema_cache = {}
        self.query_cache = {}
    
    def _get_query_system_prompt(self) -> str:
        """Get the system prompt for the query agent"""
        return """You are a Database Query Generation Agent. Your role is to:

1. Analyze user queries and understand their intent
2. Generate appropriate SQL queries for Oracle database
3. Select the most relevant tables and schemas
4. Ensure query efficiency and correctness
5. Handle complex queries with proper JOINs and aggregations

You should:
- Generate clean, efficient Oracle SQL
- Use proper Oracle syntax and functions
- Consider query performance and optimization
- Handle edge cases and error conditions
- Provide clear explanations for your query choices

Always respond with:
- The generated SQL query
- Explanation of table/schema selection
- Any assumptions made
- Performance considerations"""

    def analyze_query_intent(self, user_query: str) -> Dict[str, Any]:
        """Analyze the user query to understand intent and required data"""
        try:
            analysis_prompt = f"""
Analyze this database query request:

"{user_query}"

Please identify:
1. What type of data is being requested?
2. What entities or objects are mentioned?
3. What time periods or conditions are relevant?
4. What aggregations or calculations might be needed?
5. What tables or schemas are likely involved?

Provide a structured analysis of the query intent.
"""
            
            analysis_response = self.invoke_bedrock(analysis_prompt, self.system_prompt)
            
            # Store analysis in memory
            self.store_memory(f"query_analysis_{hash(user_query)}", analysis_response)
            
            return {
                "original_query": user_query,
                "analysis": analysis_response,
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze query intent: {e}")
            raise
    
    def select_relevant_tables(self, query_intent: Dict[str, Any]) -> List[str]:
        """Select the most relevant tables based on query intent"""
        try:
            # Get all available tables
            all_tables = db_connection.get_all_tables()
            
            if all_tables.empty:
                logger.warning("No tables found in database")
                return []
            
            # Create table selection prompt
            selection_prompt = f"""
Based on this query intent:

{json.dumps(query_intent, indent=2)}

And these available tables:

{all_tables.to_string()}

Please select the most relevant tables for this query. Consider:
1. Which tables contain the primary data needed?
2. What tables might need to be joined?
3. Are there any lookup or reference tables involved?
4. What is the optimal table selection for performance?

List only the table names that are most relevant, separated by commas.
"""
            
            selection_response = self.invoke_bedrock(selection_prompt, self.system_prompt)
            
            # Parse table names from response
            selected_tables = self._parse_table_selection(selection_response, all_tables)
            
            # Store selection in memory
            self.store_memory(f"table_selection_{hash(str(query_intent))}", selected_tables)
            
            return selected_tables
            
        except Exception as e:
            logger.error(f"Failed to select relevant tables: {e}")
            # Fallback to basic table selection
            return self._fallback_table_selection(query_intent)
    
    def _parse_table_selection(self, response: str, available_tables: pd.DataFrame) -> List[str]:
        """Parse table names from AI response"""
        try:
            # Extract table names from response
            response_lower = response.lower()
            selected_tables = []
            
            for _, row in available_tables.iterrows():
                table_name = row['TABLE_NAME']
                if table_name.lower() in response_lower:
                    selected_tables.append(table_name)
            
            # If no tables found, try to extract from response text
            if not selected_tables:
                words = response.split()
                for word in words:
                    word = word.strip('.,;:!?()[]{}"\'').upper()
                    if word in available_tables['TABLE_NAME'].values:
                        selected_tables.append(word)
            
            return list(set(selected_tables))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to parse table selection: {e}")
            return []
    
    def _fallback_table_selection(self, query_intent: Dict[str, Any]) -> List[str]:
        """Fallback table selection when AI selection fails"""
        try:
            # Get all tables
            all_tables = db_connection.get_all_tables()
            
            if all_tables.empty:
                return []
            
            # Simple keyword-based selection
            query_text = query_intent.get("original_query", "").lower()
            selected_tables = []
            
            for _, row in all_tables.iterrows():
                table_name = row['TABLE_NAME'].lower()
                
                # Check if table name contains keywords from query
                if any(keyword in table_name for keyword in ['user', 'customer', 'order', 'product', 'transaction']):
                    if any(keyword in query_text for keyword in ['user', 'customer', 'order', 'product', 'transaction']):
                        selected_tables.append(row['TABLE_NAME'])
            
            return selected_tables[:3]  # Limit to 3 tables
            
        except Exception as e:
            logger.error(f"Fallback table selection failed: {e}")
            return []
    
    def generate_sql_query(self, user_query: str, selected_tables: List[str]) -> str:
        """Generate SQL query based on user query and selected tables"""
        try:
            # Get schema information for selected tables
            table_schemas = {}
            for table in selected_tables:
                schema = db_connection.get_table_schema(table)
                table_schemas[table] = schema
            
            # Create SQL generation prompt
            sql_prompt = f"""
Generate an Oracle SQL query for this request:

"{user_query}"

Using these tables and their schemas:

{json.dumps(table_schemas, indent=2, default=str)}

Requirements:
1. Use proper Oracle SQL syntax
2. Include appropriate JOINs if multiple tables
3. Add WHERE clauses for filtering
4. Use proper column aliases
5. Consider performance (add hints if needed)
6. Handle NULL values appropriately

Return only the SQL query, no explanations.
"""
            
            sql_response = self.invoke_bedrock(sql_prompt, self.system_prompt)
            
            # Clean up the SQL response
            sql_query = self._clean_sql_response(sql_response)
            
            # Store generated SQL in memory
            self.store_memory(f"generated_sql_{hash(user_query)}", sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Failed to generate SQL query: {e}")
            raise
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean and format the SQL response from AI"""
        try:
            # Remove markdown formatting
            if response.startswith("```sql"):
                response = response[6:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Remove extra whitespace and newlines
            response = response.strip()
            
            # Ensure proper SQL formatting
            lines = response.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('--'):  # Remove comment lines
                    cleaned_lines.append(line)
            
            return ' '.join(cleaned_lines)
            
        except Exception as e:
            logger.error(f"Failed to clean SQL response: {e}")
            return response.strip()
    
    def execute_query(self, sql_query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the generated SQL query"""
        try:
            # Store query execution attempt
            self.store_memory(f"query_execution_{hash(sql_query)}", {
                "sql": sql_query,
                "params": params,
                "timestamp": self._get_current_timestamp()
            })
            
            # Execute the query
            results = db_connection.execute_query(sql_query, params)
            
            # Process results
            execution_result = {
                "success": True,
                "sql_query": sql_query,
                "parameters": params,
                "row_count": len(results),
                "columns": list(results.columns) if not results.empty else [],
                "results": results.to_dict('records') if not results.empty else [],
                "execution_timestamp": self._get_current_timestamp()
            }
            
            # Store successful execution
            self.store_memory(f"query_results_{hash(sql_query)}", execution_result)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            
            error_result = {
                "success": False,
                "sql_query": sql_query,
                "parameters": params,
                "error": str(e),
                "execution_timestamp": self._get_current_timestamp()
            }
            
            # Store error information
            self.store_memory(f"query_error_{hash(sql_query)}", error_result)
            
            return error_result
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return query results"""
        user_query = input_data.get("query", "")
        context = input_data.get("context", {})
        
        try:
            # Analyze query intent
            query_intent = self.analyze_query_intent(user_query)
            
            # Select relevant tables
            selected_tables = self.select_relevant_tables(query_intent)
            
            # Generate SQL query
            sql_query = self.generate_sql_query(user_query, selected_tables)
            
            # Execute query
            execution_result = self.execute_query(sql_query, context.get("parameters"))
            
            # Combine all results
            result = {
                "query_intent": query_intent,
                "selected_tables": selected_tables,
                "generated_sql": sql_query,
                "execution_result": execution_result,
                "timestamp": self._get_current_timestamp()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "error": str(e),
                "user_query": user_query,
                "timestamp": self._get_current_timestamp()
            }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get history of recent queries"""
        memory_summary = self.get_memory_summary()
        queries = []
        
        for key, data in memory_summary.items():
            if any(prefix in key for prefix in ["query_analysis_", "generated_sql_", "query_results_"]):
                queries.append({
                    "key": key,
                    "data": data.get("value"),
                    "timestamp": data.get("timestamp")
                })
        
        return sorted(queries, key=lambda x: x["timestamp"], reverse=True) 