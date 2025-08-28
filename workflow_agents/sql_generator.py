import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class SQLQuery:
    """Generated SQL query with metadata"""
    sql: str
    query_type: str
    tables_used: List[str]
    columns_selected: List[str]
    where_conditions: List[str]
    joins: List[str]
    group_by: List[str]
    order_by: List[str]
    estimated_rows: int
    complexity: str

@dataclass
class QueryPlan:
    """Query execution plan"""
    sql_queries: List[SQLQuery]
    execution_order: List[int]
    dependencies: Dict[int, List[int]]
    estimated_cost: float
    optimization_suggestions: List[str]

class SQLGenerator:
    """Generates SQL queries using Bedrock and context"""
    
    def __init__(self):
        self.bedrock_client = self._initialize_bedrock()
        self.query_cache = {}
        
    def _initialize_bedrock(self):
        """Initialize Amazon Bedrock client"""
        try:
            import boto3
            bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                aws_session_token=settings.AWS_SESSION_TOKEN
            )
            logger.info("Bedrock client initialized successfully")
            return bedrock_client
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            return None
    
    def _invoke_bedrock(self, prompt: str, system_prompt: str = "") -> str:
        """Invoke Amazon Bedrock model"""
        try:
            if not self.bedrock_client:
                raise Exception("Bedrock client not initialized")
            
            # Prepare the message
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Invoke Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=settings.BEDROCK_MODEL_ID,
                body=json.dumps({
                    "prompt": f"\n\nHuman: {full_prompt}\n\nAssistant:",
                    "max_tokens_to_sample": 2000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "stop_sequences": ["\n\nHuman:"]
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['completion']
            
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            return ""
    
    def generate_sql(self, user_query: str, available_schema: Dict[str, Any], 
                     use_case_context: str, comparison_strategy: str = "auto") -> SQLQuery:
        """Generate SQL query using Bedrock and context"""
        try:
            logger.info(f"Generating SQL for query: {user_query[:100]}...")
            
            # Check cache first
            cache_key = self._generate_cache_key(user_query, available_schema)
            if cache_key in self.query_cache:
                logger.info("Returning cached SQL query")
                return self.query_cache[cache_key]
            
            # Generate SQL using Bedrock
            sql_query = self._generate_sql_with_bedrock(
                user_query, available_schema, use_case_context, comparison_strategy
            )
            
            # Cache the result
            self.query_cache[cache_key] = sql_query
            
            logger.info(f"SQL generated successfully: {sql_query.query_type}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}")
            return self._generate_fallback_sql(user_query, available_schema)
    
    def _generate_sql_with_bedrock(self, user_query: str, available_schema: Dict[str, Any], 
                                  use_case_context: str, comparison_strategy: str) -> SQLQuery:
        """Generate SQL using Bedrock"""
        try:
            system_prompt = """You are an expert SQL developer and database analyst. 
            Generate Oracle SQL queries based on user requirements and available schema.
            
            Return a JSON object with these fields:
            - sql: The complete SQL query
            - query_type: SELECT, INSERT, UPDATE, DELETE, or ANALYTICAL
            - tables_used: List of table names used
            - columns_selected: List of columns in SELECT clause
            - where_conditions: List of WHERE conditions
            - joins: List of JOIN clauses
            - group_by: List of GROUP BY columns
            - order_by: List of ORDER BY columns
            - estimated_rows: Estimated number of rows returned
            - complexity: SIMPLE, MODERATE, or COMPLEX
            
            Use Oracle SQL syntax and best practices."""
            
            # Create schema summary for the prompt
            schema_summary = self._create_schema_summary_for_sql(available_schema)
            
            prompt = f"""
            Generate an Oracle SQL query for this request:
            
            User Query: "{user_query}"
            
            Use Case Context:
            {use_case_context}
            
            Available Schema:
            {json.dumps(schema_summary, indent=2)}
            
            Comparison Strategy: {comparison_strategy}
            
            Consider:
            1. Business logic from the use case context
            2. Table relationships and dependencies
            3. Appropriate comparison strategies
            4. Performance optimization
            5. Data quality considerations
            
            Generate a complete, executable Oracle SQL query.
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                sql_data = json.loads(response)
                return SQLQuery(
                    sql=sql_data.get("sql", ""),
                    query_type=sql_data.get("query_type", "SELECT"),
                    tables_used=sql_data.get("tables_used", []),
                    columns_selected=sql_data.get("columns_selected", []),
                    where_conditions=sql_data.get("where_conditions", []),
                    joins=sql_data.get("joins", []),
                    group_by=sql_data.get("group_by", []),
                    order_by=sql_data.get("order_by", []),
                    estimated_rows=sql_data.get("estimated_rows", 0),
                    complexity=sql_data.get("complexity", "MODERATE")
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for SQL generation")
                return self._generate_sql_fallback(user_query, available_schema)
                
        except Exception as e:
            logger.error(f"Failed to generate SQL with Bedrock: {e}")
            return self._generate_sql_fallback(user_query, available_schema)
    
    def _create_schema_summary_for_sql(self, available_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create schema summary optimized for SQL generation"""
        summary = {}
        
        for table_name, table_data in available_schema.items():
            summary[table_name] = {
                "description": table_data.get("description", ""),
                "columns": [
                    {
                        "name": col.get("name", ""),
                        "data_type": col.get("data_type", ""),
                        "nullable": col.get("nullable", "Y"),
                        "description": col.get("description", "")
                    }
                    for col in table_data.get("columns", [])
                ],
                "primary_keys": table_data.get("primary_keys", []),
                "foreign_keys": table_data.get("foreign_keys", []),
                "estimated_rows": table_data.get("estimated_rows", "Unknown")
            }
        
        return summary
    
    def _generate_sql_fallback(self, user_query: str, available_schema: Dict[str, Any]) -> SQLQuery:
        """Fallback SQL generation when Bedrock fails"""
        # Simple fallback SQL generation
        query_lower = user_query.lower()
        
        if "count" in query_lower or "how many" in query_lower:
            # Generate COUNT query
            tables = list(available_schema.keys())[:1]  # Use first table
            sql = f"SELECT COUNT(*) FROM {tables[0]}"
            query_type = "SELECT"
        elif "analyze" in query_lower or "compare" in query_lower:
            # Generate analytical query
            tables = list(available_schema.keys())[:2]  # Use first two tables
            if len(tables) >= 2:
                sql = f"SELECT * FROM {tables[0]} t1 JOIN {tables[1]} t2 ON t1.id = t2.id"
            else:
                sql = f"SELECT * FROM {tables[0]}"
            query_type = "ANALYTICAL"
        else:
            # Default SELECT query
            tables = list(available_schema.keys())[:1]
            sql = f"SELECT * FROM {tables[0]}"
            query_type = "SELECT"
        
        return SQLQuery(
            sql=sql,
            query_type=query_type,
            tables_used=tables,
            columns_selected=["*"],
            where_conditions=[],
            joins=[],
            group_by=[],
            order_by=[],
            estimated_rows=1000,
            complexity="SIMPLE"
        )
    
    def generate_comparison_sql(self, base_query: SQLQuery, comparison_type: str, 
                               comparison_params: Dict[str, Any]) -> SQLQuery:
        """Generate SQL for data comparison"""
        try:
            system_prompt = """You are an expert SQL developer. Generate comparison SQL queries.
            Return a JSON object with the SQL query and metadata."""
            
            prompt = f"""
            Generate a comparison SQL query based on:
            
            Base Query: {base_query.sql}
            Comparison Type: {comparison_type}
            Comparison Parameters: {json.dumps(comparison_params, indent=2)}
            
            The comparison should:
            1. Use the same base structure as the original query
            2. Apply appropriate comparison logic
            3. Return comparable results
            4. Be optimized for performance
            
            Return the SQL query in JSON format.
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                comparison_data = json.loads(response)
                return SQLQuery(
                    sql=comparison_data.get("sql", ""),
                    query_type="COMPARISON",
                    tables_used=base_query.tables_used,
                    columns_selected=base_query.columns_selected,
                    where_conditions=comparison_data.get("where_conditions", []),
                    joins=base_query.joins,
                    group_by=base_query.group_by,
                    order_by=base_query.order_by,
                    estimated_rows=base_query.estimated_rows,
                    complexity="MODERATE"
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for comparison SQL")
                return self._generate_comparison_sql_fallback(base_query, comparison_type, comparison_params)
                
        except Exception as e:
            logger.error(f"Failed to generate comparison SQL: {e}")
            return self._generate_comparison_sql_fallback(base_query, comparison_type, comparison_params)
    
    def _generate_comparison_sql_fallback(self, base_query: SQLQuery, comparison_type: str, 
                                        comparison_params: Dict[str, Any]) -> SQLQuery:
        """Fallback comparison SQL generation"""
        # Simple comparison logic
        if comparison_type == "time_based":
            # Add time-based comparison
            sql = base_query.sql.replace("SELECT", "SELECT /* Time Comparison */")
            sql += f" WHERE {comparison_params.get('time_column', 'created_date')} >= SYSDATE - {comparison_params.get('days', 30)}"
        elif comparison_type == "category_based":
            # Add category-based comparison
            sql = base_query.sql.replace("SELECT", "SELECT /* Category Comparison */")
            sql += f" WHERE {comparison_params.get('category_column', 'category')} = '{comparison_params.get('category_value', 'default')}'"
        else:
            # Default comparison
            sql = base_query.sql.replace("SELECT", "SELECT /* Comparison */")
        
        return SQLQuery(
            sql=sql,
            query_type="COMPARISON",
            tables_used=base_query.tables_used,
            columns_selected=base_query.columns_selected,
            where_conditions=base_query.where_conditions,
            joins=base_query.joins,
            group_by=base_query.group_by,
            order_by=base_query.order_by,
            estimated_rows=base_query.estimated_rows,
            complexity="MODERATE"
        )
    
    def optimize_query(self, sql_query: SQLQuery, performance_metrics: Dict[str, Any]) -> SQLQuery:
        """Optimize SQL query based on performance metrics"""
        try:
            system_prompt = """You are an expert SQL optimizer. Analyze and optimize SQL queries.
            Return a JSON object with the optimized SQL and optimization details."""
            
            prompt = f"""
            Optimize this SQL query for better performance:
            
            Original SQL: {sql_query.sql}
            Performance Metrics: {json.dumps(performance_metrics, indent=2)}
            
            Consider:
            1. Index usage
            2. Join optimization
            3. WHERE clause optimization
            4. Subquery optimization
            5. Oracle-specific optimizations
            
            Return the optimized SQL in JSON format.
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                optimized_data = json.loads(response)
                return SQLQuery(
                    sql=optimized_data.get("sql", sql_query.sql),
                    query_type=sql_query.query_type,
                    tables_used=sql_query.tables_used,
                    columns_selected=sql_query.columns_selected,
                    where_conditions=sql_query.where_conditions,
                    joins=sql_query.joins,
                    group_by=sql_query.group_by,
                    order_by=sql_query.order_by,
                    estimated_rows=sql_query.estimated_rows,
                    complexity=sql_query.complexity
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for query optimization")
                return sql_query
                
        except Exception as e:
            logger.error(f"Failed to optimize query: {e}")
            return sql_query
    
    def validate_sql(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query syntax and structure"""
        try:
            system_prompt = """You are an expert SQL validator. Validate SQL queries for:
            1. Syntax correctness
            2. Oracle compatibility
            3. Best practices
            4. Potential issues
            
            Return a JSON object with validation results."""
            
            prompt = f"""
            Validate this Oracle SQL query:
            
            SQL: {sql_query}
            
            Check for:
            1. Syntax errors
            2. Oracle compatibility
            3. Performance issues
            4. Security concerns
            5. Best practices
            
            Return validation results in JSON format.
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                validation_data = json.loads(response)
                return validation_data
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for SQL validation")
                return {"valid": False, "errors": ["Failed to validate with Bedrock"]}
                
        except Exception as e:
            logger.error(f"Failed to validate SQL: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    def _generate_cache_key(self, user_query: str, available_schema: Dict[str, Any]) -> str:
        """Generate cache key for SQL queries"""
        import hashlib
        
        # Create a hash of the inputs
        key_data = {
            "query": user_query[:200],  # Truncate long queries
            "schema_tables": sorted(available_schema.keys())
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.query_cache),
            "cache_keys": list(self.query_cache.keys())
        } 