import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import tiktoken
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class SchemaChunk:
    """Represents a chunk of schema information"""
    table_name: str
    columns: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    description: str
    estimated_tokens: int

class SchemaChunkingManager:
    """Manages schema chunking and intelligent selection using Amazon Bedrock"""
    
    def __init__(self, max_tokens_per_query: int = 5000):
        self.max_tokens_per_query = max_tokens_per_query
        self.schema_cache = {}
        self.usage_patterns = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude tokenizer
        self.bedrock_client = self._initialize_bedrock()
        
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
                    "max_tokens_to_sample": 1000,
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
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from user query using Bedrock"""
        try:
            system_prompt = """You are an expert at analyzing database queries and extracting relevant entities. 
            Your task is to identify potential table names, column names, and business entities from the query.
            Return only a JSON array of strings with the extracted entities."""
            
            prompt = f"""
            Analyze this database query and extract all potential entities:
            Query: "{query}"
            
            Look for:
            1. Table names (usually uppercase words)
            2. Column names (usually lowercase words)
            3. Business entities (like customer, product, transaction, etc.)
            4. Domain-specific terms (like URA, AMP, BP for Medicaid)
            
            Return a JSON array of strings, for example:
            ["TABLE_NAME", "column_name", "business_entity"]
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            # Try to parse JSON response
            try:
                entities = json.loads(response)
                if isinstance(entities, list):
                    return entities
            except json.JSONDecodeError:
                # Fallback to regex extraction
                logger.warning("Failed to parse Bedrock response, using fallback")
                return self._extract_entities_fallback(query)
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to extract entities with Bedrock: {e}")
            return self._extract_entities_fallback(query)
    
    def _extract_entities_fallback(self, query: str) -> List[str]:
        """Fallback entity extraction using regex patterns"""
        query_lower = query.lower()
        entities = []
        
        # Extract table-like words (usually uppercase in queries)
        table_pattern = r'\b[A-Z][A-Z0-9_]*\b'
        potential_tables = re.findall(table_pattern, query)
        entities.extend(potential_tables)
        
        # Extract quoted strings (potential table/column names)
        quoted_pattern = r'"([^"]*)"'
        quoted_entities = re.findall(quoted_pattern, query)
        entities.extend(quoted_entities)
        
        # Extract words that might be table names
        words = query.split()
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 3 and word_clean.isupper():
                entities.append(word_clean)
        
        return list(set(entities))  # Remove duplicates
    
    def find_relevant_tables(self, entities: List[str], available_tables: List[str]) -> List[str]:
        """Find tables relevant to the extracted entities using Bedrock"""
        try:
            if not entities or not available_tables:
                return []
            
            system_prompt = """You are an expert at analyzing database schemas and finding relevant tables. 
            Given a list of entities and available tables, select the most relevant ones.
            Return only a JSON array of table names."""
            
            prompt = f"""
            Given these entities: {entities}
            And these available tables: {available_tables}
            
            Select the most relevant tables (maximum 8) that would be needed to answer queries about these entities.
            Consider:
            1. Direct name matches
            2. Semantic relationships
            3. Business domain relevance
            4. Data dependencies
            
            Return a JSON array of table names, for example:
            ["TABLE1", "TABLE2", "TABLE3"]
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                selected_tables = json.loads(response)
                if isinstance(selected_tables, list):
                    # Filter to only include available tables
                    filtered_tables = [table for table in selected_tables if table in available_tables]
                    return filtered_tables[:8]  # Limit to 8 tables
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for table selection")
        
        except Exception as e:
            logger.error(f"Failed to find relevant tables with Bedrock: {e}")
        
        # Fallback to simple matching
        return self._find_relevant_tables_fallback(entities, available_tables)
    
    def _find_relevant_tables_fallback(self, entities: List[str], available_tables: List[str]) -> List[str]:
        """Fallback table selection when Bedrock fails"""
        relevant_tables = []
        
        for entity in entities:
            # Direct table name match
            if entity in available_tables:
                relevant_tables.append(entity)
                continue
            
            # Partial table name match
            for table in available_tables:
                if entity.lower() in table.lower() or table.lower() in entity.lower():
                    relevant_tables.append(table)
                    continue
                
                # Check if entity is part of table name
                table_parts = table.lower().split('_')
                if any(entity.lower() in part for part in table_parts):
                    relevant_tables.append(table)
        
        # Limit to reasonable number of tables
        return list(set(relevant_tables))[:8]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def get_relevant_schema(self, user_query: str, available_tables: List[str], 
                           full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get only schema parts relevant to the query using Bedrock"""
        try:
            # Extract entities from query using Bedrock
            entities = self.extract_entities_from_query(user_query)
            logger.info(f"Extracted entities: {entities}")
            
            # Find relevant tables using Bedrock
            relevant_tables = self.find_relevant_tables(entities, available_tables)
            logger.info(f"Relevant tables: {relevant_tables}")
            
            # Get schema parts with token management
            schema_parts = {}
            total_tokens = 0
            
            # Sort tables by relevance (you could implement a scoring system here)
            for table in relevant_tables:
                if table not in full_schema:
                    continue
                
                if total_tokens >= self.max_tokens_per_query:
                    logger.info(f"Token limit reached ({total_tokens}/{self.max_tokens_per_query})")
                    break
                
                table_schema = full_schema[table]
                table_tokens = self.estimate_tokens(json.dumps(table_schema))
                
                if total_tokens + table_tokens <= self.max_tokens_per_query:
                    schema_parts[table] = table_schema
                    total_tokens += table_tokens
                    logger.info(f"Added table {table}: {table_tokens} tokens, total: {total_tokens}")
                else:
                    logger.info(f"Skipped table {table}: would exceed token limit")
            
            # Add metadata
            result = {
                "schema_parts": schema_parts,
                "metadata": {
                    "total_tokens": total_tokens,
                    "max_tokens": self.max_tokens_per_query,
                    "tables_included": list(schema_parts.keys()),
                    "tables_analyzed": len(relevant_tables),
                    "entities_extracted": entities,
                    "token_efficiency": f"{(total_tokens/self.max_tokens_per_query)*100:.1f}%"
                }
            }
            
            # Cache the result
            cache_key = self._generate_cache_key(user_query, relevant_tables)
            self.schema_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get relevant schema: {e}")
            # Fallback: return minimal schema
            return {
                "schema_parts": {},
                "metadata": {
                    "error": str(e),
                    "fallback": True
                }
            }
    
    def _generate_cache_key(self, query: str, tables: List[str]) -> str:
        """Generate cache key for schema selection"""
        # Create a hash of query and table selection
        key_data = {
            "query": query.lower().strip(),
            "tables": sorted(tables)
        }
        return json.dumps(key_data, sort_keys=True)
    
    def get_schema_chunk(self, table_name: str, chunk_type: str = "full") -> SchemaChunk:
        """Get a specific chunk of schema information"""
        if chunk_type == "basic":
            return SchemaChunk(
                table_name=table_name,
                columns=[],  # Minimal column info
                relationships=[],
                constraints=[],
                description="Basic table information",
                estimated_tokens=100
            )
        elif chunk_type == "detailed":
            return SchemaChunk(
                table_name=table_name,
                columns=[],  # Full column info
                relationships=[],  # Basic relationships
                constraints=[],
                description="Detailed table information",
                estimated_tokens=500
            )
        else:  # full
            return SchemaChunk(
                table_name=table_name,
                columns=[],  # Full column info
                relationships=[],  # All relationships
                constraints=[],  # All constraints
                description="Complete table information",
                estimated_tokens=1000
            )
    
    def optimize_schema_for_query(self, schema: Dict[str, Any], query_complexity: str) -> Dict[str, Any]:
        """Optimize schema based on query complexity"""
        if query_complexity == "simple":
            # Remove detailed constraints and relationships
            optimized = {}
            for table, table_schema in schema.items():
                optimized[table] = {
                    "table_name": table_schema.get("table_name", table),
                    "columns": table_schema.get("columns", []),
                    "description": table_schema.get("description", "")
                }
            return optimized
        
        elif query_complexity == "moderate":
            # Keep columns and basic relationships
            optimized = {}
            for table, table_schema in schema.items():
                optimized[table] = {
                    "table_name": table_schema.get("table_name", table),
                    "columns": table_schema.get("columns", []),
                    "relationships": table_schema.get("relationships", []),
                    "description": table_schema.get("description", "")
                }
            return optimized
        
        else:  # complex
            # Return full schema
            return schema
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.schema_cache),
            "cache_keys": list(self.schema_cache.keys()),
            "usage_patterns": self.usage_patterns
        }
    
    def clear_cache(self):
        """Clear the schema cache"""
        self.schema_cache.clear()
        logger.info("Schema cache cleared")
    
    def update_usage_patterns(self, query: str, selected_tables: List[str]):
        """Update usage patterns for optimization"""
        query_key = query.lower().strip()[:100]  # Truncate long queries
        
        if query_key not in self.usage_patterns:
            self.usage_patterns[query_key] = {
                "tables": selected_tables,
                "count": 1,
                "last_used": None
            }
        else:
            self.usage_patterns[query_key]["count"] += 1
            self.usage_patterns[query_key]["last_used"] = "now"  # You could use actual timestamps 