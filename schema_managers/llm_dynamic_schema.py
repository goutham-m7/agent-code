import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
import anthropic
from config import settings

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "bedrock"

@dataclass
class DomainContext:
    """Domain context information"""
    domain: str
    keywords: List[str]
    core_concepts: List[str]
    business_entities: List[str]
    common_queries: List[str]
    schema_patterns: List[str]

class LLMDynamicSchemaManager:
    """LLM-powered schema manager that dynamically adapts to any domain"""
    
    def __init__(self, llm_provider: LLMProvider = LLMProvider.ANTHROPIC, 
                 max_tokens_per_query: int = 5000):
        self.llm_provider = llm_provider
        self.max_tokens_per_query = max_tokens_per_query
        self.domain_cache = {}
        self.query_patterns = {}
        self.llm_client = self._initialize_llm_client()
        self.domain_analyzer = DomainAnalyzer(self.llm_client)
        
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client"""
        try:
            if self.llm_provider == LLMProvider.OPENAI:
                openai.api_key = settings.OPENAI_API_KEY
                return openai
            elif self.llm_provider == LLMProvider.ANTHROPIC:
                return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            elif self.llm_provider == LLMProvider.AWS_BEDROCK:
                # Use existing Bedrock client from config
                return None  # Will use config settings
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None
    
    def analyze_domain_dynamically(self, user_query: str, available_tables: List[str], 
                                  sample_data: Dict[str, Any] = None) -> DomainContext:
        """Dynamically analyze the domain from the query and available data"""
        try:
            # Check cache first
            cache_key = self._generate_domain_cache_key(user_query, available_tables)
            if cache_key in self.domain_cache:
                logger.info("Returning cached domain analysis")
                return self.domain_cache[cache_key]
            
            # Analyze domain using LLM
            domain_analysis = self.domain_analyzer.analyze_domain(
                user_query, available_tables, sample_data
            )
            
            # Cache the result
            self.domain_cache[cache_key] = domain_analysis
            
            return domain_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze domain dynamically: {e}")
            # Return generic domain context
            return self._get_generic_domain_context()
    
    def get_adaptive_schema(self, user_query: str, available_tables: List[str], 
                           full_schema: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get schema that adapts to the domain and query dynamically"""
        try:
            start_time = time.time()
            
            # Analyze domain dynamically
            domain_context = self.analyze_domain_dynamically(user_query, available_tables)
            
            # Get query intent using LLM
            query_intent = self._analyze_query_intent_with_llm(user_query, domain_context)
            
            # Select relevant tables using LLM
            relevant_tables = self._select_tables_with_llm(
                user_query, available_tables, full_schema, domain_context, query_intent
            )
            
            # Generate optimized schema
            optimized_schema = self._generate_optimized_schema(
                relevant_tables, full_schema, query_intent, domain_context
            )
            
            execution_time = time.time() - start_time
            
            return {
                "schema": optimized_schema,
                "metadata": {
                    "domain": domain_context.domain,
                    "query_intent": query_intent,
                    "selected_tables": relevant_tables,
                    "total_tokens": self._estimate_schema_tokens(optimized_schema),
                    "max_tokens": self.max_tokens_per_query,
                    "execution_time": execution_time,
                    "domain_analysis": domain_context,
                    "llm_provider": self.llm_provider.value
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get adaptive schema: {e}")
            return self._get_fallback_schema(user_query, available_tables, full_schema)
    
    def _analyze_query_intent_with_llm(self, query: str, domain_context: DomainContext) -> Dict[str, Any]:
        """Analyze query intent using LLM"""
        try:
            prompt = f"""
            Analyze the following query in the context of {domain_context.domain} domain.
            
            Domain Context:
            - Keywords: {', '.join(domain_context.keywords)}
            - Core Concepts: {', '.join(domain_context.core_concepts)}
            - Business Entities: {', '.join(domain_context.business_entities)}
            
            Query: "{query}"
            
            Please analyze the query and provide:
            1. Intent (what the user wants to achieve)
            2. Complexity level (simple, moderate, complex)
            3. Required information types
            4. Suggested approach
            5. Key entities mentioned
            
            Respond in JSON format.
            """
            
            if self.llm_provider == LLMProvider.OPENAI:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                result = response.choices[0].message.content
            elif self.llm_provider == LLMProvider.ANTHROPIC:
                response = self.llm_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
            else:
                # Use Bedrock from config
                result = self._invoke_bedrock(prompt)
            
            # Parse JSON response
            try:
                intent_analysis = json.loads(result)
                return intent_analysis
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_intent_fallback(result)
                
        except Exception as e:
            logger.error(f"Failed to analyze query intent with LLM: {e}")
            return self._get_default_intent_analysis()
    
    def _select_tables_with_llm(self, query: str, available_tables: List[str], 
                               full_schema: Dict[str, Any], domain_context: DomainContext,
                               query_intent: Dict[str, Any]) -> List[str]:
        """Select relevant tables using LLM"""
        try:
            # Create table summaries for LLM
            table_summaries = []
            for table in available_tables[:20]:  # Limit to first 20 tables for LLM
                if table in full_schema:
                    table_data = full_schema[table]
                    summary = {
                        "table": table,
                        "description": table_data.get("description", ""),
                        "columns": len(table_data.get("columns", [])),
                        "estimated_rows": table_data.get("estimated_rows", "Unknown")
                    }
                    table_summaries.append(summary)
            
            prompt = f"""
            Given the following query and domain context, select the most relevant tables.
            
            Query: "{query}"
            Domain: {domain_context.domain}
            Intent: {query_intent.get('intent', 'Unknown')}
            Complexity: {query_intent.get('complexity', 'Unknown')}
            
            Available Tables:
            {json.dumps(table_summaries, indent=2)}
            
            Select the most relevant tables (maximum 8) and explain why each is relevant.
            Consider:
            1. Direct relevance to the query
            2. Domain-specific importance
            3. Data relationships
            4. Query complexity requirements
            
            Respond with a JSON list of table names and brief reasoning.
            """
            
            if self.llm_provider == LLMProvider.OPENAI:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.1
                )
                result = response.choices[0].message.content
            elif self.llm_provider == LLMProvider.ANTHROPIC:
                response = self.llm_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=800,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
            else:
                result = self._invoke_bedrock(prompt)
            
            # Parse table selection
            try:
                table_selection = json.loads(result)
                if isinstance(table_selection, list):
                    return table_selection
                elif isinstance(table_selection, dict) and "tables" in table_selection:
                    return table_selection["tables"]
                else:
                    return self._extract_tables_from_text(result, available_tables)
            except json.JSONDecodeError:
                return self._extract_tables_from_text(result, available_tables)
                
        except Exception as e:
            logger.error(f"Failed to select tables with LLM: {e}")
            return self._select_tables_fallback(query, available_tables, domain_context)
    
    def _generate_optimized_schema(self, selected_tables: List[str], full_schema: Dict[str, Any],
                                 query_intent: Dict[str, Any], domain_context: DomainContext) -> Dict[str, Any]:
        """Generate optimized schema based on LLM analysis"""
        try:
            optimized_schema = {}
            total_tokens = 0
            
            for table in selected_tables:
                if table in full_schema and total_tokens < self.max_tokens_per_query:
                    table_schema = full_schema[table]
                    
                    # Optimize based on query intent
                    if query_intent.get("complexity") == "simple":
                        # Minimal schema for simple queries
                        optimized_table = {
                            "table_name": table,
                            "description": table_schema.get("description", ""),
                            "key_columns": self._extract_key_columns(table_schema)
                        }
                    elif query_intent.get("complexity") == "moderate":
                        # Moderate schema
                        optimized_table = {
                            "table_name": table,
                            "description": table_schema.get("description", ""),
                            "columns": self._extract_column_summary(table_schema),
                            "relationships": self._extract_basic_relationships(table_schema)
                        }
                    else:
                        # Full schema for complex queries
                        optimized_table = table_schema
                    
                    table_tokens = self._estimate_schema_tokens(optimized_table)
                    
                    if total_tokens + table_tokens <= self.max_tokens_per_query:
                        optimized_schema[table] = optimized_table
                        total_tokens += table_tokens
                    else:
                        logger.info(f"Stopping at table {table} due to token limit")
                        break
            
            return optimized_schema
            
        except Exception as e:
            logger.error(f"Failed to generate optimized schema: {e}")
            return {}
    
    def _extract_key_columns(self, table_schema: Dict[str, Any]) -> List[str]:
        """Extract key columns from table schema"""
        key_columns = []
        
        # Primary keys
        if "primary_keys" in table_schema:
            key_columns.extend(table_schema["primary_keys"])
        
        # Foreign keys
        if "foreign_keys" in table_schema:
            for fk in table_schema["foreign_keys"]:
                if "columns" in fk:
                    key_columns.extend(fk["columns"])
        
        return list(set(key_columns))
    
    def _extract_column_summary(self, table_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract summary of columns"""
        columns = table_schema.get("columns", [])
        return [
            {
                "name": col.get("name", ""),
                "data_type": col.get("data_type", ""),
                "nullable": col.get("nullable", "Y"),
                "description": col.get("description", "")
            }
            for col in columns[:10]  # Limit to 10 columns
        ]
    
    def _extract_basic_relationships(self, table_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract basic relationship information"""
        relationships = []
        
        if "foreign_keys" in table_schema:
            for fk in table_schema["foreign_keys"][:5]:  # Limit to 5 relationships
                relationships.append({
                    "type": "foreign_key",
                    "target_table": fk.get("referenced_table", ""),
                    "columns": fk.get("columns", [])
                })
        
        return relationships
    
    def _extract_tables_from_text(self, text: str, available_tables: List[str]) -> List[str]:
        """Extract table names from LLM response text"""
        selected_tables = []
        text_lower = text.lower()
        
        for table in available_tables:
            if table.lower() in text_lower:
                selected_tables.append(table)
        
        # If no tables found, return first few
        if not selected_tables:
            selected_tables = available_tables[:5]
        
        return selected_tables[:8]  # Limit to 8 tables
    
    def _select_tables_fallback(self, query: str, available_tables: List[str], 
                               domain_context: DomainContext) -> List[str]:
        """Fallback table selection"""
        # Simple keyword-based selection
        query_lower = query.lower()
        selected_tables = []
        
        for table in available_tables:
            if any(keyword in table.lower() for keyword in domain_context.keywords):
                selected_tables.append(table)
        
        # If no domain-specific tables, return first few
        if not selected_tables:
            selected_tables = available_tables[:5]
        
        return selected_tables[:8]
    
    def _invoke_bedrock(self, prompt: str) -> str:
        """Invoke AWS Bedrock (using existing config)"""
        try:
            # This would use your existing Bedrock setup
            # For now, return a simple response
            return '{"tables": ["TABLE_1", "TABLE_2"], "reasoning": "Basic selection"}'
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            return '{"tables": [], "reasoning": "Fallback"}'
    
    def _parse_intent_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback parsing for intent analysis"""
        text_lower = text.lower()
        
        intent = "unknown"
        complexity = "moderate"
        
        if any(word in text_lower for word in ["count", "how many", "total"]):
            intent = "count"
            complexity = "simple"
        elif any(word in text_lower for word in ["analyze", "analysis", "examine"]):
            intent = "analysis"
            complexity = "complex"
        elif any(word in text_lower for word in ["compare", "difference", "versus"]):
            intent = "comparison"
            complexity = "moderate"
        
        return {
            "intent": intent,
            "complexity": complexity,
            "entities": [],
            "approach": "standard"
        }
    
    def _get_default_intent_analysis(self) -> Dict[str, Any]:
        """Default intent analysis when LLM fails"""
        return {
            "intent": "unknown",
            "complexity": "moderate",
            "entities": [],
            "approach": "standard"
        }
    
    def _get_generic_domain_context(self) -> DomainContext:
        """Generic domain context when analysis fails"""
        return DomainContext(
            domain="general",
            keywords=["data", "table", "column", "query"],
            core_concepts=["information", "analysis", "retrieval"],
            business_entities=["entities", "records", "objects"],
            common_queries=["find", "show", "get", "analyze"],
            schema_patterns=["standard", "normalized", "relational"]
        )
    
    def _get_fallback_schema(self, query: str, available_tables: List[str], 
                            full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback schema when LLM analysis fails"""
        fallback_tables = available_tables[:3]
        fallback_schema = {}
        
        for table in fallback_tables:
            if table in full_schema:
                fallback_schema[table] = {
                    "table_name": table,
                    "description": full_schema[table].get("description", ""),
                    "columns": full_schema[table].get("columns", [])[:5]
                }
        
        return {
            "schema": fallback_schema,
            "metadata": {
                "domain": "fallback",
                "query_intent": {"intent": "fallback", "complexity": "simple"},
                "selected_tables": fallback_tables,
                "total_tokens": self._estimate_schema_tokens(fallback_schema),
                "max_tokens": self.max_tokens_per_query,
                "execution_time": 0.0,
                "fallback": True
            }
        }
    
    def _estimate_schema_tokens(self, schema: Dict[str, Any]) -> int:
        """Estimate token count for schema"""
        try:
            schema_text = json.dumps(schema, indent=2)
            # Rough estimation: 1 token ≈ 4 characters
            return len(schema_text) // 4
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}")
            return 1000
    
    def _generate_domain_cache_key(self, query: str, tables: List[str]) -> str:
        """Generate cache key for domain analysis"""
        key_data = {
            "query": query.lower().strip()[:100],
            "tables": sorted(tables)
        }
        return json.dumps(key_data, sort_keys=True)
    
    def clear_cache(self):
        """Clear the domain cache"""
        self.domain_cache.clear()
        logger.info("Domain cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "domain_cache_size": len(self.domain_cache),
            "cache_keys": list(self.domain_cache.keys()),
            "llm_provider": self.llm_provider.value
        }

class DomainAnalyzer:
    """Analyzes domains using LLM"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def analyze_domain(self, query: str, available_tables: List[str], 
                      sample_data: Dict[str, Any] = None) -> DomainContext:
        """Analyze the domain from query and available data"""
        try:
            # Create analysis prompt
            prompt = self._create_domain_analysis_prompt(query, available_tables, sample_data)
            
            # Get LLM response
            response = self._get_llm_response(prompt)
            
            # Parse domain context
            return self._parse_domain_context(response)
            
        except Exception as e:
            logger.error(f"Failed to analyze domain: {e}")
            return self._get_generic_domain_context()
    
    def _create_domain_analysis_prompt(self, query: str, available_tables: List[str], 
                                     sample_data: Dict[str, Any] = None) -> str:
        """Create prompt for domain analysis"""
        prompt = f"""
        Analyze the following query and available database tables to determine the business domain.
        
        Query: "{query}"
        Available Tables: {', '.join(available_tables[:20])}
        
        """
        
        if sample_data:
            prompt += f"Sample Data: {json.dumps(sample_data, indent=2)}\n"
        
        prompt += """
        Please identify:
        1. Business domain (e.g., healthcare, ecommerce, finance, manufacturing, etc.)
        2. Key business concepts and entities
        3. Relevant keywords for this domain
        4. Common query patterns
        5. Schema organization patterns
        
        Respond with a JSON structure containing:
        {
            "domain": "domain_name",
            "keywords": ["keyword1", "keyword2"],
            "core_concepts": ["concept1", "concept2"],
            "business_entities": ["entity1", "entity2"],
            "common_queries": ["query_pattern1", "query_pattern2"],
            "schema_patterns": ["pattern1", "pattern2"]
        }
        """
        
        return prompt
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        # This would use the configured LLM client
        # For now, return a generic response
        return '''
        {
            "domain": "business_intelligence",
            "keywords": ["data", "analysis", "reporting", "metrics"],
            "core_concepts": ["analytics", "insights", "performance", "trends"],
            "business_entities": ["customers", "products", "transactions", "reports"],
            "common_queries": ["show me", "analyze", "compare", "trend"],
            "schema_patterns": ["star_schema", "normalized", "dimensional"]
        }
        '''
    
    def _parse_domain_context(self, response: str) -> DomainContext:
        """Parse domain context from LLM response"""
        try:
            data = json.loads(response)
            return DomainContext(
                domain=data.get("domain", "general"),
                keywords=data.get("keywords", []),
                core_concepts=data.get("core_concepts", []),
                business_entities=data.get("business_entities", []),
                common_queries=data.get("common_queries", []),
                schema_patterns=data.get("schema_patterns", [])
            )
        except Exception as e:
            logger.error(f"Failed to parse domain context: {e}")
            return self._get_generic_domain_context()
    
    def _get_generic_domain_context(self) -> DomainContext:
        """Generic domain context when analysis fails"""
        return DomainContext(
            domain="general",
            keywords=["data", "table", "column"],
            core_concepts=["information", "retrieval"],
            business_entities=["entities", "records"],
            common_queries=["find", "show", "get"],
            schema_patterns=["standard", "relational"]
        ) 