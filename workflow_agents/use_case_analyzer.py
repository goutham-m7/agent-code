import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class UseCaseAnalysis:
    """Result of use case analysis"""
    domain: str
    business_context: str
    key_entities: List[str]
    data_flows: List[Dict[str, Any]]
    business_rules: List[str]
    comparison_strategies: List[str]
    schema_mapping: Dict[str, Any]
    analysis_summary: str

class UseCaseAnalyzer:
    """Analyzes use case and database schema to create context for LLM"""
    
    def __init__(self):
        self.bedrock_client = self._initialize_bedrock()
        self.analysis_cache = {}
        
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
    
    def analyze_use_case(self, problem_statement: str, database_schema: Dict[str, Any]) -> UseCaseAnalysis:
        """Analyze the use case and database schema to create context"""
        try:
            logger.info("Starting use case analysis...")
            
            # Check cache first
            cache_key = self._generate_cache_key(problem_statement, database_schema)
            if cache_key in self.analysis_cache:
                logger.info("Returning cached analysis")
                return self.analysis_cache[cache_key]
            
            # Step 1: Analyze problem statement
            domain_analysis = self._analyze_problem_statement(problem_statement)
            
            # Step 2: Analyze database schema
            schema_analysis = self._analyze_database_schema(database_schema, domain_analysis)
            
            # Step 3: Create comprehensive analysis
            analysis = self._create_comprehensive_analysis(domain_analysis, schema_analysis)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            
            logger.info(f"Use case analysis completed for domain: {analysis.domain}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze use case: {e}")
            return self._create_fallback_analysis(problem_statement, database_schema)
    
    def _analyze_problem_statement(self, problem_statement: str) -> Dict[str, Any]:
        """Analyze the problem statement to understand the business domain"""
        try:
            system_prompt = """You are an expert business analyst. Analyze the problem statement to understand:
            1. Business domain (e.g., healthcare, finance, ecommerce)
            2. Key business entities and concepts
            3. Main business challenges
            4. Data requirements
            5. Expected outcomes
            
            Return a JSON object with these fields."""
            
            prompt = f"""
            Analyze this problem statement:
            "{problem_statement}"
            
            Provide a comprehensive analysis in JSON format.
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for problem analysis")
                return self._analyze_problem_statement_fallback(problem_statement)
                
        except Exception as e:
            logger.error(f"Failed to analyze problem statement: {e}")
            return self._analyze_problem_statement_fallback(problem_statement)
    
    def _analyze_problem_statement_fallback(self, problem_statement: str) -> Dict[str, Any]:
        """Fallback analysis when Bedrock fails"""
        problem_lower = problem_statement.lower()
        
        # Simple keyword-based analysis
        if any(word in problem_lower for word in ["medicaid", "ura", "rebate", "drug", "pharmaceutical"]):
            domain = "healthcare_medicaid"
            key_entities = ["drugs", "pricing", "rebates", "manufacturers", "ura"]
        elif any(word in problem_lower for word in ["customer", "order", "product", "ecommerce"]):
            domain = "ecommerce"
            key_entities = ["customers", "orders", "products", "inventory"]
        elif any(word in problem_lower for word in ["transaction", "financial", "account", "payment"]):
            domain = "finance"
            key_entities = ["transactions", "accounts", "payments", "balances"]
        else:
            domain = "general"
            key_entities = ["data", "records", "entities"]
        
        return {
            "domain": domain,
            "key_entities": key_entities,
            "business_challenges": ["data_quality", "discrepancies", "analysis"],
            "data_requirements": ["structured_data", "comparisons", "reporting"]
        }
    
    def _analyze_database_schema(self, database_schema: Dict[str, Any], domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the database schema in context of the domain"""
        try:
            system_prompt = """You are an expert database analyst. Analyze the database schema to understand:
            1. Table relationships and dependencies
            2. Key business entities represented
            3. Data quality indicators
            4. Potential comparison strategies
            5. Schema optimization opportunities
            
            Return a JSON object with these fields."""
            
            # Create schema summary for analysis
            schema_summary = self._create_schema_summary(database_schema)
            
            prompt = f"""
            Analyze this database schema in the context of {domain_analysis.get('domain', 'business')} domain:
            
            Schema Summary:
            {json.dumps(schema_summary, indent=2)}
            
            Domain Context:
            {json.dumps(domain_analysis, indent=2)}
            
            Provide comprehensive analysis in JSON format.
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for schema analysis")
                return self._analyze_database_schema_fallback(database_schema, domain_analysis)
                
        except Exception as e:
            logger.error(f"Failed to analyze database schema: {e}")
            return self._analyze_database_schema_fallback(database_schema, domain_analysis)
    
    def _create_schema_summary(self, database_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the database schema for analysis"""
        summary = {
            "total_tables": len(database_schema),
            "table_summaries": {},
            "relationships": [],
            "data_types": {}
        }
        
        for table_name, table_data in database_schema.items():
            # Table summary
            summary["table_summaries"][table_name] = {
                "description": table_data.get("description", ""),
                "columns_count": len(table_data.get("columns", [])),
                "estimated_rows": table_data.get("estimated_rows", "Unknown"),
                "table_size": table_data.get("table_size", "Unknown")
            }
            
            # Data types
            columns = table_data.get("columns", [])
            for col in columns:
                data_type = col.get("data_type", "Unknown")
                if data_type not in summary["data_types"]:
                    summary["data_types"][data_type] = 0
                summary["data_types"][data_type] += 1
            
            # Relationships
            if "foreign_keys" in table_data:
                for fk in table_data["foreign_keys"]:
                    summary["relationships"].append({
                        "from_table": table_name,
                        "to_table": fk.get("referenced_table", ""),
                        "type": "foreign_key"
                    })
        
        return summary
    
    def _analyze_database_schema_fallback(self, database_schema: Dict[str, Any], domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback schema analysis when Bedrock fails"""
        return {
            "table_relationships": "basic_analysis",
            "key_entities": list(database_schema.keys())[:10],
            "data_quality_indicators": ["row_counts", "constraints", "indexes"],
            "comparison_strategies": ["time_based", "category_based", "value_based"]
        }
    
    def _create_comprehensive_analysis(self, domain_analysis: Dict[str, Any], schema_analysis: Dict[str, Any]) -> UseCaseAnalysis:
        """Create comprehensive use case analysis"""
        try:
            system_prompt = """You are an expert business analyst. Create a comprehensive analysis that combines:
            1. Domain understanding
            2. Schema analysis
            3. Business rules
            4. Data comparison strategies
            5. Implementation recommendations
            
            Return a JSON object with all required fields."""
            
            prompt = f"""
            Create a comprehensive analysis combining:
            
            Domain Analysis:
            {json.dumps(domain_analysis, indent=2)}
            
            Schema Analysis:
            {json.dumps(schema_analysis, indent=2)}
            
            Provide a complete analysis in JSON format including:
            - domain
            - business_context
            - key_entities
            - data_flows
            - business_rules
            - comparison_strategies
            - schema_mapping
            - analysis_summary
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                analysis_data = json.loads(response)
                return UseCaseAnalysis(
                    domain=analysis_data.get("domain", "unknown"),
                    business_context=analysis_data.get("business_context", ""),
                    key_entities=analysis_data.get("key_entities", []),
                    data_flows=analysis_data.get("data_flows", []),
                    business_rules=analysis_data.get("business_rules", []),
                    comparison_strategies=analysis_data.get("comparison_strategies", []),
                    schema_mapping=analysis_data.get("schema_mapping", {}),
                    analysis_summary=analysis_data.get("analysis_summary", "")
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for comprehensive analysis")
                return self._create_comprehensive_analysis_fallback(domain_analysis, schema_analysis)
                
        except Exception as e:
            logger.error(f"Failed to create comprehensive analysis: {e}")
            return self._create_comprehensive_analysis_fallback(domain_analysis, schema_analysis)
    
    def _create_comprehensive_analysis_fallback(self, domain_analysis: Dict[str, Any], schema_analysis: Dict[str, Any]) -> UseCaseAnalysis:
        """Fallback comprehensive analysis when Bedrock fails"""
        domain = domain_analysis.get("domain", "general")
        
        return UseCaseAnalysis(
            domain=domain,
            business_context=f"Business analysis for {domain} domain",
            key_entities=domain_analysis.get("key_entities", []),
            data_flows=[{"type": "standard", "description": "Basic data flow"}],
            business_rules=["data_validation", "business_logic", "quality_checks"],
            comparison_strategies=["time_based", "category_based", "value_based"],
            schema_mapping={"type": "standard", "description": "Basic schema mapping"},
            analysis_summary=f"Analysis completed for {domain} domain with fallback methods"
        )
    
    def _generate_cache_key(self, problem_statement: str, database_schema: Dict[str, Any]) -> str:
        """Generate cache key for analysis"""
        import hashlib
        
        # Create a hash of the inputs
        key_data = {
            "problem": problem_statement[:200],  # Truncate long statements
            "schema_tables": sorted(database_schema.keys())
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_analysis_for_llm_context(self, analysis: UseCaseAnalysis) -> str:
        """Format analysis for LLM context and memory"""
        context = f"""
# USE CASE ANALYSIS CONTEXT

## Domain: {analysis.domain}
## Business Context: {analysis.business_context}

## Key Entities:
{chr(10).join([f"- {entity}" for entity in analysis.key_entities])}

## Data Flows:
{chr(10).join([f"- {flow.get('type', 'Unknown')}: {flow.get('description', 'No description')}" for flow in analysis.data_flows])}

## Business Rules:
{chr(10).join([f"- {rule}" for rule in analysis.business_rules])}

## Comparison Strategies:
{chr(10).join([f"- {strategy}" for strategy in analysis.comparison_strategies])}

## Schema Mapping:
{json.dumps(analysis.schema_mapping, indent=2)}

## Analysis Summary:
{analysis.analysis_summary}

---
Use this context to understand the business domain and guide your analysis.
"""
        return context
    
    def clear_cache(self):
        """Clear the analysis cache"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.analysis_cache),
            "cache_keys": list(self.analysis_cache.keys())
        } 