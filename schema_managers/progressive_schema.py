import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SchemaComplexity(Enum):
    """Schema complexity levels"""
    MINIMAL = "minimal"
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"

@dataclass
class SchemaLoadRequest:
    """Request for schema loading"""
    query: str
    complexity: str
    required_tables: List[str]
    max_tokens: int
    timestamp: float
    user_context: Dict[str, Any] = None

@dataclass
class SchemaLoadResult:
    """Result of schema loading"""
    schema: Dict[str, Any]
    complexity: SchemaComplexity
    tokens_used: int
    tables_included: List[str]
    load_time: float
    metadata: Dict[str, Any]

class ProgressiveSchemaLoader:
    """Loads schema progressively based on query complexity and adapts dynamically."""
    
    def __init__(self):
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.schema_cache = {}
        self.load_history = []
        self.performance_metrics = {}
        self.adaptive_thresholds = {
            "minimal": {"tokens": 1000, "tables": 2},
            "basic": {"tokens": 3000, "tables": 5},
            "detailed": {"tokens": 6000, "tables": 8},
            "full": {"tokens": 10000, "tables": 15}
        }
    
    def load_schema_progressively(self, user_query: str, context: Dict[str, Any] = None, 
                                 max_tokens: int = 5000) -> SchemaLoadResult:
        """Load schema progressively based on query analysis"""
        start_time = time.time()
        
        try:
            # Create load request
            request = SchemaLoadRequest(
                query=user_query,
                complexity=self.complexity_analyzer.analyze_complexity(user_query),
                required_tables=context.get("required_tables", []) if context else [],
                max_tokens=max_tokens,
                timestamp=start_time,
                user_context=context
            )
            
            logger.info(f"Loading schema for query: {user_query[:100]}...")
            logger.info(f"Detected complexity: {request.complexity}")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.schema_cache:
                cached_result = self.schema_cache[cache_key]
                logger.info("Returning cached schema")
                
                # Update performance metrics
                self._update_performance_metrics("cache_hit", time.time() - start_time)
                
                return cached_result
            
            # Determine initial complexity level
            initial_complexity = self._determine_initial_complexity(request)
            logger.info(f"Initial complexity level: {initial_complexity.value}")
            
            # Load schema at initial level
            schema_result = self._load_schema_at_level(request, initial_complexity)
            
            # Check if we need to adjust complexity
            adjusted_complexity = self._adjust_complexity_if_needed(request, schema_result)
            
            if adjusted_complexity != initial_complexity:
                logger.info(f"Adjusting complexity from {initial_complexity.value} to {adjusted_complexity.value}")
                schema_result = self._load_schema_at_level(request, adjusted_complexity)
            
            # Create final result
            load_time = time.time() - start_time
            result = SchemaLoadResult(
                schema=schema_result["schema"],
                complexity=adjusted_complexity,
                tokens_used=schema_result["tokens_used"],
                tables_included=schema_result["tables_included"],
                load_time=load_time,
                metadata={
                    "initial_complexity": initial_complexity.value,
                    "final_complexity": adjusted_complexity.value,
                    "adjustment_reason": schema_result.get("adjustment_reason", "none"),
                    "cache_key": cache_key,
                    "performance_metrics": self._get_performance_summary()
                }
            )
            
            # Cache the result
            self.schema_cache[cache_key] = result
            
            # Update history and metrics
            self._update_load_history(request, result)
            self._update_performance_metrics("load_time", load_time)
            
            logger.info(f"Schema loaded successfully: {adjusted_complexity.value} complexity, {result.tokens_used} tokens")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load schema progressively: {e}")
            # Return minimal fallback schema
            return self._create_fallback_schema(user_query, max_tokens, time.time() - start_time)
    
    def _determine_initial_complexity(self, request: SchemaLoadRequest) -> SchemaComplexity:
        """Determine initial complexity level based on query analysis"""
        complexity_mapping = {
            "simple": SchemaComplexity.MINIMAL,
            "moderate": SchemaComplexity.BASIC,
            "complex": SchemaComplexity.DETAILED
        }
        
        # Start with query complexity
        initial_level = complexity_mapping.get(request.complexity, SchemaComplexity.BASIC)
        
        # Adjust based on user context
        if request.user_context:
            # Check for specific requirements
            if request.user_context.get("require_full_schema"):
                initial_level = SchemaComplexity.FULL
            elif request.user_context.get("require_detailed_schema"):
                initial_level = SchemaComplexity.DETAILED
            elif request.user_context.get("minimal_schema_only"):
                initial_level = SchemaComplexity.MINIMAL
            
            # Check for domain-specific requirements
            domain = request.user_context.get("domain", "general")
            if domain == "medicaid" and "ura" in request.query.lower():
                # Medicaid URA queries often need detailed schema
                initial_level = max(initial_level, SchemaComplexity.DETAILED)
        
        # Adjust based on token limits
        max_tokens = request.max_tokens
        if max_tokens < 2000:
            initial_level = SchemaComplexity.MINIMAL
        elif max_tokens < 4000:
            initial_level = min(initial_level, SchemaComplexity.BASIC)
        elif max_tokens < 8000:
            initial_level = min(initial_level, SchemaComplexity.DETAILED)
        
        return initial_level
    
    def _load_schema_at_level(self, request: SchemaLoadRequest, 
                             complexity: SchemaComplexity) -> Dict[str, Any]:
        """Load schema at a specific complexity level"""
        try:
            # Get available tables (this would come from your database connection)
            available_tables = self._get_available_tables(request)
            
            # Get tables based on complexity level
            selected_tables = self._select_tables_for_complexity(
                available_tables, complexity, request.required_tables
            )
            
            # Load schema for selected tables
            schema = self._load_table_schemas(selected_tables, complexity)
            
            # Calculate token usage
            tokens_used = self._estimate_schema_tokens(schema)
            
            # Check if we need to reduce complexity
            adjustment_reason = None
            if tokens_used > request.max_tokens:
                logger.warning(f"Schema too large ({tokens_used} tokens), reducing complexity")
                schema = self._reduce_schema_complexity(schema, request.max_tokens)
                tokens_used = self._estimate_schema_tokens(schema)
                adjustment_reason = "token_limit_exceeded"
            
            return {
                "schema": schema,
                "tokens_used": tokens_used,
                "tables_included": selected_tables,
                "adjustment_reason": adjustment_reason
            }
            
        except Exception as e:
            logger.error(f"Failed to load schema at level {complexity.value}: {e}")
            raise
    
    def _select_tables_for_complexity(self, available_tables: List[str], 
                                    complexity: SchemaComplexity, 
                                    required_tables: List[str]) -> List[str]:
        """Select tables based on complexity level"""
        # Always include required tables
        selected_tables = required_tables.copy()
        
        # Get complexity limits
        limits = self.adaptive_thresholds[complexity.value]
        max_tables = limits["tables"]
        
        # Add additional tables based on complexity
        remaining_slots = max_tables - len(selected_tables)
        
        if remaining_slots > 0:
            # Add tables based on complexity
            if complexity == SchemaComplexity.MINIMAL:
                # Only essential tables
                additional_tables = self._get_essential_tables(available_tables)
            elif complexity == SchemaComplexity.BASIC:
                # Core tables
                additional_tables = self._get_core_tables(available_tables)
            elif complexity == SchemaComplexity.DETAILED:
                # Extended tables
                additional_tables = self._get_extended_tables(available_tables)
            else:  # FULL
                # All relevant tables
                additional_tables = available_tables
            
            # Filter out already selected tables
            additional_tables = [t for t in additional_tables if t not in selected_tables]
            
            # Add up to remaining slots
            selected_tables.extend(additional_tables[:remaining_slots])
        
        return selected_tables[:max_tables]
    
    def _get_essential_tables(self, available_tables: List[str]) -> List[str]:
        """Get essential tables for minimal complexity"""
        # Define essential table patterns
        essential_patterns = [
            "user", "customer", "patient", "account", "main", "master", "core"
        ]
        
        essential_tables = []
        for table in available_tables:
            table_lower = table.lower()
            if any(pattern in table_lower for pattern in essential_patterns):
                essential_tables.append(table)
        
        return essential_tables[:3]  # Limit to 3 essential tables
    
    def _get_core_tables(self, available_tables: List[str]) -> List[str]:
        """Get core tables for basic complexity"""
        # Include essential tables plus some core business tables
        core_tables = self._get_essential_tables(available_tables)
        
        # Add some core business tables
        core_patterns = [
            "order", "product", "transaction", "payment", "inventory"
        ]
        
        for table in available_tables:
            if len(core_tables) >= 5:  # Limit to 5 total
                break
            
            table_lower = table.lower()
            if any(pattern in table_lower for pattern in core_patterns):
                if table not in core_tables:
                    core_tables.append(table)
        
        return core_tables
    
    def _get_extended_tables(self, available_tables: List[str]) -> List[str]:
        """Get extended tables for detailed complexity"""
        # Include core tables plus extended business tables
        extended_tables = self._get_core_tables(available_tables)
        
        # Add extended tables
        extended_patterns = [
            "history", "log", "audit", "detail", "item", "line", "config", "setting"
        ]
        
        for table in available_tables:
            if len(extended_tables) >= 8:  # Limit to 8 total
                break
            
            table_lower = table.lower()
            if any(pattern in table_lower for pattern in extended_patterns):
                if table not in extended_tables:
                    extended_tables.append(table)
        
        return extended_tables
    
    def _load_table_schemas(self, table_names: List[str], 
                           complexity: SchemaComplexity) -> Dict[str, Any]:
        """Load schema for specific tables at given complexity level"""
        schemas = {}
        
        for table_name in table_names:
            # This would integrate with your actual database connection
            # For now, we'll create mock schemas
            table_schema = self._create_mock_table_schema(table_name, complexity)
            schemas[table_name] = table_schema
        
        return schemas
    
    def _create_mock_table_schema(self, table_name: str, 
                                 complexity: SchemaComplexity) -> Dict[str, Any]:
        """Create mock table schema for testing (replace with actual DB calls)"""
        base_schema = {
            "table_name": table_name,
            "description": f"Table for {table_name} data"
        }
        
        if complexity == SchemaComplexity.MINIMAL:
            return base_schema
        
        elif complexity == SchemaComplexity.BASIC:
            base_schema.update({
                "columns": [
                    {"name": "ID", "type": "NUMBER", "nullable": "N"},
                    {"name": "NAME", "type": "VARCHAR2", "nullable": "Y"}
                ],
                "primary_keys": ["ID"]
            })
        
        elif complexity == SchemaComplexity.DETAILED:
            base_schema.update({
                "columns": [
                    {"name": "ID", "type": "NUMBER", "nullable": "N", "description": "Primary key"},
                    {"name": "NAME", "type": "VARCHAR2", "nullable": "Y", "description": "Name field"},
                    {"name": "CREATED_DATE", "type": "DATE", "nullable": "N", "description": "Creation date"}
                ],
                "primary_keys": ["ID"],
                "indexes": [{"name": "IDX_NAME", "columns": ["NAME"]}],
                "estimated_rows": 1000
            })
        
        else:  # FULL
            base_schema.update({
                "columns": [
                    {"name": "ID", "type": "NUMBER", "nullable": "N", "description": "Primary key", "precision": 10},
                    {"name": "NAME", "type": "VARCHAR2", "nullable": "Y", "description": "Name field", "length": 100},
                    {"name": "CREATED_DATE", "type": "DATE", "nullable": "N", "description": "Creation date"},
                    {"name": "UPDATED_DATE", "type": "DATE", "nullable": "Y", "description": "Last update date"},
                    {"name": "STATUS", "type": "VARCHAR2", "nullable": "N", "description": "Status", "length": 20}
                ],
                "primary_keys": ["ID"],
                "foreign_keys": [],
                "unique_constraints": [{"name": "UK_NAME", "columns": ["NAME"]}],
                "indexes": [
                    {"name": "IDX_NAME", "columns": ["NAME"], "type": "NORMAL"},
                    {"name": "IDX_STATUS", "columns": ["STATUS"], "type": "NORMAL"}
                ],
                "estimated_rows": 1000,
                "table_size": "1MB",
                "last_analyzed": "2024-01-01"
            })
        
        return base_schema
    
    def _adjust_complexity_if_needed(self, request: SchemaLoadRequest, 
                                   schema_result: Dict[str, Any]) -> SchemaComplexity:
        """Adjust complexity if needed based on results"""
        current_complexity = self._determine_initial_complexity(request)
        
        # Check if we need to increase complexity
        if schema_result["tokens_used"] < request.max_tokens * 0.3:  # Using less than 30% of tokens
            # Consider increasing complexity
            if current_complexity == SchemaComplexity.MINIMAL:
                return SchemaComplexity.BASIC
            elif current_complexity == SchemaComplexity.BASIC:
                return SchemaComplexity.DETAILED
        
        # Check if we need to decrease complexity
        if schema_result["tokens_used"] > request.max_tokens * 0.9:  # Using more than 90% of tokens
            # Consider decreasing complexity
            if current_complexity == SchemaComplexity.FULL:
                return SchemaComplexity.DETAILED
            elif current_complexity == SchemaComplexity.DETAILED:
                return SchemaComplexity.BASIC
            elif current_complexity == SchemaComplexity.BASIC:
                return SchemaComplexity.MINIMAL
        
        return current_complexity
    
    def _reduce_schema_complexity(self, schema: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Reduce schema complexity to fit within token limits"""
        reduced_schema = {}
        current_tokens = 0
        
        for table_name, table_data in schema.items():
            # Create minimal table info
            minimal_table = {
                "table_name": table_name,
                "description": table_data.get("description", "")
            }
            
            # Add basic columns if we have space
            if "columns" in table_data:
                basic_columns = []
                for col in table_data["columns"][:3]:  # Limit to 3 columns
                    basic_columns.append({
                        "name": col.get("name", ""),
                        "type": col.get("type", "")
                    })
                minimal_table["columns"] = basic_columns
            
            table_tokens = self._estimate_schema_tokens(minimal_table)
            
            if current_tokens + table_tokens <= max_tokens:
                reduced_schema[table_name] = minimal_table
                current_tokens += table_tokens
            else:
                logger.info(f"Stopping at table {table_name} due to token limit")
                break
        
        return reduced_schema
    
    def _estimate_schema_tokens(self, schema: Dict[str, Any]) -> int:
        """Estimate token count for schema"""
        try:
            schema_text = json.dumps(schema, indent=2)
            # Rough estimation: 1 token ≈ 4 characters
            return len(schema_text) // 4
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}")
            return 1000  # Default estimate
    
    def _get_available_tables(self, request: SchemaLoadRequest) -> List[str]:
        """Get available tables (integrate with your database connection)"""
        # This would call your database connection
        # For now, return mock tables
        return [
            "USERS", "CUSTOMERS", "ORDERS", "PRODUCTS", "INVENTORY",
            "PAYMENTS", "SHIPPING", "AUDIT_LOGS", "CONFIGURATION",
            "DRUGS", "PRICING", "REBATES", "MANUFACTURERS", "NDC_CODES"
        ]
    
    def _generate_cache_key(self, request: SchemaLoadRequest) -> str:
        """Generate cache key for schema request"""
        key_data = {
            "query": request.query.lower().strip()[:100],
            "complexity": request.complexity,
            "required_tables": sorted(request.required_tables),
            "max_tokens": request.max_tokens
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _update_load_history(self, request: SchemaLoadRequest, result: SchemaLoadResult):
        """Update load history for analysis"""
        history_entry = {
            "timestamp": request.timestamp,
            "query": request.query[:100],
            "initial_complexity": request.complexity,
            "final_complexity": result.complexity.value,
            "tokens_used": result.tokens_used,
            "load_time": result.load_time,
            "tables_count": len(result.tables_included)
        }
        
        self.load_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.load_history) > 100:
            self.load_history = self.load_history[-100:]
    
    def _update_performance_metrics(self, metric: str, value: float):
        """Update performance metrics"""
        if metric not in self.performance_metrics:
            self.performance_metrics[metric] = []
        
        self.performance_metrics[metric].append(value)
        
        # Keep only last 1000 values
        if len(self.performance_metrics[metric]) > 1000:
            self.performance_metrics[metric] = self.performance_metrics[metric][-1000:]
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for metric, values in self.performance_metrics.items():
            if values:
                summary[metric] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0
                }
        
        return summary
    
    def _create_fallback_schema(self, query: str, max_tokens: int, 
                               load_time: float) -> SchemaLoadResult:
        """Create fallback schema when loading fails"""
        fallback_schema = {
            "FALLBACK_TABLE": {
                "table_name": "FALLBACK_TABLE",
                "description": "Fallback table for failed schema loading"
            }
        }
        
        return SchemaLoadResult(
            schema=fallback_schema,
            complexity=SchemaComplexity.MINIMAL,
            tokens_used=100,
            tables_included=["FALLBACK_TABLE"],
            load_time=load_time,
            metadata={
                "error": "Schema loading failed, using fallback",
                "fallback": True
            }
        )
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get statistics about schema loading"""
        if not self.load_history:
            return {"message": "No load history available"}
        
        # Analyze complexity distribution
        complexity_counts = {}
        load_times = []
        token_usage = []
        
        for entry in self.load_history:
            complexity_counts[entry["final_complexity"]] = complexity_counts.get(entry["final_complexity"], 0) + 1
            load_times.append(entry["load_time"])
            token_usage.append(entry["tokens_used"])
        
        return {
            "total_loads": len(self.load_history),
            "complexity_distribution": complexity_counts,
            "performance": {
                "average_load_time": sum(load_times) / len(load_times) if load_times else 0,
                "average_tokens": sum(token_usage) / len(token_usage) if token_usage else 0,
                "cache_size": len(self.schema_cache)
            },
            "recent_loads": self.load_history[-10:] if self.load_history else []
        }
    
    def clear_cache(self):
        """Clear the schema cache"""
        self.schema_cache.clear()
        logger.info("Schema cache cleared")
    
    def update_adaptive_thresholds(self, new_thresholds: Dict[str, Dict[str, int]]):
        """Update adaptive thresholds"""
        for complexity, limits in new_thresholds.items():
            if complexity in self.adaptive_thresholds:
                self.adaptive_thresholds[complexity].update(limits)
                logger.info(f"Updated thresholds for {complexity}: {limits}")
            else:
                logger.warning(f"Unknown complexity level: {complexity}")

class QueryComplexityAnalyzer:
    """Analyzes query complexity for progressive schema loading"""
    
    def __init__(self):
        self.complexity_patterns = {
            "simple": [
                "count", "how many", "total", "show", "list", "get", "find",
                "basic", "simple", "overview", "summary"
            ],
            "moderate": [
                "analyze", "compare", "group by", "order by", "where", "join",
                "filter", "sort", "aggregate", "calculate"
            ],
            "complex": [
                "complex", "advanced", "detailed", "comprehensive", "full",
                "subquery", "cte", "window function", "pivot", "unpivot",
                "recursive", "hierarchical", "multi-level"
            ]
        }
    
    def analyze_complexity(self, query: str) -> str:
        """Analyze query complexity"""
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_scores = {"simple": 0, "moderate": 0, "complex": 0}
        
        for level, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    complexity_scores[level] += 1
        
        # Determine complexity based on scores
        if complexity_scores["complex"] > 0:
            return "complex"
        elif complexity_scores["moderate"] > 0:
            return "moderate"
        else:
            return "simple" 