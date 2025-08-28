import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SchemaLevel(Enum):
    """Schema detail levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"

@dataclass
class SchemaLevelInfo:
    """Information about a schema level"""
    level: SchemaLevel
    description: str
    estimated_tokens: int
    includes: List[str]
    excludes: List[str]

class HierarchicalSchemaManager:
    """Manages hierarchical schema representation with multiple detail levels"""
    
    def __init__(self):
        self.schema_levels = self._initialize_schema_levels()
        self.level_cache = {}
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
    def _initialize_schema_levels(self) -> Dict[SchemaLevel, SchemaLevelInfo]:
        """Initialize the different schema levels"""
        return {
            SchemaLevel.BASIC: SchemaLevelInfo(
                level=SchemaLevel.BASIC,
                description="Basic table information - names, descriptions, and key columns",
                estimated_tokens=1000,
                includes=["table_name", "description", "key_columns", "table_type"],
                excludes=["detailed_constraints", "relationships", "indexes", "triggers", "full_column_details"]
            ),
            SchemaLevel.DETAILED: SchemaLevelInfo(
                level=SchemaLevel.DETAILED,
                description="Detailed schema with columns, basic relationships, and constraints",
                estimated_tokens=5000,
                includes=["table_name", "description", "columns", "basic_relationships", "primary_keys", "foreign_keys"],
                excludes=["detailed_indexes", "triggers", "stored_procedures", "complex_constraints"]
            ),
            SchemaLevel.FULL: SchemaLevelInfo(
                level=SchemaLevel.FULL,
                description="Complete schema with all details, relationships, and metadata",
                estimated_tokens=15000,
                includes=["table_name", "description", "columns", "relationships", "constraints", "indexes", "triggers", "metadata"],
                excludes=[]
            )
        }
    
    def get_schema_level(self, query_complexity: str, required_detail: str = None) -> SchemaLevel:
        """Determine appropriate schema level based on query complexity"""
        if required_detail:
            # User specified detail level
            try:
                return SchemaLevel(required_detail.lower())
            except ValueError:
                logger.warning(f"Invalid detail level: {required_detail}, using complexity-based selection")
        
        # Auto-determine based on complexity
        if query_complexity == "simple":
            return SchemaLevel.BASIC
        elif query_complexity == "moderate":
            return SchemaLevel.DETAILED
        else:  # complex
            return SchemaLevel.FULL
    
    def get_level_1_schema(self, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get basic schema information (Level 1)"""
        basic_schema = {}
        
        for table_name, table_data in full_schema.items():
            basic_schema[table_name] = {
                "table_name": table_name,
                "description": table_data.get("description", f"Table {table_name}"),
                "table_type": table_data.get("table_type", "TABLE"),
                "key_columns": self._extract_key_columns(table_data),
                "estimated_rows": table_data.get("estimated_rows", "Unknown"),
                "last_analyzed": table_data.get("last_analyzed", "Unknown")
            }
        
        return basic_schema
    
    def get_level_2_schema(self, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed schema information (Level 2)"""
        detailed_schema = {}
        
        for table_name, table_data in full_schema.items():
            detailed_schema[table_name] = {
                "table_name": table_name,
                "description": table_data.get("description", f"Table {table_name}"),
                "columns": self._extract_column_summary(table_data),
                "primary_keys": table_data.get("primary_keys", []),
                "foreign_keys": self._extract_foreign_keys(table_data),
                "basic_relationships": self._extract_basic_relationships(table_data),
                "table_size": table_data.get("table_size", "Unknown"),
                "indexes": self._extract_basic_indexes(table_data)
            }
        
        return detailed_schema
    
    def get_level_3_schema(self, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get full schema information (Level 3)"""
        # Return the full schema as-is
        return full_schema
    
    def _extract_key_columns(self, table_data: Dict[str, Any]) -> List[str]:
        """Extract key columns from table data"""
        key_columns = []
        
        # Primary keys
        if "primary_keys" in table_data:
            key_columns.extend(table_data["primary_keys"])
        
        # Foreign keys
        if "foreign_keys" in table_data:
            for fk in table_data["foreign_keys"]:
                if "columns" in fk:
                    key_columns.extend(fk["columns"])
        
        # Unique columns
        if "unique_constraints" in table_data:
            for constraint in table_data["unique_constraints"]:
                if "columns" in constraint:
                    key_columns.extend(constraint["columns"])
        
        return list(set(key_columns))  # Remove duplicates
    
    def _extract_column_summary(self, table_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract summary of columns"""
        columns = table_data.get("columns", [])
        column_summary = []
        
        for col in columns:
            summary = {
                "name": col.get("name", ""),
                "data_type": col.get("data_type", ""),
                "nullable": col.get("nullable", "Y"),
                "description": col.get("description", "")
            }
            
            # Add key information
            if col.get("name") in table_data.get("primary_keys", []):
                summary["is_primary_key"] = True
            
            column_summary.append(summary)
        
        return column_summary
    
    def _extract_foreign_keys(self, table_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract foreign key information"""
        foreign_keys = table_data.get("foreign_keys", [])
        fk_summary = []
        
        for fk in foreign_keys:
            summary = {
                "name": fk.get("name", ""),
                "columns": fk.get("columns", []),
                "referenced_table": fk.get("referenced_table", ""),
                "referenced_columns": fk.get("referenced_columns", [])
            }
            fk_summary.append(summary)
        
        return fk_summary
    
    def _extract_basic_relationships(self, table_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract basic relationship information"""
        relationships = []
        
        # Foreign key relationships
        for fk in table_data.get("foreign_keys", []):
            relationships.append({
                "type": "foreign_key",
                "target_table": fk.get("referenced_table", ""),
                "relationship": "many-to-one"
            })
        
        # Referenced by relationships (reverse foreign keys)
        for ref in table_data.get("referenced_by", []):
            relationships.append({
                "type": "referenced_by",
                "source_table": ref.get("source_table", ""),
                "relationship": "one-to-many"
            })
        
        return relationships
    
    def _extract_basic_indexes(self, table_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract basic index information"""
        indexes = table_data.get("indexes", [])
        index_summary = []
        
        for idx in indexes:
            summary = {
                "name": idx.get("name", ""),
                "columns": idx.get("columns", []),
                "type": idx.get("type", "NORMAL"),
                "unique": idx.get("unique", False)
            }
            index_summary.append(summary)
        
        return index_summary
    
    def get_schema_by_level(self, level: SchemaLevel, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get schema at the specified level"""
        cache_key = f"{level.value}_{hash(json.dumps(full_schema, sort_keys=True))}"
        
        if cache_key in self.level_cache:
            logger.info(f"Returning cached schema for level {level.value}")
            return self.level_cache[cache_key]
        
        # Generate schema for the level
        if level == SchemaLevel.BASIC:
            schema = self.get_level_1_schema(full_schema)
        elif level == SchemaLevel.DETAILED:
            schema = self.get_level_2_schema(full_schema)
        else:  # FULL
            schema = self.get_level_3_schema(full_schema)
        
        # Cache the result
        self.level_cache[cache_key] = schema
        
        return schema
    
    def get_adaptive_schema(self, user_query: str, full_schema: Dict[str, Any], 
                           max_tokens: int = 5000) -> Dict[str, Any]:
        """Get schema that adapts to query complexity and token limits"""
        try:
            # Analyze query complexity
            complexity = self.complexity_analyzer.analyze_complexity(user_query)
            logger.info(f"Query complexity: {complexity}")
            
            # Determine appropriate level
            level = self.get_schema_level(complexity)
            logger.info(f"Selected schema level: {level.value}")
            
            # Get schema for the level
            schema = self.get_schema_by_level(level, full_schema)
            
            # Check if we need to reduce detail further due to token limits
            estimated_tokens = self._estimate_schema_tokens(schema)
            
            if estimated_tokens > max_tokens:
                logger.info(f"Schema too large ({estimated_tokens} tokens), reducing detail")
                schema = self._reduce_schema_detail(schema, max_tokens)
            
            # Add metadata
            result = {
                "schema": schema,
                "metadata": {
                    "level": level.value,
                    "complexity": complexity,
                    "estimated_tokens": self._estimate_schema_tokens(schema),
                    "max_tokens": max_tokens,
                    "tables_included": list(schema.keys()),
                    "efficiency": f"{(self._estimate_schema_tokens(schema)/max_tokens)*100:.1f}%"
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get adaptive schema: {e}")
            # Fallback to basic schema
            return {
                "schema": self.get_level_1_schema(full_schema),
                "metadata": {
                    "level": "basic",
                    "error": str(e),
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
            return 1000  # Default estimate
    
    def _reduce_schema_detail(self, schema: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Reduce schema detail to fit within token limits"""
        reduced_schema = {}
        current_tokens = 0
        
        for table_name, table_data in schema.items():
            # Create minimal table info
            minimal_table = {
                "table_name": table_name,
                "description": table_data.get("description", ""),
                "key_columns": table_data.get("key_columns", [])
            }
            
            table_tokens = self._estimate_schema_tokens(minimal_table)
            
            if current_tokens + table_tokens <= max_tokens:
                reduced_schema[table_name] = minimal_table
                current_tokens += table_tokens
            else:
                logger.info(f"Stopping at table {table_name} due to token limit")
                break
        
        return reduced_schema
    
    def get_level_comparison(self, table_name: str, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different levels for a specific table"""
        if table_name not in full_schema:
            return {"error": f"Table {table_name} not found"}
        
        table_data = full_schema[table_name]
        
        comparison = {
            "table_name": table_name,
            "levels": {}
        }
        
        for level in SchemaLevel:
            level_schema = self.get_schema_by_level(level, {table_name: table_data})
            comparison["levels"][level.value] = {
                "estimated_tokens": self._estimate_schema_tokens(level_schema),
                "includes": self.schema_levels[level].includes,
                "excludes": self.schema_levels[level].excludes,
                "description": self.schema_levels[level].description
            }
        
        return comparison
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "level_cache_size": len(self.level_cache),
            "cache_keys": list(self.level_cache.keys()),
            "schema_levels": {level.value: info.description for level, info in self.schema_levels.items()}
        }
    
    def clear_cache(self):
        """Clear the level cache"""
        self.level_cache.clear()
        logger.info("Level cache cleared")

class QueryComplexityAnalyzer:
    """Analyzes query complexity to determine appropriate schema level"""
    
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