import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent types"""
    COUNT = "count"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    SEARCH = "search"
    AGGREGATION = "aggregation"
    RELATIONSHIP = "relationship"
    QUALITY = "quality"
    DISCREPANCY = "discrepancy"
    REPORTING = "reporting"
    MAINTENANCE = "maintenance"

@dataclass
class IntentPattern:
    """Pattern for identifying query intent"""
    intent: QueryIntent
    keywords: List[str]
    patterns: List[str]
    required_tables: List[str]
    optional_tables: List[str]
    schema_complexity: str

class IntentBasedSchemaSelector:
    """Selects schema based on query intent classification"""
    
    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.domain_patterns = self._initialize_domain_patterns()
        self.intent_cache = {}
        self.confidence_threshold = 0.6
        
    def _initialize_intent_patterns(self) -> List[IntentPattern]:
        """Initialize patterns for different query intents"""
        return [
            IntentPattern(
                intent=QueryIntent.COUNT,
                keywords=["count", "how many", "total", "number of", "quantity"],
                patterns=[r"count\s+\*", r"how\s+many", r"total\s+number"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="basic"
            ),
            IntentPattern(
                intent=QueryIntent.ANALYSIS,
                keywords=["analyze", "analysis", "examine", "investigate", "study"],
                patterns=[r"analyze", r"analysis", r"examine", r"investigate"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="detailed"
            ),
            IntentPattern(
                intent=QueryIntent.COMPARISON,
                keywords=["compare", "difference", "versus", "vs", "against", "mismatch"],
                patterns=[r"compare", r"difference", r"versus", r"mismatch"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="detailed"
            ),
            IntentPattern(
                intent=QueryIntent.SEARCH,
                keywords=["find", "search", "locate", "get", "retrieve", "show"],
                patterns=[r"find", r"search", r"locate", r"get", r"retrieve"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="moderate"
            ),
            IntentPattern(
                intent=QueryIntent.AGGREGATION,
                keywords=["sum", "average", "avg", "min", "max", "group by", "aggregate"],
                patterns=[r"sum\s*\(", r"avg\s*\(", r"group\s+by", r"aggregate"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="moderate"
            ),
            IntentPattern(
                intent=QueryIntent.RELATIONSHIP,
                keywords=["join", "relationship", "connect", "link", "associate"],
                patterns=[r"join", r"relationship", r"connect", r"link"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="detailed"
            ),
            IntentPattern(
                intent=QueryIntent.QUALITY,
                keywords=["quality", "validate", "check", "verify", "integrity"],
                patterns=[r"quality", r"validate", r"check", r"verify"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="detailed"
            ),
            IntentPattern(
                intent=QueryIntent.DISCREPANCY,
                keywords=["discrepancy", "mismatch", "error", "inconsistency", "difference"],
                patterns=[r"discrepancy", r"mismatch", r"error", r"inconsistency"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="full"
            ),
            IntentPattern(
                intent=QueryIntent.REPORTING,
                keywords=["report", "summary", "overview", "dashboard", "metrics"],
                patterns=[r"report", r"summary", r"overview", r"dashboard"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="moderate"
            ),
            IntentPattern(
                intent=QueryIntent.MAINTENANCE,
                keywords=["maintain", "update", "insert", "delete", "modify", "clean"],
                patterns=[r"maintain", r"update", r"insert", r"delete", r"modify"],
                required_tables=[],
                optional_tables=[],
                schema_complexity="detailed"
            )
        ]
    
    def _initialize_domain_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific patterns"""
        return {
            "medicaid": {
                "keywords": ["medicaid", "ura", "rebate", "amp", "bp", "cpi", "ndc", "manufacturer"],
                "core_tables": ["DRUGS", "PRICING", "REBATES", "MANUFACTURERS", "NDC_CODES"],
                "formula_tables": ["URA_FORMULAS", "CPI_DATA", "BASELINE_DATA"],
                "comparison_tables": ["MEDICAID_DATA", "CALCULATED_URA", "DISCREPANCIES"],
                "schema_complexity": "full"
            },
            "ecommerce": {
                "keywords": ["customer", "order", "product", "inventory", "payment", "shipping"],
                "core_tables": ["CUSTOMERS", "ORDERS", "PRODUCTS", "INVENTORY"],
                "analysis_tables": ["SALES", "REVENUE", "ANALYTICS"],
                "relationship_tables": ["ORDER_ITEMS", "CUSTOMER_ADDRESSES"],
                "schema_complexity": "detailed"
            },
            "healthcare": {
                "keywords": ["patient", "diagnosis", "treatment", "medication", "provider"],
                "core_tables": ["PATIENTS", "DIAGNOSES", "TREATMENTS", "MEDICATIONS"],
                "analysis_tables": ["PATIENT_HISTORY", "TREATMENT_OUTCOMES"],
                "relationship_tables": ["PATIENT_PROVIDERS", "MEDICATION_PRESCRIPTIONS"],
                "schema_complexity": "detailed"
            },
            "financial": {
                "keywords": ["account", "transaction", "balance", "payment", "invoice"],
                "core_tables": ["ACCOUNTS", "TRANSACTIONS", "BALANCES", "PAYMENTS"],
                "analysis_tables": ["FINANCIAL_REPORTS", "REVENUE_ANALYSIS"],
                "relationship_tables": ["ACCOUNT_HOLDERS", "TRANSACTION_DETAILS"],
                "schema_complexity": "detailed"
            }
        }
    
    def classify_query_intent(self, query: str) -> Tuple[QueryIntent, float, Dict[str, Any]]:
        """Classify the intent of a user query"""
        query_lower = query.lower()
        
        best_intent = None
        best_score = 0.0
        best_pattern = None
        
        for pattern in self.intent_patterns:
            score = self._calculate_intent_score(query_lower, pattern)
            
            if score > best_score:
                best_score = score
                best_intent = pattern.intent
                best_pattern = pattern
        
        # Normalize score
        normalized_score = min(best_score / 10.0, 1.0)  # Cap at 1.0
        
        # Get domain context
        domain_context = self._identify_domain_context(query_lower)
        
        result = {
            "intent": best_intent.value if best_intent else "unknown",
            "confidence": normalized_score,
            "pattern": best_pattern.intent.value if best_pattern else None,
            "domain": domain_context,
            "keywords_found": self._extract_found_keywords(query_lower, best_pattern) if best_pattern else []
        }
        
        return best_intent, normalized_score, result
    
    def _calculate_intent_score(self, query: str, pattern: IntentPattern) -> float:
        """Calculate score for intent pattern matching"""
        score = 0.0
        
        # Keyword matching
        for keyword in pattern.keywords:
            if keyword in query:
                score += 2.0
        
        # Pattern matching
        for regex_pattern in pattern.patterns:
            if re.search(regex_pattern, query, re.IGNORECASE):
                score += 3.0
        
        # Exact phrase matching (higher weight)
        for keyword in pattern.keywords:
            if f" {keyword} " in f" {query} ":
                score += 1.5
        
        return score
    
    def _identify_domain_context(self, query: str) -> str:
        """Identify the domain context of the query"""
        domain_scores = {}
        
        for domain, domain_info in self.domain_patterns.items():
            score = 0
            for keyword in domain_info["keywords"]:
                if keyword in query:
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # Return domain with highest score
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _extract_found_keywords(self, query: str, pattern: IntentPattern) -> List[str]:
        """Extract keywords that were found in the query"""
        found_keywords = []
        for keyword in pattern.keywords:
            if keyword in query:
                found_keywords.append(keyword)
        return found_keywords
    
    def select_schema_by_intent(self, query: str, available_tables: List[str], 
                               full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Select schema based on query intent"""
        try:
            # Classify intent
            intent, confidence, context = self.classify_query_intent(query)
            
            if confidence < self.confidence_threshold:
                logger.warning(f"Low confidence in intent classification: {confidence}")
                # Fallback to basic schema
                return self._get_fallback_schema(available_tables, full_schema)
            
            logger.info(f"Query intent: {intent.value}, confidence: {confidence:.2f}")
            
            # Get domain-specific tables
            domain_tables = self._get_domain_tables(context["domain"], intent, available_tables)
            
            # Get intent-specific tables
            intent_tables = self._get_intent_tables(intent, available_tables)
            
            # Combine and deduplicate
            selected_tables = list(set(domain_tables + intent_tables))
            
            # Limit tables based on intent complexity
            max_tables = self._get_max_tables_for_intent(intent)
            if len(selected_tables) > max_tables:
                selected_tables = selected_tables[:max_tables]
            
            # Get schema for selected tables
            selected_schema = {}
            total_tokens = 0
            max_tokens = self._get_max_tokens_for_intent(intent)
            
            for table in selected_tables:
                if table in full_schema:
                    table_schema = full_schema[table]
                    table_tokens = self._estimate_tokens(json.dumps(table_schema))
                    
                    if total_tokens + table_tokens <= max_tokens:
                        selected_schema[table] = table_schema
                        total_tokens += table_tokens
                    else:
                        logger.info(f"Skipped table {table}: would exceed token limit")
            
            # Cache the result
            cache_key = self._generate_cache_key(query, selected_tables)
            self.intent_cache[cache_key] = {
                "intent": intent.value,
                "confidence": confidence,
                "selected_tables": selected_tables,
                "schema": selected_schema
            }
            
            return {
                "schema": selected_schema,
                "metadata": {
                    "intent": intent.value,
                    "confidence": confidence,
                    "domain": context["domain"],
                    "selected_tables": selected_tables,
                    "total_tokens": total_tokens,
                    "max_tokens": max_tokens,
                    "efficiency": f"{(total_tokens/max_tokens)*100:.1f}%",
                    "context": context
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to select schema by intent: {e}")
            return self._get_fallback_schema(available_tables, full_schema)
    
    def _get_domain_tables(self, domain: str, intent: QueryIntent, available_tables: List[str]) -> List[str]:
        """Get domain-specific tables based on intent"""
        if domain not in self.domain_patterns:
            return []
        
        domain_info = self.domain_patterns[domain]
        domain_tables = []
        
        # Add core tables
        domain_tables.extend(domain_info["core_tables"])
        
        # Add intent-specific tables
        if intent == QueryIntent.ANALYSIS:
            domain_tables.extend(domain_info.get("analysis_tables", []))
        elif intent == QueryIntent.RELATIONSHIP:
            domain_tables.extend(domain_info.get("relationship_tables", []))
        elif intent == QueryIntent.COMPARISON:
            domain_tables.extend(domain_info.get("comparison_tables", []))
        
        # Filter to available tables
        return [table for table in domain_tables if table in available_tables]
    
    def _get_intent_tables(self, intent: QueryIntent, available_tables: List[str]) -> List[str]:
        """Get tables based on query intent"""
        # This could be enhanced with a mapping of intents to table types
        # For now, return empty list and rely on domain tables
        return []
    
    def _get_max_tables_for_intent(self, intent: QueryIntent) -> int:
        """Get maximum number of tables for intent"""
        table_limits = {
            QueryIntent.COUNT: 3,
            QueryIntent.SEARCH: 5,
            QueryIntent.ANALYSIS: 8,
            QueryIntent.COMPARISON: 6,
            QueryIntent.AGGREGATION: 5,
            QueryIntent.RELATIONSHIP: 7,
            QueryIntent.QUALITY: 6,
            QueryIntent.DISCREPANCY: 10,
            QueryIntent.REPORTING: 8,
            QueryIntent.MAINTENANCE: 4
        }
        
        return table_limits.get(intent, 5)
    
    def _get_max_tokens_for_intent(self, intent: QueryIntent) -> int:
        """Get maximum tokens for intent"""
        token_limits = {
            QueryIntent.COUNT: 2000,
            QueryIntent.SEARCH: 3000,
            QueryIntent.ANALYSIS: 6000,
            QueryIntent.COMPARISON: 5000,
            QueryIntent.AGGREGATION: 4000,
            QueryIntent.RELATIONSHIP: 6000,
            QueryIntent.QUALITY: 5000,
            QueryIntent.DISCREPANCY: 8000,
            QueryIntent.REPORTING: 4000,
            QueryIntent.MAINTENANCE: 3000
        }
        
        return token_limits.get(intent, 4000)
    
    def _get_fallback_schema(self, available_tables: List[str], full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback schema when intent classification fails"""
        # Return basic schema for first few tables
        fallback_tables = available_tables[:3]
        fallback_schema = {}
        
        for table in fallback_tables:
            if table in full_schema:
                fallback_schema[table] = full_schema[table]
        
        return {
            "schema": fallback_schema,
            "metadata": {
                "intent": "fallback",
                "confidence": 0.0,
                "domain": "general",
                "selected_tables": fallback_tables,
                "total_tokens": self._estimate_tokens(json.dumps(fallback_schema)),
                "max_tokens": 3000,
                "efficiency": "100%",
                "fallback": True
            }
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _generate_cache_key(self, query: str, tables: List[str]) -> str:
        """Generate cache key for intent-based selection"""
        key_data = {
            "query": query.lower().strip()[:100],  # Truncate long queries
            "tables": sorted(tables)
        }
        return json.dumps(key_data, sort_keys=True)
    
    def get_intent_history(self) -> List[Dict[str, Any]]:
        """Get history of intent classifications"""
        history = []
        
        for cache_key, cache_data in self.intent_cache.items():
            try:
                key_data = json.loads(cache_key)
                history.append({
                    "query": key_data.get("query", ""),
                    "intent": cache_data["intent"],
                    "confidence": cache_data["confidence"],
                    "tables": cache_data["selected_tables"],
                    "timestamp": "now"  # You could add actual timestamps
                })
            except Exception as e:
                logger.warning(f"Failed to parse cache key: {e}")
        
        return history
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get statistics about intent classifications"""
        intent_counts = {}
        domain_counts = {}
        confidence_scores = []
        
        for cache_data in self.intent_cache.values():
            # Count intents
            intent = cache_data["intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            # Collect confidence scores
            confidence_scores.append(cache_data["confidence"])
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "total_queries": len(self.intent_cache),
            "intent_distribution": intent_counts,
            "average_confidence": avg_confidence,
            "cache_size": len(self.intent_cache)
        }
    
    def clear_cache(self):
        """Clear the intent cache"""
        self.intent_cache.clear()
        logger.info("Intent cache cleared")
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold for intent classification"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Confidence threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid confidence threshold: {new_threshold}. Must be between 0.0 and 1.0") 