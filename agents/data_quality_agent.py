import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent
from database.connection import db_connection

logger = logging.getLogger(__name__)

class DataQualityAgent(BaseAgent):
    """Agent responsible for data quality analysis and discrepancy detection"""
    
    def __init__(self):
        super().__init__("DataQuality")
        self.system_prompt = self._get_data_quality_system_prompt()
        self.quality_rules = self._initialize_quality_rules()
    
    def _get_data_quality_system_prompt(self) -> str:
        """Get the system prompt for the data quality agent"""
        return """You are a Data Quality Analysis Agent. Your role is to:

1. Analyze data for quality issues and discrepancies
2. Identify anomalies, outliers, and data inconsistencies
3. Apply business rules and validation logic
4. Generate data quality reports and recommendations
5. Suggest data cleansing and improvement strategies

You should:
- Think systematically about data quality dimensions
- Consider business context and rules
- Identify patterns and trends in data issues
- Provide actionable recommendations
- Use statistical methods when appropriate

Always respond with:
- Identified data quality issues
- Severity levels and impact assessment
- Root cause analysis
- Recommended actions
- Data quality metrics"""

    def _initialize_quality_rules(self) -> Dict[str, Any]:
        """Initialize default data quality rules"""
        return {
            "completeness": {
                "null_threshold": 0.1,  # 10% null values threshold
                "empty_string_threshold": 0.05
            },
            "accuracy": {
                "outlier_threshold": 3.0,  # Standard deviations for outlier detection
                "range_validation": True
            },
            "consistency": {
                "format_validation": True,
                "cross_reference_check": True
            },
            "timeliness": {
                "freshness_threshold_hours": 24
            }
        }
    
    def analyze_data_quality(self, table_name: str, business_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data quality for a specific table"""
        try:
            # Get table schema and sample data
            schema = db_connection.get_table_schema(table_name)
            sample_data = self._get_sample_data(table_name)
            
            # Merge business rules with default rules
            rules = self._merge_quality_rules(business_rules)
            
            # Perform quality analysis
            quality_analysis = {
                "table_name": table_name,
                "timestamp": self._get_current_timestamp(),
                "schema_analysis": self._analyze_schema_quality(schema),
                "data_analysis": self._analyze_data_quality(sample_data, rules),
                "business_rule_violations": self._check_business_rules(sample_data, business_rules),
                "overall_quality_score": 0.0,
                "recommendations": []
            }
            
            # Calculate overall quality score
            quality_analysis["overall_quality_score"] = self._calculate_quality_score(quality_analysis)
            
            # Generate recommendations
            quality_analysis["recommendations"] = self._generate_quality_recommendations(quality_analysis)
            
            # Store analysis in memory
            self.store_memory(f"quality_analysis_{table_name}", quality_analysis)
            
            return quality_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze data quality for table {table_name}: {e}")
            raise
    
    def _get_sample_data(self, table_name: str, sample_size: int = 1000) -> pd.DataFrame:
        """Get sample data from the table for analysis"""
        try:
            query = f"SELECT * FROM {table_name} WHERE ROWNUM <= {sample_size}"
            return db_connection.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to get sample data: {e}")
            return pd.DataFrame()
    
    def _analyze_schema_quality(self, schema: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the quality of table schema"""
        try:
            if schema.empty:
                return {"error": "No schema information available"}
            
            analysis = {
                "total_columns": len(schema),
                "nullable_columns": len(schema[schema['NULLABLE'] == 'Y']),
                "data_types": schema['DATA_TYPE'].value_counts().to_dict(),
                "column_lengths": {
                    "min": schema['DATA_LENGTH'].min(),
                    "max": schema['DATA_LENGTH'].max(),
                    "avg": schema['DATA_LENGTH'].mean()
                },
                "issues": []
            }
            
            # Check for potential issues
            if analysis["nullable_columns"] > analysis["total_columns"] * 0.8:
                analysis["issues"].append("Too many nullable columns may indicate data quality issues")
            
            if len(analysis["data_types"]) > 10:
                analysis["issues"].append("High variety of data types may indicate inconsistent data modeling")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Schema quality analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_data_quality(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of actual data"""
        try:
            if data.empty:
                return {"error": "No data available for analysis"}
            
            analysis = {
                "total_rows": len(data),
                "completeness": self._analyze_completeness(data, rules),
                "accuracy": self._analyze_accuracy(data, rules),
                "consistency": self._analyze_consistency(data, rules),
                "timeliness": self._analyze_timeliness(data, rules)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_completeness(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data completeness"""
        try:
            completeness = {}
            
            for column in data.columns:
                null_count = data[column].isnull().sum()
                null_percentage = (null_count / len(data)) * 100
                
                completeness[column] = {
                    "null_count": int(null_count),
                    "null_percentage": round(null_percentage, 2),
                    "completeness_score": round(100 - null_percentage, 2),
                    "is_acceptable": null_percentage <= rules["completeness"]["null_threshold"] * 100
                }
            
            return completeness
            
        except Exception as e:
            logger.error(f"Completeness analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_accuracy(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data accuracy and identify outliers"""
        try:
            accuracy = {}
            
            for column in data.columns:
                col_data = data[column]
                
                # Skip non-numeric columns for statistical analysis
                if not pd.api.types.is_numeric_dtype(col_data):
                    accuracy[column] = {"type": "non_numeric", "analysis": "Not applicable"}
                    continue
                
                # Remove null values for analysis
                clean_data = col_data.dropna()
                
                if len(clean_data) == 0:
                    accuracy[column] = {"type": "numeric", "all_null": True}
                    continue
                
                # Basic statistics
                mean_val = clean_data.mean()
                std_val = clean_data.std()
                
                # Outlier detection
                outlier_threshold = rules["accuracy"]["outlier_threshold"]
                outliers = clean_data[abs(clean_data - mean_val) > outlier_threshold * std_val]
                
                accuracy[column] = {
                    "type": "numeric",
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "min": float(clean_data.min()),
                    "max": float(clean_data.max()),
                    "outlier_count": len(outliers),
                    "outlier_percentage": round((len(outliers) / len(clean_data)) * 100, 2),
                    "has_outliers": len(outliers) > 0
                }
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Accuracy analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_consistency(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data consistency and format"""
        try:
            consistency = {}
            
            for column in data.columns:
                col_data = data[column]
                
                # Format consistency check
                if pd.api.types.is_string_dtype(col_data):
                    # Check for consistent formatting
                    non_null_data = col_data.dropna()
                    if len(non_null_data) > 0:
                        # Check for mixed case, special characters, etc.
                        mixed_case = any(str(val).isupper() != str(val).islower() for val in non_null_data if str(val).isalpha())
                        
                        consistency[column] = {
                            "type": "string",
                            "mixed_case": mixed_case,
                            "unique_values": len(non_null_data.unique()),
                            "most_common": non_null_data.value_counts().head(3).to_dict()
                        }
                else:
                    consistency[column] = {"type": "non_string", "analysis": "Not applicable"}
            
            return consistency
            
        except Exception as e:
            logger.error(f"Consistency analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_timeliness(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data timeliness and freshness"""
        try:
            timeliness = {}
            
            # Look for timestamp/date columns
            timestamp_columns = []
            for column in data.columns:
                if any(keyword in column.lower() for keyword in ['date', 'time', 'created', 'updated', 'modified']):
                    timestamp_columns.append(column)
            
            if not timestamp_columns:
                return {"message": "No timestamp columns found for timeliness analysis"}
            
            for column in timestamp_columns:
                col_data = data[column]
                non_null_data = col_data.dropna()
                
                if len(non_null_data) == 0:
                    timeliness[column] = {"all_null": True}
                    continue
                
                # Convert to datetime if possible
                try:
                    if pd.api.types.is_datetime64_any_dtype(non_null_data):
                        datetime_data = non_null_data
                    else:
                        datetime_data = pd.to_datetime(non_null_data, errors='coerce')
                    
                    if datetime_data.notna().any():
                        latest_date = datetime_data.max()
                        earliest_date = datetime_data.min()
                        current_date = pd.Timestamp.now()
                        
                        # Calculate age of latest data
                        data_age_hours = (current_date - latest_date).total_seconds() / 3600
                        
                        timeliness[column] = {
                            "latest_date": str(latest_date),
                            "earliest_date": str(earliest_date),
                            "data_age_hours": round(data_age_hours, 2),
                            "is_fresh": data_age_hours <= rules["timeliness"]["freshness_threshold_hours"],
                            "total_records": len(datetime_data)
                        }
                    else:
                        timeliness[column] = {"conversion_failed": True}
                        
                except Exception as e:
                    timeliness[column] = {"error": str(e)}
            
            return timeliness
            
        except Exception as e:
            logger.error(f"Timeliness analysis failed: {e}")
            return {"error": str(e)}
    
    def _check_business_rules(self, data: pd.DataFrame, business_rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check data against business rules"""
        try:
            violations = []
            
            if not business_rules:
                return violations
            
            for rule_name, rule_config in business_rules.items():
                try:
                    if rule_config.get("type") == "range_check":
                        column = rule_config.get("column")
                        min_val = rule_config.get("min")
                        max_val = rule_config.get("max")
                        
                        if column in data.columns and min_val is not None and max_val is not None:
                            violations_count = len(data[(data[column] < min_val) | (data[column] > max_val)])
                            if violations_count > 0:
                                violations.append({
                                    "rule_name": rule_name,
                                    "rule_type": "range_check",
                                    "column": column,
                                    "violations_count": violations_count,
                                    "expected_range": f"{min_val} to {max_val}"
                                })
                    
                    elif rule_config.get("type") == "uniqueness_check":
                        column = rule_config.get("column")
                        if column in data.columns:
                            total_rows = len(data)
                            unique_values = len(data[column].dropna().unique())
                            if total_rows != unique_values:
                                violations.append({
                                    "rule_name": rule_name,
                                    "rule_type": "uniqueness_check",
                                    "column": column,
                                    "total_rows": total_rows,
                                    "unique_values": unique_values,
                                    "duplicates": total_rows - unique_values
                                })
                    
                    elif rule_config.get("type") == "custom_sql":
                        sql_query = rule_config.get("sql")
                        if sql_query:
                            try:
                                result = db_connection.execute_query(sql_query)
                                if len(result) > 0:
                                    violations.append({
                                        "rule_name": rule_name,
                                        "rule_type": "custom_sql",
                                        "sql": sql_query,
                                        "violations_found": len(result)
                                    })
                            except Exception as e:
                                violations.append({
                                    "rule_name": rule_name,
                                    "rule_type": "custom_sql",
                                    "sql": sql_query,
                                    "error": str(e)
                                })
                
                except Exception as e:
                    logger.error(f"Business rule check failed for {rule_name}: {e}")
                    violations.append({
                        "rule_name": rule_name,
                        "error": str(e)
                    })
            
            return violations
            
        except Exception as e:
            logger.error(f"Business rule checking failed: {e}")
            return [{"error": str(e)}]
    
    def _merge_quality_rules(self, business_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Merge business rules with default quality rules"""
        rules = self.quality_rules.copy()
        
        if business_rules:
            # Override default rules with business rules
            for category, category_rules in business_rules.items():
                if category in rules:
                    rules[category].update(category_rules)
                else:
                    rules[category] = category_rules
        
        return rules
    
    def _calculate_quality_score(self, quality_analysis: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        try:
            scores = []
            
            # Schema quality score
            if "schema_analysis" in quality_analysis and "issues" in quality_analysis["schema_analysis"]:
                schema_score = max(0, 100 - len(quality_analysis["schema_analysis"]["issues"]) * 10)
                scores.append(schema_score)
            
            # Data quality scores
            if "data_analysis" in quality_analysis:
                data_analysis = quality_analysis["data_analysis"]
                
                # Completeness score
                if "completeness" in data_analysis and isinstance(data_analysis["completeness"], dict):
                    completeness_scores = [col_data.get("completeness_score", 0) for col_data in data_analysis["completeness"].values() if isinstance(col_data, dict)]
                    if completeness_scores:
                        scores.append(sum(completeness_scores) / len(completeness_scores))
                
                # Accuracy score
                if "accuracy" in data_analysis and isinstance(data_analysis["accuracy"], dict):
                    accuracy_scores = []
                    for col_data in data_analysis["accuracy"].values():
                        if isinstance(col_data, dict) and "outlier_percentage" in col_data:
                            accuracy_score = max(0, 100 - col_data["outlier_percentage"])
                            accuracy_scores.append(accuracy_score)
                    if accuracy_scores:
                        scores.append(sum(accuracy_scores) / len(accuracy_scores))
            
            # Business rule violations score
            if "business_rule_violations" in quality_analysis:
                violations = quality_analysis["business_rule_violations"]
                if violations and not any("error" in v for v in violations):
                    rule_score = max(0, 100 - len(violations) * 20)
                    scores.append(rule_score)
                else:
                    scores.append(100)  # No violations or errors
            
            if scores:
                return round(sum(scores) / len(scores), 2)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _generate_quality_recommendations(self, quality_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality analysis"""
        try:
            recommendations = []
            
            # Schema recommendations
            if "schema_analysis" in quality_analysis and "issues" in quality_analysis["schema_analysis"]:
                for issue in quality_analysis["schema_analysis"]["issues"]:
                    recommendations.append(f"Schema: {issue}")
            
            # Data quality recommendations
            if "data_analysis" in quality_analysis:
                data_analysis = quality_analysis["data_analysis"]
                
                # Completeness recommendations
                if "completeness" in data_analysis and isinstance(data_analysis["completeness"], dict):
                    for column, col_data in data_analysis["completeness"].items():
                        if isinstance(col_data, dict) and not col_data.get("is_acceptable", True):
                            recommendations.append(f"Completeness: Column '{column}' has {col_data['null_percentage']}% null values - consider data validation or default values")
                
                # Accuracy recommendations
                if "accuracy" in data_analysis and isinstance(data_analysis["accuracy"], dict):
                    for column, col_data in data_analysis["accuracy"].items():
                        if isinstance(col_data, dict) and col_data.get("has_outliers", False):
                            recommendations.append(f"Accuracy: Column '{column}' has {col_data['outlier_percentage']}% outliers - investigate for data entry errors")
            
            # Business rule recommendations
            if "business_rule_violations" in quality_analysis:
                violations = quality_analysis["business_rule_violations"]
                for violation in violations:
                    if "error" not in violation:
                        recommendations.append(f"Business Rule: {violation.get('rule_name', 'Unknown')} - {violation.get('violations_count', 0)} violations found")
            
            # Overall recommendations
            overall_score = quality_analysis.get("overall_quality_score", 0)
            if overall_score < 70:
                recommendations.append("Overall: Data quality score is low - implement comprehensive data quality monitoring")
            elif overall_score < 90:
                recommendations.append("Overall: Data quality score is moderate - focus on identified issues")
            else:
                recommendations.append("Overall: Data quality score is good - maintain current standards")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return [f"Error generating recommendations: {e}"]
    
    def detect_discrepancies(self, table_name: str, comparison_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Detect specific discrepancies based on comparison criteria"""
        try:
            # This method will be expanded based on your business logic requirements
            # For now, it provides a framework for discrepancy detection
            
            discrepancy_analysis = {
                "table_name": table_name,
                "timestamp": self._get_current_timestamp(),
                "comparison_criteria": comparison_criteria,
                "discrepancies_found": [],
                "summary": {}
            }
            
            # Store for future business logic implementation
            self.store_memory(f"discrepancy_analysis_{table_name}", discrepancy_analysis)
            
            return discrepancy_analysis
            
        except Exception as e:
            logger.error(f"Discrepancy detection failed: {e}")
            raise
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return data quality analysis"""
        table_name = input_data.get("table_name", "")
        business_rules = input_data.get("business_rules", {})
        
        try:
            # Perform data quality analysis
            quality_analysis = self.analyze_data_quality(table_name, business_rules)
            
            return quality_analysis
            
        except Exception as e:
            logger.error(f"Data quality processing failed: {e}")
            return {
                "error": str(e),
                "table_name": table_name,
                "timestamp": self._get_current_timestamp()
            }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_quality_history(self) -> List[Dict[str, Any]]:
        """Get history of recent quality analyses"""
        memory_summary = self.get_memory_summary()
        analyses = []
        
        for key, data in memory_summary.items():
            if any(prefix in key for prefix in ["quality_analysis_", "discrepancy_analysis_"]):
                analyses.append({
                    "key": key,
                    "data": data.get("value"),
                    "timestamp": data.get("timestamp")
                })
        
        return sorted(analyses, key=lambda x: x["timestamp"], reverse=True) 