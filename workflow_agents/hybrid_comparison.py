import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Result of data comparison"""
    comparison_type: str
    base_data: pd.DataFrame
    comparison_data: Optional[pd.DataFrame] = None
    discrepancies: List[Dict[str, Any]] = []
    statistics: Dict[str, Any] = None
    insights: List[str] = []
    confidence_score: float = 0.0

@dataclass
class Discrepancy:
    """Individual discrepancy found"""
    type: str
    severity: str
    description: str
    affected_records: int
    affected_columns: List[str]
    business_impact: str
    suggested_action: str
    confidence: float

class HybridComparisonEngine:
    """Hybrid comparison engine combining SQL, Python, and LLM"""
    
    def __init__(self):
        self.bedrock_client = self._initialize_bedrock()
        self.comparison_cache = {}
        
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
    
    def compare_data(self, base_data: pd.DataFrame, comparison_data: Optional[pd.DataFrame] = None,
                    comparison_type: str = "auto", comparison_params: Dict[str, Any] = None) -> ComparisonResult:
        """Main method to compare data using hybrid approach"""
        try:
            logger.info(f"Starting data comparison: {comparison_type}")
            
            # Determine comparison type if auto
            if comparison_type == "auto":
                comparison_type = self._determine_comparison_type(base_data)
            
            # Execute comparison based on type
            if comparison_type == "time_based":
                result = self._time_based_comparison(base_data, comparison_params)
            elif comparison_type == "category_based":
                result = self._category_based_comparison(base_data, comparison_params)
            elif comparison_type == "dataset_comparison":
                if comparison_data is not None:
                    result = self._dataset_comparison(base_data, comparison_data)
                else:
                    raise ValueError("Comparison data required for dataset comparison")
            else:
                result = self._general_analysis(base_data)
            
            # Use LLM to analyze results and generate insights
            result.insights = self._generate_insights_with_llm(result)
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence_score(result)
            
            logger.info(f"Comparison completed: {len(result.discrepancies)} discrepancies found")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compare data: {e}")
            return self._create_error_result(base_data, str(e))
    
    def _determine_comparison_type(self, data: pd.DataFrame) -> str:
        """Automatically determine the best comparison type"""
        try:
            # Check for time columns
            time_columns = self._find_time_columns(data)
            if time_columns:
                return "time_based"
            
            # Check for category columns
            category_columns = self._find_category_columns(data)
            if category_columns:
                return "category_based"
            
            # Default to general analysis
            return "general_analysis"
            
        except Exception as e:
            logger.warning(f"Failed to determine comparison type: {e}")
            return "general_analysis"
    
    def _find_time_columns(self, data: pd.DataFrame) -> List[str]:
        """Find columns that contain time/date data"""
        time_columns = []
        
        for col in data.columns:
            try:
                # Check if column name suggests time
                if any(time_word in col.lower() for time_word in ['date', 'time', 'created', 'updated', 'timestamp']):
                    time_columns.append(col)
                    continue
                
                # Check if column data looks like time
                sample_values = data[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Try to parse as datetime
                    pd.to_datetime(sample_values, errors='coerce')
                    if sample_values.notna().sum() > len(sample_values) * 0.8:  # 80% success rate
                        time_columns.append(col)
                        
            except Exception:
                continue
        
        return time_columns
    
    def _find_category_columns(self, data: pd.DataFrame) -> List[str]:
        """Find columns that contain categorical data"""
        category_columns = []
        
        for col in data.columns:
            try:
                # Check if column name suggests category
                if any(cat_word in col.lower() for cat_word in ['category', 'type', 'status', 'group', 'class']):
                    category_columns.append(col)
                    continue
                
                # Check if column data looks categorical
                unique_ratio = data[col].nunique() / len(data[col])
                if unique_ratio < 0.1:  # Less than 10% unique values
                    category_columns.append(col)
                    
            except Exception:
                continue
        
        return category_columns
    
    def _time_based_comparison(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> ComparisonResult:
        """Perform time-based comparison (YoY, period-over-period)"""
        try:
            if params is None:
                params = {}
            
            # Find time column
            time_column = params.get('time_column')
            if not time_column:
                time_columns = self._find_time_columns(data)
                if not time_columns:
                    raise ValueError("No time column found for time-based comparison")
                time_column = time_columns[0]
            
            # Convert time column to datetime
            data_copy = data.copy()
            data_copy[time_column] = pd.to_datetime(data_copy[time_column], errors='coerce')
            
            # Remove rows with invalid dates
            data_copy = data_copy.dropna(subset=[time_column])
            
            if len(data_copy) == 0:
                raise ValueError("No valid time data found")
            
            # Determine comparison periods
            comparison_days = params.get('comparison_days', 30)
            current_period_end = data_copy[time_column].max()
            current_period_start = current_period_end - timedelta(days=comparison_days)
            previous_period_end = current_period_start
            previous_period_start = previous_period_end - timedelta(days=comparison_days)
            
            # Split data into periods
            current_data = data_copy[
                (data_copy[time_column] >= current_period_start) & 
                (data_copy[time_column] <= current_period_end)
            ]
            
            previous_data = data_copy[
                (data_copy[time_column] >= previous_period_start) & 
                (data_copy[time_column] <= previous_period_end)
            ]
            
            # Calculate statistics
            current_stats = self._calculate_period_statistics(current_data, time_column)
            previous_stats = self._calculate_period_statistics(previous_data, time_column)
            
            # Find discrepancies
            discrepancies = self._find_time_based_discrepancies(current_stats, previous_stats, params)
            
            return ComparisonResult(
                comparison_type="time_based",
                base_data=current_data,
                comparison_data=previous_data,
                discrepancies=discrepancies,
                statistics={
                    "current_period": current_stats,
                    "previous_period": previous_stats,
                    "comparison_params": params
                }
            )
            
        except Exception as e:
            logger.error(f"Time-based comparison failed: {e}")
            return self._create_error_result(data, f"Time-based comparison failed: {e}")
    
    def _category_based_comparison(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> ComparisonResult:
        """Perform category-based comparison"""
        try:
            if params is None:
                params = {}
            
            # Find category column
            category_column = params.get('category_column')
            if not category_column:
                category_columns = self._find_category_columns(data)
                if not category_columns:
                    raise ValueError("No category column found for category-based comparison")
                category_column = category_columns[0]
            
            # Group data by category
            grouped_data = data.groupby(category_column)
            
            # Calculate statistics for each category
            category_stats = {}
            for category, group_data in grouped_data:
                category_stats[category] = self._calculate_category_statistics(group_data, category_column)
            
            # Find discrepancies between categories
            discrepancies = self._find_category_based_discrepancies(category_stats, params)
            
            return ComparisonResult(
                comparison_type="category_based",
                base_data=data,
                comparison_data=None,
                discrepancies=discrepancies,
                statistics={
                    "category_statistics": category_stats,
                    "comparison_params": params
                }
            )
            
        except Exception as e:
            logger.error(f"Category-based comparison failed: {e}")
            return self._create_error_result(data, f"Category-based comparison failed: {e}")
    
    def _dataset_comparison(self, base_data: pd.DataFrame, comparison_data: pd.DataFrame) -> ComparisonResult:
        """Compare two different datasets"""
        try:
            # Align datasets
            aligned_data = self._align_datasets(base_data, comparison_data)
            
            # Find discrepancies
            discrepancies = self._find_dataset_discrepancies(
                aligned_data['base'], 
                aligned_data['comparison']
            )
            
            # Calculate comparison statistics
            comparison_stats = self._calculate_dataset_comparison_statistics(
                aligned_data['base'], 
                aligned_data['comparison']
            )
            
            return ComparisonResult(
                comparison_type="dataset_comparison",
                base_data=aligned_data['base'],
                comparison_data=aligned_data['comparison'],
                discrepancies=discrepancies,
                statistics=comparison_stats
            )
            
        except Exception as e:
            logger.error(f"Dataset comparison failed: {e}")
            return self._create_error_result(base_data, f"Dataset comparison failed: {e}")
    
    def _general_analysis(self, data: pd.DataFrame) -> ComparisonResult:
        """Perform general data analysis and quality checks"""
        try:
            # Basic data quality checks
            quality_issues = self._check_data_quality(data)
            
            # Statistical analysis
            statistics = self._calculate_general_statistics(data)
            
            # Find anomalies
            anomalies = self._find_data_anomalies(data)
            
            # Combine all issues
            all_issues = quality_issues + anomalies
            
            return ComparisonResult(
                comparison_type="general_analysis",
                base_data=data,
                comparison_data=None,
                discrepancies=all_issues,
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"General analysis failed: {e}")
            return self._create_error_result(data, f"General analysis failed: {e}")
    
    def _calculate_period_statistics(self, data: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """Calculate statistics for a time period"""
        stats = {
            "record_count": len(data),
            "time_range": {
                "start": data[time_column].min(),
                "end": data[time_column].max()
            }
        }
        
        # Calculate numeric column statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            stats[col] = {
                "mean": data[col].mean(),
                "median": data[col].median(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
                "sum": data[col].sum()
            }
        
        return stats
    
    def _calculate_category_statistics(self, data: pd.DataFrame, category_column: str) -> Dict[str, Any]:
        """Calculate statistics for a category"""
        stats = {
            "record_count": len(data),
            "category": data[category_column].iloc[0] if len(data) > 0 else None
        }
        
        # Calculate numeric column statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != category_column:
                stats[col] = {
                    "mean": data[col].mean(),
                    "median": data[col].median(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "sum": data[col].sum()
                }
        
        return stats
    
    def _align_datasets(self, base_data: pd.DataFrame, comparison_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Align two datasets for comparison"""
        # Find common columns
        common_columns = list(set(base_data.columns) & set(comparison_data.columns))
        
        # Align data types
        aligned_base = base_data[common_columns].copy()
        aligned_comparison = comparison_data[common_columns].copy()
        
        # Convert data types to match
        for col in common_columns:
            if aligned_base[col].dtype != aligned_comparison[col].dtype:
                try:
                    # Try to convert to a common type
                    if aligned_base[col].dtype in ['object', 'string']:
                        aligned_comparison[col] = aligned_comparison[col].astype(str)
                    elif aligned_comparison[col].dtype in ['object', 'string']:
                        aligned_base[col] = aligned_base[col].astype(str)
                except Exception:
                    logger.warning(f"Could not align data types for column {col}")
        
        return {
            "base": aligned_base,
            "comparison": aligned_comparison
        }
    
    def _find_time_based_discrepancies(self, current_stats: Dict[str, Any], 
                                     previous_stats: Dict[str, Any], 
                                     params: Dict[str, Any]) -> List[Discrepancy]:
        """Find discrepancies in time-based comparison"""
        discrepancies = []
        threshold = params.get('threshold', 0.1)  # 10% default threshold
        
        for col, current_col_stats in current_stats.items():
            if col in ['record_count', 'time_range']:
                continue
            
            if isinstance(current_col_stats, dict) and 'mean' in current_col_stats:
                current_mean = current_col_stats['mean']
                previous_mean = previous_stats.get(col, {}).get('mean')
                
                if previous_mean is not None and previous_mean != 0:
                    change_ratio = abs(current_mean - previous_mean) / abs(previous_mean)
                    
                    if change_ratio > threshold:
                        discrepancies.append(Discrepancy(
                            type="time_based_change",
                            severity="medium" if change_ratio < 0.5 else "high",
                            description=f"Significant change in {col}: {change_ratio:.2%} change",
                            affected_records=current_stats['record_count'],
                            affected_columns=[col],
                            business_impact="Period-over-period variation detected",
                            suggested_action="Investigate cause of change",
                            confidence=min(0.9, change_ratio)
                        ))
        
        return discrepancies
    
    def _find_category_based_discrepancies(self, category_stats: Dict[str, Any], 
                                        params: Dict[str, Any]) -> List[Discrepancy]:
        """Find discrepancies in category-based comparison"""
        discrepancies = []
        threshold = params.get('threshold', 0.2)  # 20% default threshold
        
        # Find the category with the most records as baseline
        baseline_category = max(category_stats.keys(), 
                              key=lambda x: category_stats[x]['record_count'])
        baseline_stats = category_stats[baseline_category]
        
        for category, stats in category_stats.items():
            if category == baseline_category:
                continue
            
            for col, col_stats in stats.items():
                if col in ['record_count', 'category']:
                    continue
                
                if isinstance(col_stats, dict) and 'mean' in col_stats:
                    baseline_mean = baseline_stats.get(col, {}).get('mean')
                    category_mean = col_stats['mean']
                    
                    if baseline_mean is not None and baseline_mean != 0:
                        difference_ratio = abs(category_mean - baseline_mean) / abs(baseline_mean)
                        
                        if difference_ratio > threshold:
                            discrepancies.append(Discrepancy(
                                type="category_variation",
                                severity="medium" if difference_ratio < 0.5 else "high",
                                description=f"Category {category} differs from baseline in {col}: {difference_ratio:.2%} difference",
                                affected_records=stats['record_count'],
                                affected_columns=[col],
                                business_impact="Category-specific variation detected",
                                suggested_action="Investigate category differences",
                                confidence=min(0.9, difference_ratio)
                            ))
        
        return discrepancies
    
    def _find_dataset_discrepancies(self, base_data: pd.DataFrame, 
                                  comparison_data: pd.DataFrame) -> List[Discrepancy]:
        """Find discrepancies between two datasets"""
        discrepancies = []
        
        # Check for missing columns
        missing_columns = set(base_data.columns) - set(comparison_data.columns)
        if missing_columns:
            discrepancies.append(Discrepancy(
                type="missing_columns",
                severity="high",
                description=f"Missing columns in comparison data: {missing_columns}",
                affected_records=len(comparison_data),
                affected_columns=list(missing_columns),
                business_impact="Data structure mismatch",
                suggested_action="Align data schemas",
                confidence=1.0
            ))
        
        # Check for data type mismatches
        for col in set(base_data.columns) & set(comparison_data.columns):
            if base_data[col].dtype != comparison_data[col].dtype:
                discrepancies.append(Discrepancy(
                    type="data_type_mismatch",
                    severity="medium",
                    description=f"Data type mismatch in column {col}: {base_data[col].dtype} vs {comparison_data[col].dtype}",
                    affected_records=len(comparison_data),
                    affected_columns=[col],
                    business_impact="Data type inconsistency",
                    suggested_action="Standardize data types",
                    confidence=0.9
                ))
        
        return discrepancies
    
    def _check_data_quality(self, data: pd.DataFrame) -> List[Discrepancy]:
        """Check basic data quality issues"""
        issues = []
        
        # Check for missing values
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_ratio = missing_count / len(data)
            
            if missing_ratio > 0.1:  # More than 10% missing
                issues.append(Discrepancy(
                    type="missing_data",
                    severity="medium" if missing_ratio < 0.3 else "high",
                    description=f"High missing data in {col}: {missing_ratio:.2%}",
                    affected_records=missing_count,
                    affected_columns=[col],
                    business_impact="Data completeness issues",
                    suggested_action="Investigate missing data causes",
                    confidence=0.9
                ))
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues.append(Discrepancy(
                type="duplicate_records",
                severity="medium",
                description=f"Duplicate records found: {duplicate_count}",
                affected_records=duplicate_count,
                affected_columns=data.columns.tolist(),
                business_impact="Data integrity issues",
                suggested_action="Remove or investigate duplicates",
                confidence=0.9
            ))
        
        return issues
    
    def _find_data_anomalies(self, data: pd.DataFrame) -> List[Discrepancy]:
        """Find statistical anomalies in data"""
        anomalies = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Calculate z-scores
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outliers = z_scores > 3  # 3 standard deviations
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                anomalies.append(Discrepancy(
                    type="statistical_outlier",
                    severity="low" if outlier_count < len(data) * 0.01 else "medium",
                    description=f"Statistical outliers in {col}: {outlier_count} records",
                    affected_records=outlier_count,
                    affected_columns=[col],
                    business_impact="Potential data quality issues",
                    suggested_action="Review outlier records",
                    confidence=0.8
                ))
        
        return anomalies
    
    def _calculate_dataset_comparison_statistics(self, base_data: pd.DataFrame, 
                                              comparison_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for dataset comparison"""
        return {
            "base_dataset": {
                "record_count": len(base_data),
                "columns": list(base_data.columns),
                "data_types": {col: str(dtype) for col, dtype in base_data.dtypes.items()}
            },
            "comparison_dataset": {
                "record_count": len(comparison_data),
                "columns": list(comparison_data.columns),
                "data_types": {col: str(dtype) for col, dtype in comparison_data.dtypes.items()}
            },
            "alignment": {
                "common_columns": list(set(base_data.columns) & set(comparison_data.columns)),
                "base_only_columns": list(set(base_data.columns) - set(comparison_data.columns)),
                "comparison_only_columns": list(set(comparison_data.columns) - set(base_data.columns))
            }
        }
    
    def _calculate_general_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate general statistics for data"""
        return {
            "record_count": len(data),
            "column_count": len(data.columns),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "missing_data_summary": data.isnull().sum().to_dict()
        }
    
    def _generate_insights_with_llm(self, result: ComparisonResult) -> List[str]:
        """Use LLM to generate insights from comparison results"""
        try:
            system_prompt = """You are an expert data analyst. Analyze the comparison results and provide insights.
            Return a JSON array of insight strings."""
            
            # Prepare data summary for LLM
            data_summary = {
                "comparison_type": result.comparison_type,
                "discrepancy_count": len(result.discrepancies),
                "discrepancy_types": list(set([d.type for d in result.discrepancies])),
                "severity_distribution": {
                    severity: len([d for d in result.discrepancies if d.severity == severity])
                    for severity in set([d.severity for d in result.discrepancies])
                },
                "statistics_summary": str(result.statistics)[:500] if result.statistics else "No statistics"
            }
            
            prompt = f"""
            Analyze these data comparison results and provide business insights:
            
            {json.dumps(data_summary, indent=2)}
            
            Provide 3-5 insights in JSON array format, focusing on:
            1. Business implications
            2. Data quality observations
            3. Recommended actions
            4. Risk assessment
            5. Opportunities for improvement
            """
            
            response = self._invoke_bedrock(prompt, system_prompt)
            
            try:
                insights = json.loads(response)
                if isinstance(insights, list):
                    return insights
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bedrock response for insights")
        
        except Exception as e:
            logger.error(f"Failed to generate insights with LLM: {e}")
        
        # Fallback insights
        return [
            f"Found {len(result.discrepancies)} discrepancies in {result.comparison_type} analysis",
            "Review discrepancies for business impact assessment",
            "Consider data quality improvements based on findings"
        ]
    
    def _calculate_confidence_score(self, result: ComparisonResult) -> float:
        """Calculate confidence score for the comparison result"""
        if not result.discrepancies:
            return 0.9  # High confidence if no issues found
        
        # Calculate confidence based on discrepancy quality
        total_confidence = sum(d.confidence for d in result.discrepancies)
        avg_confidence = total_confidence / len(result.discrepancies)
        
        # Adjust based on data quality
        if result.statistics and 'record_count' in result.statistics:
            record_count = result.statistics.get('record_count', 0)
            if record_count > 1000:
                data_quality_factor = 1.0
            elif record_count > 100:
                data_quality_factor = 0.9
            else:
                data_quality_factor = 0.8
        else:
            data_quality_factor = 0.8
        
        return min(0.95, avg_confidence * data_quality_factor)
    
    def _create_error_result(self, data: pd.DataFrame, error_message: str) -> ComparisonResult:
        """Create error result when comparison fails"""
        return ComparisonResult(
            comparison_type="error",
            base_data=data,
            discrepancies=[Discrepancy(
                type="comparison_error",
                severity="high",
                description=f"Comparison failed: {error_message}",
                affected_records=len(data),
                affected_columns=data.columns.tolist(),
                business_impact="Unable to perform comparison",
                suggested_action="Check data and retry",
                confidence=0.0
            )],
            statistics={"error": error_message}
        )
    
    def clear_cache(self):
        """Clear the comparison cache"""
        self.comparison_cache.clear()
        logger.info("Comparison cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.comparison_cache),
            "cache_keys": list(self.comparison_cache.keys())
        } 