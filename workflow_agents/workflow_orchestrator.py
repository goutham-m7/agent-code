import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from .use_case_analyzer import UseCaseAnalyzer, UseCaseAnalysis
from .sql_generator import SQLGenerator, SQLQuery
from .hybrid_comparison import HybridComparisonEngine, ComparisonResult
from database.connection import db_connection
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class WorkflowStep:
    """Represents a step in the workflow"""
    step_name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class WorkflowResult:
    """Complete workflow result"""
    workflow_id: str
    user_query: str
    steps: List[WorkflowStep]
    final_result: Dict[str, Any]
    total_duration: float
    status: str
    discrepancies_found: int
    confidence_score: float

class WorkflowOrchestrator:
    """Main orchestrator for the complete workflow"""
    
    def __init__(self):
        self.use_case_analyzer = UseCaseAnalyzer()
        self.sql_generator = SQLGenerator()
        self.comparison_engine = HybridComparisonEngine()
        self.workflow_history = {}
        self.current_workflow = None
        
    def execute_workflow(self, user_query: str, problem_statement: str, 
                        database_schema: Dict[str, Any], 
                        data_analyst_verification: bool = True) -> WorkflowResult:
        """Execute the complete workflow as described"""
        try:
            workflow_id = f"workflow_{int(time.time())}"
            logger.info(f"Starting workflow {workflow_id}: {user_query[:100]}...")
            
            # Initialize workflow
            self.current_workflow = {
                "id": workflow_id,
                "user_query": user_query,
                "problem_statement": problem_statement,
                "database_schema": database_schema,
                "steps": [],
                "start_time": datetime.now()
            }
            
            # Step 1: Analyze use case and create context
            step1 = self._execute_step_1_use_case_analysis(problem_statement, database_schema)
            self.current_workflow["steps"].append(step1)
            
            if step1.status == "failed":
                return self._create_failed_workflow_result(workflow_id, user_query, step1.error)
            
            use_case_analysis = step1.result["analysis"]
            use_case_context = step1.result["context"]
            
            # Step 2: Data Analyst Verification (one-time)
            if data_analyst_verification:
                step2 = self._execute_step_2_analyst_verification(use_case_analysis)
                self.current_workflow["steps"].append(step2)
                
                if step2.status == "failed":
                    return self._create_failed_workflow_result(workflow_id, user_query, step2.error)
            
            # Step 3: Extract entities from user query
            step3 = self._execute_step_3_entity_extraction(user_query, use_case_context)
            self.current_workflow["steps"].append(step3)
            
            if step3.status == "failed":
                return self._create_failed_workflow_result(workflow_id, user_query, step3.error)
            
            entities = step3.result["entities"]
            
            # Step 4: Generate SQL using context and memory
            step4 = self._execute_step_4_sql_generation(user_query, database_schema, use_case_context, entities)
            self.current_workflow["steps"].append(step4)
            
            if step4.status == "failed":
                return self._create_failed_workflow_result(workflow_id, user_query, step4.error)
            
            sql_query = step4.result["sql_query"]
            
            # Step 5: Execute SQL and retrieve data
            step5 = self._execute_step_5_data_retrieval(sql_query)
            self.current_workflow["steps"].append(step5)
            
            if step5.status == "failed":
                return self._create_failed_workflow_result(workflow_id, user_query, step5.error)
            
            retrieved_data = step5.result["data"]
            
            # Step 6: Hybrid comparison and discrepancy detection
            step6 = self._execute_step_6_hybrid_comparison(retrieved_data, use_case_context)
            self.current_workflow["steps"].append(step6)
            
            if step6.status == "failed":
                return self._create_failed_workflow_result(workflow_id, user_query, step6.error)
            
            comparison_result = step6.result["comparison_result"]
            
            # Step 7: Generate final report
            step7 = self._execute_step_7_report_generation(
                user_query, use_case_analysis, sql_query, comparison_result
            )
            self.current_workflow["steps"].append(step7)
            
            # Calculate total duration
            total_duration = (datetime.now() - self.current_workflow["start_time"]).total_seconds()
            
            # Create final result
            workflow_result = WorkflowResult(
                workflow_id=workflow_id,
                user_query=user_query,
                steps=self.current_workflow["steps"],
                final_result=step7.result,
                total_duration=total_duration,
                status="completed",
                discrepancies_found=len(comparison_result.discrepancies),
                confidence_score=comparison_result.confidence_score
            )
            
            # Store in history
            self.workflow_history[workflow_id] = workflow_result
            
            logger.info(f"Workflow {workflow_id} completed successfully in {total_duration:.2f}s")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return self._create_failed_workflow_result(workflow_id, user_query, str(e))
    
    def _execute_step_1_use_case_analysis(self, problem_statement: str, 
                                         database_schema: Dict[str, Any]) -> WorkflowStep:
        """Step 1: Analyze use case and create context"""
        step = WorkflowStep(
            step_name="Use Case Analysis",
            status="running",
            start_time=datetime.now()
        )
        
        try:
            logger.info("Executing Step 1: Use Case Analysis")
            
            # Analyze use case
            use_case_analysis = self.use_case_analyzer.analyze_use_case(
                problem_statement, database_schema
            )
            
            # Create context for LLM
            use_case_context = self.use_case_analyzer.get_analysis_for_llm_context(use_case_analysis)
            
            step.status = "completed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = {
                "analysis": use_case_analysis,
                "context": use_case_context
            }
            
            logger.info(f"Step 1 completed: {step.duration:.2f}s")
            return step
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.error = str(e)
            logger.error(f"Step 1 failed: {e}")
            return step
    
    def _execute_step_2_analyst_verification(self, use_case_analysis: UseCaseAnalysis) -> WorkflowStep:
        """Step 2: Data Analyst Verification (one-time)"""
        step = WorkflowStep(
            step_name="Data Analyst Verification",
            status="running",
            start_time=datetime.now()
        )
        
        try:
            logger.info("Executing Step 2: Data Analyst Verification")
            
            # In a real implementation, this would involve human interaction
            # For now, we'll simulate the verification process
            
            verification_result = {
                "verified": True,
                "verified_by": "Data Analyst",
                "verification_date": datetime.now().isoformat(),
                "comments": "Use case analysis verified and approved",
                "recommendations": [
                    "Proceed with workflow execution",
                    "Monitor data quality during comparison",
                    "Validate business rules implementation"
                ]
            }
            
            step.status = "completed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = verification_result
            
            logger.info(f"Step 2 completed: {step.duration:.2f}s")
            return step
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.error = str(e)
            logger.error(f"Step 2 failed: {e}")
            return step
    
    def _execute_step_3_entity_extraction(self, user_query: str, 
                                        use_case_context: str) -> WorkflowStep:
        """Step 3: Extract entities from user query"""
        step = WorkflowStep(
            step_name="Entity Extraction",
            status="running",
            start_time=datetime.now()
        )
        
        try:
            logger.info("Executing Step 3: Entity Extraction")
            
            # Use the schema chunking manager for entity extraction
            from schema_managers.schema_chunking import SchemaChunkingManager
            
            entity_extractor = SchemaChunkingManager()
            entities = entity_extractor.extract_entities_from_query(user_query)
            
            step.status = "completed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = {
                "entities": entities,
                "extraction_method": "Bedrock-powered entity extraction"
            }
            
            logger.info(f"Step 3 completed: {step.duration:.2f}s, extracted {len(entities)} entities")
            return step
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.error = str(e)
            logger.error(f"Step 3 failed: {e}")
            return step
    
    def _execute_step_4_sql_generation(self, user_query: str, database_schema: Dict[str, Any],
                                      use_case_context: str, entities: List[str]) -> WorkflowStep:
        """Step 4: Generate SQL using context and memory"""
        step = WorkflowStep(
            step_name="SQL Generation",
            status="running",
            start_time=datetime.now()
        )
        
        try:
            logger.info("Executing Step 4: SQL Generation")
            
            # Generate SQL using the context
            sql_query = self.sql_generator.generate_sql(
                user_query=user_query,
                available_schema=database_schema,
                use_case_context=use_case_context,
                comparison_strategy="auto"
            )
            
            # Validate SQL
            validation_result = self.sql_generator.validate_sql(sql_query.sql)
            
            step.status = "completed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = {
                "sql_query": sql_query,
                "validation": validation_result,
                "generation_method": "Bedrock-powered SQL generation"
            }
            
            logger.info(f"Step 4 completed: {step.duration:.2f}s, SQL type: {sql_query.query_type}")
            return step
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.error = str(e)
            logger.error(f"Step 4 failed: {e}")
            return step
    
    def _execute_step_5_data_retrieval(self, sql_query: SQLQuery) -> WorkflowStep:
        """Step 5: Execute SQL and retrieve data"""
        step = WorkflowStep(
            step_name="Data Retrieval",
            status="running",
            start_time=datetime.now()
        )
        
        try:
            logger.info("Executing Step 5: Data Retrieval")
            
            # Execute SQL query
            data = db_connection.execute_query(sql_query.sql)
            
            # Basic data validation
            if data.empty:
                logger.warning("SQL query returned no data")
            
            step.status = "completed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = {
                "data": data,
                "record_count": len(data),
                "column_count": len(data.columns),
                "execution_success": True
            }
            
            logger.info(f"Step 5 completed: {step.duration:.2f}s, retrieved {len(data)} records")
            return step
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.error = str(e)
            logger.error(f"Step 5 failed: {e}")
            return step
    
    def _execute_step_6_hybrid_comparison(self, retrieved_data: pd.DataFrame, 
                                        use_case_context: str) -> WorkflowStep:
        """Step 6: Hybrid comparison and discrepancy detection"""
        step = WorkflowStep(
            step_name="Hybrid Comparison",
            status="running",
            start_time=datetime.now()
        )
        
        try:
            logger.info("Executing Step 6: Hybrid Comparison")
            
            # Perform hybrid comparison
            comparison_result = self.comparison_engine.compare_data(
                base_data=retrieved_data,
                comparison_data=None,  # Will be determined automatically
                comparison_type="auto",
                comparison_params=None
            )
            
            step.status = "completed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = {
                "comparison_result": comparison_result,
                "discrepancies_found": len(comparison_result.discrepancies),
                "comparison_type": comparison_result.comparison_type,
                "confidence_score": comparison_result.confidence_score
            }
            
            logger.info(f"Step 6 completed: {step.duration:.2f}s, found {len(comparison_result.discrepancies)} discrepancies")
            return step
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.error = str(e)
            logger.error(f"Step 6 failed: {e}")
            return step
    
    def _execute_step_7_report_generation(self, user_query: str, use_case_analysis: UseCaseAnalysis,
                                        sql_query: SQLQuery, comparison_result: ComparisonResult) -> WorkflowStep:
        """Step 7: Generate final report"""
        step = WorkflowStep(
            step_name="Report Generation",
            status="running",
            start_time=datetime.now()
        )
        
        try:
            logger.info("Executing Step 7: Report Generation")
            
            # Generate comprehensive report
            report = self._generate_comprehensive_report(
                user_query, use_case_analysis, sql_query, comparison_result
            )
            
            step.status = "completed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.result = report
            
            logger.info(f"Step 7 completed: {step.duration:.2f}s")
            return step
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            step.error = str(e)
            logger.error(f"Step 7 failed: {e}")
            return step
    
    def _generate_comprehensive_report(self, user_query: str, use_case_analysis: UseCaseAnalysis,
                                    sql_query: SQLQuery, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Generate comprehensive report combining all results"""
        report = {
            "executive_summary": {
                "query": user_query,
                "domain": use_case_analysis.domain,
                "discrepancies_found": len(comparison_result.discrepancies),
                "confidence_score": comparison_result.confidence_score,
                "execution_timestamp": datetime.now().isoformat()
            },
            "use_case_analysis": {
                "domain": use_case_analysis.domain,
                "business_context": use_case_analysis.business_context,
                "key_entities": use_case_analysis.key_entities,
                "business_rules": use_case_analysis.business_rules
            },
            "sql_execution": {
                "query": sql_query.sql,
                "query_type": sql_query.query_type,
                "tables_used": sql_query.tables_used,
                "complexity": sql_query.complexity
            },
            "data_analysis": {
                "comparison_type": comparison_result.comparison_type,
                "statistics": comparison_result.statistics,
                "insights": comparison_result.insights
            },
            "discrepancies": [
                {
                    "type": d.type,
                    "severity": d.severity,
                    "description": d.description,
                    "business_impact": d.business_impact,
                    "suggested_action": d.suggested_action,
                    "confidence": d.confidence
                }
                for d in comparison_result.discrepancies
            ],
            "recommendations": self._generate_recommendations(comparison_result, use_case_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, comparison_result: ComparisonResult, 
                                use_case_analysis: UseCaseAnalysis) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on discrepancies
        if comparison_result.discrepancies:
            high_severity_count = len([d for d in comparison_result.discrepancies if d.severity == "high"])
            if high_severity_count > 0:
                recommendations.append(f"Immediate attention required: {high_severity_count} high-severity discrepancies found")
            
            recommendations.append("Review all discrepancies for business impact assessment")
            recommendations.append("Implement data quality monitoring for affected areas")
        
        # Based on domain
        if use_case_analysis.domain == "healthcare_medicaid":
            recommendations.append("Ensure compliance with Medicaid Drug Rebate Program requirements")
            recommendations.append("Validate URA calculations against CMS guidelines")
        elif use_case_analysis.domain == "finance":
            recommendations.append("Implement financial data validation controls")
            recommendations.append("Review transaction patterns for compliance")
        
        # General recommendations
        recommendations.append("Establish regular data quality review processes")
        recommendations.append("Document business rules and validation logic")
        recommendations.append("Monitor system performance and optimize queries as needed")
        
        return recommendations
    
    def _create_failed_workflow_result(self, workflow_id: str, user_query: str, error: str) -> WorkflowResult:
        """Create result for failed workflow"""
        return WorkflowResult(
            workflow_id=workflow_id,
            user_query=user_query,
            steps=self.current_workflow["steps"] if self.current_workflow else [],
            final_result={"error": error},
            total_duration=0.0,
            status="failed",
            discrepancies_found=0,
            confidence_score=0.0
        )
    
    def get_workflow_history(self) -> Dict[str, WorkflowResult]:
        """Get workflow execution history"""
        return self.workflow_history
    
    def get_workflow_by_id(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get specific workflow by ID"""
        return self.workflow_history.get(workflow_id)
    
    def clear_workflow_history(self):
        """Clear workflow history"""
        self.workflow_history.clear()
        logger.info("Workflow history cleared")
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        if not self.workflow_history:
            return {"total_workflows": 0}
        
        total_workflows = len(self.workflow_history)
        successful_workflows = len([w for w in self.workflow_history.values() if w.status == "completed"])
        failed_workflows = total_workflows - successful_workflows
        
        total_discrepancies = sum(w.discrepancies_found for w in self.workflow_history.values())
        avg_confidence = sum(w.confidence_score for w in self.workflow_history.values()) / total_workflows
        
        return {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "failed_workflows": failed_workflows,
            "success_rate": f"{(successful_workflows/total_workflows)*100:.1f}%",
            "total_discrepancies_found": total_discrepancies,
            "average_confidence_score": f"{avg_confidence:.2f}",
            "average_execution_time": f"{sum(w.total_duration for w in self.workflow_history.values())/total_workflows:.2f}s"
        } 