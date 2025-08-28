"""
Workflow Agents Package

This package contains the complete workflow system that implements the user's requirements:
1. Use Case Analyzer - Analyzes problem statement and database schema
2. SQL Generator - Generates SQL using Bedrock and context
3. Hybrid Comparison Engine - Combines SQL, Python, and LLM for data comparison
4. Workflow Orchestrator - Orchestrates the complete workflow

The workflow follows this pattern:
1. Analyze use case → Create context and memory
2. Data analyst verification (one-time)
3. Extract entities from user query
4. Generate SQL using context and memory
5. Execute SQL and retrieve data
6. Hybrid comparison (SQL + Python + LLM)
7. Report discrepancies to user
"""

from .use_case_analyzer import UseCaseAnalyzer, UseCaseAnalysis
from .sql_generator import SQLGenerator, SQLQuery, QueryPlan
from .hybrid_comparison import HybridComparisonEngine, ComparisonResult, Discrepancy
from .workflow_orchestrator import WorkflowOrchestrator, WorkflowStep, WorkflowResult

__all__ = [
    'UseCaseAnalyzer',
    'UseCaseAnalysis', 
    'SQLGenerator',
    'SQLQuery',
    'QueryPlan',
    'HybridComparisonEngine',
    'ComparisonResult',
    'Discrepancy',
    'WorkflowOrchestrator',
    'WorkflowStep',
    'WorkflowResult'
]

__version__ = "1.0.0"
__author__ = "Deep Agent System"
__description__ = "Complete workflow system for data analysis and discrepancy detection" 