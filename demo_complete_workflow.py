#!/usr/bin/env python3
"""
Complete Workflow Demo

This script demonstrates the complete workflow system that implements the user's requirements:
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

import sys
import os
import json
import logging
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_medicaid_ura_problem_statement() -> str:
    """Create the Medicaid URA problem statement as provided by the user"""
    return """
    The Medicaid Unit Rebate Amount (URA) mismatch between a calculated URA (using formulas from the Medicaid Drug Rebate Program) and the Medicaid Drug Program (MDP)-provided URA can arise due to several reasons. These reasons typically fall into the following categories:
    
    1. Data Discrepancies
    - Incorrect AMP or BP Data: The Average Manufacturer Price (AMP) and Best Price (BP) values used in your calculation may differ from what CMS received from the manufacturer.
    - Quarterly vs. Monthly Data Issues: Mismatches can happen if you're using preliminary or estimated AMP/BP data while CMS uses final, certified values.
    - Outdated NDC Mapping: Incorrect or outdated NDC (National Drug Code) product information may lead to calculation on the wrong formulation or package size.
    
    2. Formula Misapplication or Policy Differences
    - Generic vs. Brand Calculation Error: Brand drugs follow a different URA formula than generics. Misclassification can yield different URAs.
    - Missing CPI Penalty Adjustment: The additional rebate based on CPI-U (inflation penalty) may be omitted or miscalculated in your method.
    
    3. Rebate Type Misunderstanding
    - Baseline AMP or BP Errors: The baseline AMP/BP (from when the drug was first launched) affects CPI-based calculations.
    - Line Extension Rebate Misapplied: For oral solid dosage forms, additional rebates for line extensions can apply under ACA §2501.
    
    4. Timing and Reporting Lag
    - CMS Updates Lag: The URA shown in MDP may include adjustments or corrections not yet reflected in your dataset.
    - Restated Data: Manufacturers sometimes restate AMP/BP data for previous quarters.
    
    5. Cap on URA
    - URA Cap Applied (100% AMP): As of 2024, the cap on URAs (which used to be 100% of AMP) has been removed due to the American Rescue Plan Act.
    
    6. CMS or MDP System-Specific Adjustments
    - Zero URAs for Non-Rebatable Drugs: Some NDCs may have zero URA if they are not covered or terminated.
    - Medicaid Drug Rebate Agreement Changes: Any updates to manufacturer agreements can affect URA values.
    """

def create_medicaid_schema() -> Dict[str, Any]:
    """Create a mock Medicaid database schema for demonstration"""
    return {
        "DRUGS": {
            "table_name": "DRUGS",
            "description": "Comprehensive drug information table containing National Drug Code (NDC) data, drug names, formulations, strengths, and manufacturer details for Medicaid Drug Rebate Program calculations and URA analysis.",
            "columns": [
                {"name": "NDC_CODE", "data_type": "VARCHAR2(20)", "nullable": "N", "description": "National Drug Code identifier"},
                {"name": "DRUG_NAME", "data_type": "VARCHAR2(500)", "nullable": "N", "description": "Official drug name"},
                {"name": "GENERIC_NAME", "data_type": "VARCHAR2(500)", "nullable": "Y", "description": "Generic drug name if applicable"},
                {"name": "STRENGTH", "data_type": "VARCHAR2(100)", "nullable": "Y", "description": "Drug strength and unit"},
                {"name": "DOSAGE_FORM", "data_type": "VARCHAR2(100)", "nullable": "Y", "description": "Dosage form (tablet, capsule, etc.)"},
                {"name": "MANUFACTURER_ID", "data_type": "NUMBER(10)", "nullable": "N", "description": "Reference to manufacturer table"},
                {"name": "BRAND_OR_GENERIC", "data_type": "VARCHAR2(20)", "nullable": "N", "description": "B for Brand, G for Generic"},
                {"name": "LAUNCH_DATE", "data_type": "DATE", "nullable": "Y", "description": "Date drug was first launched"},
                {"name": "TERMINATION_DATE", "data_type": "DATE", "nullable": "Y", "description": "Date drug was terminated if applicable"},
                {"name": "REBATE_ELIGIBLE", "data_type": "VARCHAR2(1)", "nullable": "N", "description": "Y if eligible for rebates, N if not"}
            ],
            "primary_keys": ["NDC_CODE"],
            "foreign_keys": [
                {"columns": ["MANUFACTURER_ID"], "referenced_table": "MANUFACTURERS", "referenced_columns": ["MANUFACTURER_ID"]}
            ],
            "estimated_rows": 50000,
            "table_size": "25MB"
        },
        
        "PRICING": {
            "table_name": "PRICING",
            "description": "Drug pricing information including Average Manufacturer Price (AMP), Best Price (BP), and other pricing data required for URA calculations in the Medicaid Drug Rebate Program.",
            "columns": [
                {"name": "PRICING_ID", "data_type": "NUMBER(15)", "nullable": "N", "description": "Primary key for pricing records"},
                {"name": "NDC_CODE", "data_type": "VARCHAR2(20)", "nullable": "N", "description": "Reference to drugs table"},
                {"name": "QUARTER", "data_type": "VARCHAR2(6)", "nullable": "N", "description": "Quarter in YYYYQ format"},
                {"name": "AMP", "data_type": "NUMBER(10,2)", "nullable": "Y", "description": "Average Manufacturer Price"},
                {"name": "BP", "data_type": "NUMBER(10,2)", "nullable": "Y", "description": "Best Price"},
                {"name": "WAC", "data_type": "NUMBER(10,2)", "nullable": "Y", "description": "Wholesale Acquisition Cost"},
                {"name": "ASP", "data_type": "NUMBER(10,2)", "nullable": "Y", "description": "Average Sales Price"},
                {"name": "REPORTING_DATE", "data_type": "DATE", "nullable": "N", "description": "Date pricing was reported"},
                {"name": "FINAL_FLAG", "data_type": "VARCHAR2(1)", "nullable": "N", "description": "F for Final, P for Preliminary"},
                {"name": "RESTATED_FLAG", "data_type": "VARCHAR2(1)", "nullable": "N", "description": "Y if restated, N if original"}
            ],
            "primary_keys": ["PRICING_ID"],
            "foreign_keys": [
                {"columns": ["NDC_CODE"], "referenced_table": "DRUGS", "referenced_columns": ["NDC_CODE"]}
            ],
            "estimated_rows": 200000,
            "table_size": "50MB"
        },
        
        "REBATES": {
            "table_name": "REBATES",
            "description": "Medicaid drug rebate calculations and URA values including calculated and CMS-provided amounts.",
            "columns": [
                {"name": "REBATE_ID", "data_type": "NUMBER(15)", "nullable": "N", "description": "Primary key for rebate records"},
                {"name": "NDC_CODE", "data_type": "VARCHAR2(20)", "nullable": "N", "description": "Reference to drugs table"},
                {"name": "QUARTER", "data_type": "VARCHAR2(6)", "nullable": "N", "description": "Quarter in YYYYQ format"},
                {"name": "CALCULATED_URA", "data_type": "NUMBER(10,2)", "nullable": "Y", "description": "URA calculated using our formulas"},
                {"name": "CMS_URA", "data_type": "NUMBER(10,2)", "nullable": "Y", "description": "URA provided by CMS/MDP"},
                {"name": "URA_DIFFERENCE", "data_type": "NUMBER(10,2)", "nullable": "Y", "description": "Difference between calculated and CMS URA"},
                {"name": "DIFFERENCE_PERCENTAGE", "data_type": "NUMBER(5,2)", "nullable": "Y", "description": "Percentage difference"},
                {"name": "DISCREPANCY_FLAG", "data_type": "VARCHAR2(1)", "nullable": "N", "description": "Y if discrepancy > threshold, N if not"},
                {"name": "CALCULATION_DATE", "data_type": "DATE", "nullable": "N", "description": "Date URA was calculated"},
                {"name": "NOTES", "data_type": "VARCHAR2(1000)", "nullable": "Y", "description": "Additional notes about the calculation"}
            ],
            "primary_keys": ["REBATE_ID"],
            "foreign_keys": [
                {"columns": ["NDC_CODE"], "referenced_table": "DRUGS", "referenced_columns": ["NDC_CODE"]}
            ],
            "estimated_rows": 150000,
            "table_size": "35MB"
        },
        
        "MANUFACTURERS": {
            "table_name": "MANUFACTURERS",
            "description": "Drug manufacturer information including company details and Medicaid rebate agreements.",
            "columns": [
                {"name": "MANUFACTURER_ID", "data_type": "NUMBER(10)", "nullable": "N", "description": "Primary key for manufacturer"},
                {"name": "MANUFACTURER_NAME", "data_type": "VARCHAR2(200)", "nullable": "N", "description": "Company name"},
                {"name": "FEIN", "data_type": "VARCHAR2(20)", "nullable": "Y", "description": "Federal Employer Identification Number"},
                {"name": "REBATE_AGREEMENT_DATE", "data_type": "DATE", "nullable": "Y", "description": "Date of Medicaid rebate agreement"},
                {"name": "AGREEMENT_STATUS", "data_type": "VARCHAR2(20)", "nullable": "N", "description": "Active, Terminated, Pending"},
                {"name": "CONTACT_EMAIL", "data_type": "VARCHAR2(100)", "nullable": "Y", "description": "Primary contact email"},
                {"name": "CONTACT_PHONE", "data_type": "VARCHAR2(20)", "nullable": "Y", "description": "Primary contact phone"}
            ],
            "primary_keys": ["MANUFACTURER_ID"],
            "foreign_keys": [],
            "estimated_rows": 5000,
            "table_size": "5MB"
        }
    }

def create_sample_queries() -> List[str]:
    """Create sample queries for demonstration"""
    return [
        "Find URA discrepancies greater than 10% for Q1 2024",
        "Compare AMP vs BP pricing data across different manufacturers",
        "Analyze CPI penalty adjustments for brand drugs",
        "Identify drugs with missing or incorrect baseline pricing data",
        "Find quarterly URA variations that exceed normal thresholds"
    ]

def demo_individual_components():
    """Demo individual workflow components"""
    print("🔧 Testing Individual Components")
    print("=" * 50)
    
    try:
        # Test Use Case Analyzer
        print("\n1️⃣ Testing Use Case Analyzer...")
        from workflow_agents import UseCaseAnalyzer
        
        analyzer = UseCaseAnalyzer()
        problem_statement = create_medicaid_ura_problem_statement()
        database_schema = create_medicaid_schema()
        
        analysis = analyzer.analyze_use_case(problem_statement, database_schema)
        print(f"✅ Domain: {analysis.domain}")
        print(f"✅ Key Entities: {len(analysis.key_entities)}")
        print(f"✅ Business Rules: {len(analysis.business_rules)}")
        
        # Test SQL Generator
        print("\n2️⃣ Testing SQL Generator...")
        from workflow_agents import SQLGenerator
        
        sql_gen = SQLGenerator()
        use_case_context = analyzer.get_analysis_for_llm_context(analysis)
        
        sql_query = sql_gen.generate_sql(
            "Find URA discrepancies for Q1 2024",
            database_schema,
            use_case_context
        )
        print(f"✅ SQL Type: {sql_query.query_type}")
        print(f"✅ Tables Used: {len(sql_query.tables_used)}")
        print(f"✅ Complexity: {sql_query.complexity}")
        
        # Test Hybrid Comparison Engine
        print("\n3️⃣ Testing Hybrid Comparison Engine...")
        from workflow_agents import HybridComparisonEngine
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            'NDC_CODE': ['12345-6789-01', '12345-6789-02', '12345-6789-03'],
            'CALCULATED_URA': [100.00, 150.00, 200.00],
            'CMS_URA': [95.00, 160.00, 190.00],
            'QUARTER': ['2024Q1', '2024Q1', '2024Q1']
        })
        
        comparison_engine = HybridComparisonEngine()
        comparison_result = comparison_engine.compare_data(sample_data, comparison_type="auto")
        
        print(f"✅ Comparison Type: {comparison_result.comparison_type}")
        print(f"✅ Discrepancies Found: {len(comparison_result.discrepancies)}")
        print(f"✅ Confidence Score: {comparison_result.confidence_score:.2f}")
        
        print("\n✅ All individual components working!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def demo_complete_workflow():
    """Demo the complete workflow"""
    print("\n🚀 Complete Workflow Demo")
    print("=" * 50)
    
    try:
        from workflow_agents import WorkflowOrchestrator
        
        # Initialize orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Prepare inputs
        problem_statement = create_medicaid_ura_problem_statement()
        database_schema = create_medicaid_schema()
        user_query = "Find URA discrepancies greater than 10% for Q1 2024"
        
        print(f"📋 Problem Statement: {len(problem_statement)} characters")
        print(f"🗄️  Database Schema: {len(database_schema)} tables")
        print(f"❓ User Query: {user_query}")
        print(f"\n🔄 Executing complete workflow...")
        
        # Execute workflow
        workflow_result = orchestrator.execute_workflow(
            user_query=user_query,
            problem_statement=problem_statement,
            database_schema=database_schema,
            data_analyst_verification=True
        )
        
        # Display results
        print(f"\n✅ Workflow completed successfully!")
        print(f"🆔 Workflow ID: {workflow_result.workflow_id}")
        print(f"⏱️  Total Duration: {workflow_result.total_duration:.2f}s")
        print(f"🔍 Discrepancies Found: {workflow_result.discrepancies_found}")
        print(f"🎯 Confidence Score: {workflow_result.confidence_score:.2f}")
        
        # Show workflow steps
        print(f"\n📊 Workflow Steps:")
        for i, step in enumerate(workflow_result.steps, 1):
            status_icon = "✅" if step.status == "completed" else "❌"
            print(f"  {i}. {step.step_name}: {status_icon} ({step.duration:.2f}s)")
        
        # Show final report summary
        if workflow_result.final_result and "executive_summary" in workflow_result.final_result:
            summary = workflow_result.final_result["executive_summary"]
            print(f"\n📋 Executive Summary:")
            print(f"  Domain: {summary.get('domain', 'Unknown')}")
            print(f"  Discrepancies: {summary.get('discrepancies_found', 0)}")
            print(f"  Confidence: {summary.get('confidence_score', 0):.2f}")
        
        # Show discrepancies
        if workflow_result.final_result and "discrepancies" in workflow_result.final_result:
            discrepancies = workflow_result.final_result["discrepancies"]
            if discrepancies:
                print(f"\n🚨 Discrepancies Found:")
                for i, disc in enumerate(discrepancies[:3], 1):  # Show first 3
                    print(f"  {i}. {disc['type']} - {disc['severity']}: {disc['description']}")
                if len(discrepancies) > 3:
                    print(f"  ... and {len(discrepancies) - 3} more")
        
        # Show recommendations
        if workflow_result.final_result and "recommendations" in workflow_result.final_result:
            recommendations = workflow_result.final_result["recommendations"]
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                print(f"  {i}. {rec}")
            if len(recommendations) > 3:
                print(f"  ... and {len(recommendations) - 3} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_multiple_queries():
    """Demo workflow with multiple queries"""
    print("\n🔄 Multiple Queries Demo")
    print("=" * 50)
    
    try:
        from workflow_agents import WorkflowOrchestrator
        
        orchestrator = WorkflowOrchestrator()
        problem_statement = create_medicaid_ura_problem_statement()
        database_schema = create_medicaid_schema()
        sample_queries = create_sample_queries()
        
        print(f"📋 Testing {len(sample_queries)} different queries...")
        
        results = []
        for i, query in enumerate(sample_queries, 1):
            print(f"\n  {i}. {query[:60]}...")
            
            try:
                result = orchestrator.execute_workflow(
                    user_query=query,
                    problem_statement=problem_statement,
                    database_schema=database_schema,
                    data_analyst_verification=False  # Skip verification for demo
                )
                
                if result.status == "completed":
                    print(f"     ✅ Completed: {result.discrepancies_found} discrepancies")
                    results.append(result)
                else:
                    print(f"     ❌ Failed: {result.final_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        # Show summary
        if results:
            print(f"\n📊 Summary:")
            print(f"  Successful Queries: {len(results)}")
            print(f"  Total Discrepancies: {sum(r.discrepancies_found for r in results)}")
            print(f"  Average Confidence: {sum(r.confidence_score for r in results)/len(results):.2f}")
            
            # Show workflow statistics
            stats = orchestrator.get_workflow_statistics()
            print(f"  Success Rate: {stats.get('success_rate', '0%')}")
            print(f"  Average Execution Time: {stats.get('average_execution_time', '0s')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple queries demo failed: {e}")
        return False

def main():
    """Main demo function"""
    print("🎯 Complete Workflow System Demo")
    print("=" * 60)
    print("This demo shows the complete workflow system that implements:")
    print("1. Use Case Analysis → Context & Memory")
    print("2. Data Analyst Verification (one-time)")
    print("3. Entity Extraction from User Query")
    print("4. SQL Generation using Context & Memory")
    print("5. Data Retrieval")
    print("6. Hybrid Comparison (SQL + Python + LLM)")
    print("7. Discrepancy Reporting")
    print("=" * 60)
    
    # Test individual components first
    if not demo_individual_components():
        print("\n❌ Individual component tests failed. Cannot proceed with workflow demo.")
        return 1
    
    # Demo complete workflow
    if not demo_complete_workflow():
        print("\n❌ Complete workflow demo failed.")
        return 1
    
    # Demo multiple queries
    if not demo_multiple_queries():
        print("\n❌ Multiple queries demo failed.")
        return 1
    
    print("\n🎉 All demos completed successfully!")
    print("\n💡 The system is now ready for production use with:")
    print("  • Amazon Bedrock integration for LLM capabilities")
    print("  • Oracle database connectivity (oracledb)")
    print("  • Complete workflow automation")
    print("  • Hybrid comparison engine")
    print("  • Comprehensive reporting")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 