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
    """Create the actual Medicaid database schema as provided by the user"""
    return {
        "MN_MCD_CLAIM": {
            "table_name": "MN_MCD_CLAIM",
            "description": "Main claims table for all Medicaid drug rebate claims. Contains claim-level information including status, program, state, labeler, and rebate amounts.",
            "columns": [
                {"name": "MCD_CLAIM_ID", "data_type": "NUMBER", "nullable": "N", "description": "Primary key for claim identification"},
                {"name": "BUS_ELEM_ID", "data_type": "NUMBER", "nullable": "Y", "description": "Business element identifier"},
                {"name": "ORIGINAL_CLAIM_ID", "data_type": "NUMBER", "nullable": "Y", "description": "Original claim ID for adjustments"},
                {"name": "PROGRAM_ID", "data_type": "NUMBER", "nullable": "Y", "description": "Associated Medicaid program"},
                {"name": "URA_PRICELIST_ID", "data_type": "NUMBER", "nullable": "Y", "description": "URA pricing list reference"},
                {"name": "CLAIM_NUM", "data_type": "VARCHAR2", "nullable": "Y", "description": "Human-readable claim number"},
                {"name": "REV_NUM", "data_type": "NUMBER", "nullable": "Y", "description": "Revision number for claim updates"},
                {"name": "CLAIM_STATUS", "data_type": "VARCHAR2", "nullable": "Y", "description": "Current status (Pending, Approved, Rejected, etc.)"},
                {"name": "CLAIM_TYPE", "data_type": "VARCHAR2", "nullable": "Y", "description": "Type of claim (Original, Adjustment, Conversion)"},
                {"name": "ORIG_QTR", "data_type": "VARCHAR2", "nullable": "Y", "description": "Original quarter for the claim"},
                {"name": "STATE", "data_type": "VARCHAR2", "nullable": "Y", "description": "State where claim is filed"},
                {"name": "LABELER", "data_type": "VARCHAR2", "nullable": "Y", "description": "Drug manufacturer/labeler"},
                {"name": "REBATE_DUE", "data_type": "NUMBER", "nullable": "Y", "description": "Calculated rebate amount due"},
                {"name": "DUE_DATE", "data_type": "DATE", "nullable": "Y", "description": "When rebate payment is due"}
            ],
            "primary_keys": ["MCD_CLAIM_ID"],
            "foreign_keys": [
                {"columns": ["PROGRAM_ID"], "referenced_table": "MN_MCD_PROGRAM", "referenced_columns": ["MCD_PROGRAM_ID"]},
                {"columns": ["URA_PRICELIST_ID"], "referenced_table": "MN_MCD_PRICELIST_PUBLISHED", "referenced_columns": ["URA_PRICELIST_ID"]}
            ],
            "estimated_rows": 100000,
            "table_size": "50MB"
        },
        
        "MN_MCD_CLAIM_LINE": {
            "table_name": "MN_MCD_CLAIM_LINE",
            "description": "Detailed line items for each claim containing drug-specific information, URA amounts, units, and rebate calculations.",
            "columns": [
                {"name": "MCD_CLAIM_LINE_ID", "data_type": "NUMBER", "nullable": "N", "description": "Primary key for claim line"},
                {"name": "CLAIM_ID", "data_type": "NUMBER", "nullable": "N", "description": "Reference to parent claim"},
                {"name": "PRODUCT_ID", "data_type": "NUMBER", "nullable": "Y", "description": "Drug product identifier"},
                {"name": "URA_PRICELIST_ID", "data_type": "NUMBER", "nullable": "Y", "description": "URA pricing list for this line"},
                {"name": "SYS_CALC_URA", "data_type": "NUMBER", "nullable": "Y", "description": "System-calculated URA amount"},
                {"name": "OVERRIDE_URA", "data_type": "NUMBER", "nullable": "Y", "description": "Manually overridden URA amount"},
                {"name": "INV_URA", "data_type": "NUMBER", "nullable": "Y", "description": "Invoice URA amount"},
                {"name": "INV_UNITS", "data_type": "NUMBER", "nullable": "Y", "description": "Invoice units (quantity)"},
                {"name": "INV_REQ_REBATE", "data_type": "NUMBER", "nullable": "Y", "description": "Invoice required rebate amount"},
                {"name": "REBATE_DUE", "data_type": "NUMBER", "nullable": "Y", "description": "Final rebate amount due for this line"},
                {"name": "URA_CALC_QTR", "data_type": "VARCHAR2", "nullable": "Y", "description": "Quarter used for URA calculation"},
                {"name": "VALIDATION_STATUS", "data_type": "VARCHAR2", "nullable": "Y", "description": "Data validation status"},
                {"name": "REPORTED_STATUS", "data_type": "VARCHAR2", "nullable": "Y", "description": "Reporting status to CMS"}
            ],
            "primary_keys": ["MCD_CLAIM_LINE_ID"],
            "foreign_keys": [
                {"columns": ["CLAIM_ID"], "referenced_table": "MN_MCD_CLAIM", "referenced_columns": ["MCD_CLAIM_ID"]},
                {"columns": ["URA_PRICELIST_ID"], "referenced_table": "MN_MCD_PRICELIST_PUBLISHED", "referenced_columns": ["URA_PRICELIST_ID"]}
            ],
            "estimated_rows": 500000,
            "table_size": "100MB"
        },
        
        "MN_MCD_PROGRAM": {
            "table_name": "MN_MCD_PROGRAM",
            "description": "Defines Medicaid drug rebate programs and their rules, formulas, and configuration.",
            "columns": [
                {"name": "MCD_PROGRAM_ID", "data_type": "NUMBER", "nullable": "N", "description": "Primary key for program"},
                {"name": "PROGRAM_SHORT_NAME", "data_type": "VARCHAR2", "nullable": "Y", "description": "Short name for program identification"},
                {"name": "EXTERNAL_PROGRAM_NAME", "data_type": "VARCHAR2", "nullable": "Y", "description": "External program name"},
                {"name": "START_CAL_QTR", "data_type": "VARCHAR2", "nullable": "Y", "description": "Program start quarter"},
                {"name": "END_CAL_QTR", "data_type": "VARCHAR2", "nullable": "Y", "description": "Program end quarter"},
                {"name": "PROGRAM_TYPE", "data_type": "VARCHAR2", "nullable": "Y", "description": "Type of program"},
                {"name": "PROGRAM_STATUS", "data_type": "VARCHAR2", "nullable": "Y", "description": "Current program status"},
                {"name": "DEFAULT_FORMULA_ID", "data_type": "NUMBER", "nullable": "Y", "description": "Default URA calculation formula"}
            ],
            "primary_keys": ["MCD_PROGRAM_ID"],
            "foreign_keys": [],
            "estimated_rows": 1000,
            "table_size": "5MB"
        },
        
        "MN_MCD_PAYMENT": {
            "table_name": "MN_MCD_PAYMENT",
            "description": "Tracks rebate payments and disbursements, including amounts, status, and payment details.",
            "columns": [
                {"name": "MCD_PAYMENT_ID", "data_type": "NUMBER", "nullable": "N", "description": "Primary key for payment"},
                {"name": "PAY_NUM", "data_type": "VARCHAR2", "nullable": "Y", "description": "Payment number"},
                {"name": "STATUS", "data_type": "VARCHAR2", "nullable": "Y", "description": "Payment status (Pending, Processed, Mailed, etc.)"},
                {"name": "CACHED_REBATE_AMOUNT", "data_type": "NUMBER", "nullable": "Y", "description": "Cached rebate amount"},
                {"name": "CACHED_INTEREST", "data_type": "NUMBER", "nullable": "Y", "description": "Cached interest amount"},
                {"name": "CACHED_TOTAL_AMOUNT", "data_type": "NUMBER", "nullable": "Y", "description": "Total amount including rebate and interest"},
                {"name": "CHECK_NUM", "data_type": "VARCHAR2", "nullable": "Y", "description": "Check number for payment"},
                {"name": "CHECK_DATE", "data_type": "DATE", "nullable": "Y", "description": "Date check was issued"},
                {"name": "PAID_DATE", "data_type": "DATE", "nullable": "Y", "description": "Date payment was processed"},
                {"name": "STATE", "data_type": "VARCHAR2", "nullable": "Y", "description": "State for the payment"}
            ],
            "primary_keys": ["MCD_PAYMENT_ID"],
            "foreign_keys": [],
            "estimated_rows": 50000,
            "table_size": "25MB"
        },
        
        "MN_MCD_VALIDATION_MSG": {
            "table_name": "MN_MCD_VALIDATION_MSG",
            "description": "Tracks validation results and business rule compliance for claims and claim lines.",
            "columns": [
                {"name": "MCD_MSG_ID", "data_type": "NUMBER", "nullable": "N", "description": "Primary key"},
                {"name": "CLAIM_LINE_ID", "data_type": "NUMBER", "nullable": "Y", "description": "Reference to claim line"},
                {"name": "VALIDATION_CLASS_NAME", "data_type": "VARCHAR2", "nullable": "Y", "description": "Type of validation"},
                {"name": "SEVERITY", "data_type": "VARCHAR2", "nullable": "Y", "description": "Severity level (Error, Warning, Info)"},
                {"name": "DISPLAY_ORDER", "data_type": "NUMBER", "nullable": "Y", "description": "Order for display"},
                {"name": "RECOM_DISP_UNITS", "data_type": "NUMBER", "nullable": "Y", "description": "Recommended dispute units"},
                {"name": "RECOM_DISPUTE_CODES", "data_type": "VARCHAR2", "nullable": "Y", "description": "Recommended dispute codes"},
                {"name": "FORMULA_DEF", "data_type": "VARCHAR2", "nullable": "Y", "description": "Formula definition"},
                {"name": "FORMULA_EXP", "data_type": "VARCHAR2", "nullable": "Y", "description": "Formula expression"}
            ],
            "primary_keys": ["MCD_MSG_ID"],
            "foreign_keys": [
                {"columns": ["CLAIM_LINE_ID"], "referenced_table": "MN_MCD_CLAIM_LINE", "referenced_columns": ["MCD_CLAIM_LINE_ID"]}
            ],
            "estimated_rows": 200000,
            "table_size": "40MB"
        },
        
        "MN_MCD_PRICELIST_PUBLISHED": {
            "table_name": "MN_MCD_PRICELIST_PUBLISHED",
            "description": "Published URA pricing lists for different quarters and programs.",
            "columns": [
                {"name": "URA_PRICELIST_ID", "data_type": "NUMBER", "nullable": "N", "description": "Primary key for pricing list"},
                {"name": "URA_PRICELIST_NAME", "data_type": "VARCHAR2", "nullable": "Y", "description": "Name of the pricing list"},
                {"name": "QUARTER", "data_type": "VARCHAR2", "nullable": "Y", "description": "Quarter for the pricing list"},
                {"name": "PROGRAM_SHORT_NAME", "data_type": "VARCHAR2", "nullable": "Y", "description": "Associated program"}
            ],
            "primary_keys": ["URA_PRICELIST_ID"],
            "foreign_keys": [],
            "estimated_rows": 5000,
            "table_size": "2MB"
        }
    }

def create_sample_queries() -> List[str]:
    """Create sample queries for demonstration using actual Medicaid schema"""
    return [
        "Find URA discrepancies between SYS_CALC_URA and INV_URA greater than 10% for Q1 2024",
        "Compare rebate amounts across different states and programs",
        "Analyze validation issues by severity and claim status",
        "Identify claims with missing or incorrect URA pricing list references",
        "Find quarterly rebate variations that exceed tolerance thresholds",
        "Analyze payment patterns by state and claim type",
        "Find claims with validation errors that need immediate attention"
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
            "Find URA discrepancies between SYS_CALC_URA and INV_URA for Q1 2024",
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
        
        # Create sample data using actual Medicaid schema fields
        sample_data = pd.DataFrame({
            'MCD_CLAIM_LINE_ID': [1001, 1002, 1003],
            'SYS_CALC_URA': [100.00, 150.00, 200.00],
            'INV_URA': [95.00, 160.00, 190.00],
            'INV_UNITS': [100, 150, 200],
            'REBATE_DUE': [5.00, -10.00, 10.00],
            'ORIG_QTR': ['2024Q1', '2024Q1', '2024Q1']
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
        user_query = "Find URA discrepancies between SYS_CALC_URA and INV_URA greater than 10% for Q1 2024"
        
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