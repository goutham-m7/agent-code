#!/usr/bin/env python3
"""
Test Script for Actual Medicaid Schema

This script tests the workflow system with your actual Medicaid database schema
to ensure everything works correctly with the real table structures.
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_schema_import():
    """Test that the actual Medicaid schema can be imported and used"""
    try:
        from demo_complete_workflow import create_medicaid_schema
        
        schema = create_medicaid_schema()
        print(f"✅ Schema imported successfully: {len(schema)} tables")
        
        # Check key tables
        expected_tables = [
            "MN_MCD_CLAIM",
            "MN_MCD_CLAIM_LINE", 
            "MN_MCD_PROGRAM",
            "MN_MCD_PAYMENT",
            "MN_MCD_VALIDATION_MSG"
        ]
        
        for table in expected_tables:
            if table in schema:
                print(f"✅ Found table: {table}")
                table_info = schema[table]
                print(f"   - Columns: {len(table_info['columns'])}")
                print(f"   - Estimated rows: {table_info['estimated_rows']}")
            else:
                print(f"❌ Missing table: {table}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Schema import failed: {e}")
        return False

def test_use_case_analyzer():
    """Test the use case analyzer with actual Medicaid schema"""
    try:
        from demo_complete_workflow import create_medicaid_ura_problem_statement, create_medicaid_schema
        from workflow_agents import UseCaseAnalyzer
        
        problem_statement = create_medicaid_ura_problem_statement()
        database_schema = create_medicaid_schema()
        
        analyzer = UseCaseAnalyzer()
        analysis = analyzer.analyze_use_case(problem_statement, database_schema)
        
        print(f"✅ Use case analysis completed:")
        print(f"   - Domain: {analysis.domain}")
        print(f"   - Key Entities: {len(analysis.key_entities)}")
        print(f"   - Business Rules: {len(analysis.business_rules)}")
        
        # Check if it recognized Medicaid domain
        if "medicaid" in analysis.domain.lower() or "healthcare" in analysis.domain.lower():
            print("✅ Correctly identified Medicaid/healthcare domain")
        else:
            print(f"⚠️  Domain identification may need adjustment: {analysis.domain}")
        
        return True
        
    except Exception as e:
        print(f"❌ Use case analyzer test failed: {e}")
        return False

def test_sql_generator():
    """Test SQL generation with actual Medicaid schema"""
    try:
        from demo_complete_workflow import create_medicaid_ura_problem_statement, create_medicaid_schema
        from workflow_agents import UseCaseAnalyzer, SQLGenerator
        
        problem_statement = create_medicaid_ura_problem_statement()
        database_schema = create_medicaid_schema()
        
        analyzer = UseCaseAnalyzer()
        analysis = analyzer.analyze_use_case(problem_statement, database_schema)
        use_case_context = analyzer.get_analysis_for_llm_context(analysis)
        
        sql_gen = SQLGenerator()
        sql_query = sql_gen.generate_sql(
            "Find URA discrepancies between SYS_CALC_URA and INV_URA for Q1 2024",
            database_schema,
            use_case_context
        )
        
        print(f"✅ SQL generation completed:")
        print(f"   - SQL Type: {sql_query.query_type}")
        print(f"   - Tables Used: {len(sql_query.tables_used)}")
        print(f"   - Complexity: {sql_query.complexity}")
        
        # Check if it used the right tables
        expected_tables = ["MN_MCD_CLAIM_LINE", "MN_MCD_CLAIM"]
        for table in expected_tables:
            if table in sql_query.tables_used:
                print(f"✅ Correctly used table: {table}")
            else:
                print(f"⚠️  Expected table not used: {table}")
        
        return True
        
    except Exception as e:
        print(f"❌ SQL generator test failed: {e}")
        return False

def test_hybrid_comparison():
    """Test hybrid comparison with actual Medicaid schema data"""
    try:
        import pandas as pd
        from workflow_agents import HybridComparisonEngine
        
        # Create sample data using actual Medicaid schema fields
        sample_data = pd.DataFrame({
            'MCD_CLAIM_LINE_ID': [1001, 1002, 1003, 1004, 1005],
            'SYS_CALC_URA': [100.00, 150.00, 200.00, 175.00, 125.00],
            'INV_URA': [95.00, 160.00, 190.00, 180.00, 120.00],
            'INV_UNITS': [100, 150, 200, 175, 125],
            'REBATE_DUE': [5.00, -10.00, 10.00, -5.00, 5.00],
            'ORIG_QTR': ['2024Q1', '2024Q1', '2024Q1', '2024Q1', '2024Q1'],
            'STATE': ['CA', 'CA', 'CA', 'NY', 'NY']
        })
        
        comparison_engine = HybridComparisonEngine()
        comparison_result = comparison_engine.compare_data(sample_data, comparison_type="auto")
        
        print(f"✅ Hybrid comparison completed:")
        print(f"   - Comparison Type: {comparison_result.comparison_type}")
        print(f"   - Discrepancies Found: {len(comparison_result.discrepancies)}")
        print(f"   - Confidence Score: {comparison_result.confidence_score:.2f}")
        
        # Show some discrepancies
        if comparison_result.discrepancies:
            print("   - Sample Discrepancies:")
            for i, disc in enumerate(comparison_result.discrepancies[:3], 1):
                print(f"     {i}. {disc.type}: {disc.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid comparison test failed: {e}")
        return False

def test_workflow_orchestrator():
    """Test the workflow orchestrator with actual Medicaid schema"""
    try:
        from demo_complete_workflow import create_medicaid_ura_problem_statement, create_medicaid_schema
        from workflow_agents import WorkflowOrchestrator
        
        problem_statement = create_medicaid_ura_problem_statement()
        database_schema = create_medicaid_schema()
        
        orchestrator = WorkflowOrchestrator()
        
        print("✅ Workflow orchestrator initialized successfully")
        print(f"   - Use Case Analyzer: {'✅' if orchestrator.use_case_analyzer else '❌'}")
        print(f"   - SQL Generator: {'✅' if orchestrator.sql_generator else '❌'}")
        print(f"   - Comparison Engine: {'✅' if orchestrator.comparison_engine else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow orchestrator test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Actual Medicaid Schema Integration")
    print("=" * 60)
    
    tests = [
        ("Schema Import", test_schema_import),
        ("Use Case Analyzer", test_use_case_analyzer),
        ("SQL Generator", test_sql_generator),
        ("Hybrid Comparison", test_hybrid_comparison),
        ("Workflow Orchestrator", test_workflow_orchestrator)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Medicaid schema is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: python demo_complete_workflow.py")
        print("   2. Configure your Oracle database connection")
        print("   3. Set up your AWS Bedrock credentials")
        print("   4. Start using the workflow system!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 