#!/usr/bin/env python3
"""
Test Script for All Schema Managers
Tests and compares the performance of different schema management approaches
"""

import sys
import os
import json
import time
import logging
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_schema() -> Dict[str, Any]:
    """Create a mock schema for testing (simulates 50,000 token schema)"""
    logger.info("Creating mock schema for testing...")
    
    # Create a large schema to simulate real-world scenario
    schema = {}
    
    # Create multiple tables with detailed information
    for i in range(20):  # 20 tables
        table_name = f"TABLE_{i:02d}"
        
        # Create columns
        columns = []
        for j in range(15):  # 15 columns per table
            column = {
                "name": f"COL_{j:02d}",
                "data_type": "VARCHAR2(255)",
                "nullable": "Y" if j % 3 == 0 else "N",
                "description": f"This is a detailed description for column {j} in table {table_name} that contains important business data for the organization's operational processes and analytical requirements.",
                "default_value": None,
                "data_length": 255,
                "data_precision": None,
                "data_scale": None,
                "column_id": j + 1
            }
            columns.append(column)
        
        # Create constraints
        constraints = [
            {
                "name": f"PK_{table_name}",
                "constraint_type": "P",
                "columns": ["COL_00"],
                "description": "Primary key constraint for the table"
            },
            {
                "name": f"UK_{table_name}_NAME",
                "constraint_type": "U",
                "columns": ["COL_01"],
                "description": "Unique constraint for the name column"
            }
        ]
        
        # Create indexes
        indexes = [
            {
                "name": f"IDX_{table_name}_COL01",
                "columns": ["COL_01"],
                "type": "NORMAL",
                "unique": False,
                "description": "Index for improving query performance on COL_01"
            },
            {
                "name": f"IDX_{table_name}_COL02",
                "columns": ["COL_02"],
                "type": "NORMAL",
                "unique": False,
                "description": "Index for improving query performance on COL_02"
            }
        ]
        
        # Create relationships
        relationships = []
        if i > 0:  # Create foreign keys to previous tables
            relationships.append({
                "name": f"FK_{table_name}_REF_TABLE_{i-1:02d}",
                "columns": ["COL_03"],
                "referenced_table": f"TABLE_{i-1:02d}",
                "referenced_columns": ["COL_00"],
                "description": f"Foreign key relationship to TABLE_{i-1:02d}"
            })
        
        schema[table_name] = {
            "table_name": table_name,
            "description": f"This is a comprehensive table {table_name} that stores critical business information including customer data, transaction records, product details, and various operational metrics. The table is designed to support multiple business processes and analytical requirements across different departments and use cases.",
            "table_type": "TABLE",
            "columns": columns,
            "constraints": constraints,
            "indexes": indexes,
            "relationships": relationships,
            "estimated_rows": 100000 + (i * 50000),
            "table_size": f"{10 + (i * 5)}MB",
            "last_analyzed": "2024-01-01",
            "tablespace_name": "USERS",
            "partitioned": "NO",
            "temporary": "N",
            "secondary": "N",
            "nested": "NO",
            "buffer_pool": "DEFAULT",
            "flash_cache": "DEFAULT",
            "cell_flash_cache": "DEFAULT",
            "row_movement": "DISABLED",
            "global_stats": "YES",
            "user_stats": "NO",
            "duration": "DEFAULT",
            "skip_corrupt": "DISABLED",
            "monitoring": "YES",
            "cluster_owner": None,
            "dependencies": "DISABLED",
            "compression": "DISABLED",
            "compression_for": "NONE",
            "deduplication": "NONE",
            "inmemory": "DISABLED",
            "inmemory_priority": "DEFAULT",
            "inmemory_distribute": "DEFAULT",
            "inmemory_compression": "DEFAULT",
            "inmemory_duplicate": "DEFAULT",
            "default_collation": "USING_NLS_COMP",
            "duplicate": "NONE",
            "sharding": "NONE",
            "cellmemory": "DEFAULT",
            "inmemory_service": "DEFAULT",
            "inmemory_size": "DEFAULT",
            "automatic": "YES",
            "automatic_optimizer_stats": "YES",
            "automatic_optimizer_costs": "YES",
            "segment_creation": "AUTO",
            "result_cache": "DEFAULT"
        }
    
    # Add some domain-specific tables for different use cases
    domain_tables = {
        # Healthcare/Medicaid
        "DRUGS": {
            "table_name": "DRUGS",
            "description": "Comprehensive drug information table containing National Drug Code (NDC) data, drug names, formulations, strengths, and manufacturer details for Medicaid Drug Rebate Program calculations and URA analysis.",
            "table_type": "TABLE",
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
            "constraints": [
                {"name": "PK_DRUGS", "constraint_type": "P", "columns": ["NDC_CODE"]},
                {"name": "FK_DRUGS_MANUFACTURER", "constraint_type": "R", "columns": ["MANUFACTURER_ID"], "referenced_table": "MANUFACTURERS", "referenced_columns": ["MANUFACTURER_ID"]}
            ],
            "estimated_rows": 50000,
            "table_size": "25MB"
        },
        
        # E-commerce
        "CUSTOMERS": {
            "table_name": "CUSTOMERS",
            "description": "Customer information table containing personal details, contact information, and customer preferences for e-commerce operations and marketing campaigns.",
            "table_type": "TABLE",
            "columns": [
                {"name": "CUSTOMER_ID", "data_type": "NUMBER(10)", "nullable": "N", "description": "Unique customer identifier"},
                {"name": "FIRST_NAME", "data_type": "VARCHAR2(50)", "nullable": "N", "description": "Customer first name"},
                {"name": "LAST_NAME", "data_type": "VARCHAR2(50)", "nullable": "N", "description": "Customer last name"},
                {"name": "EMAIL", "data_type": "VARCHAR2(100)", "nullable": "N", "description": "Customer email address"},
                {"name": "PHONE", "data_type": "VARCHAR2(20)", "nullable": "Y", "description": "Customer phone number"},
                {"name": "ADDRESS", "data_type": "VARCHAR2(200)", "nullable": "Y", "description": "Customer address"},
                {"name": "CITY", "data_type": "VARCHAR2(50)", "nullable": "Y", "description": "Customer city"},
                {"name": "STATE", "data_type": "VARCHAR2(50)", "nullable": "Y", "description": "Customer state"},
                {"name": "ZIP_CODE", "data_type": "VARCHAR2(10)", "nullable": "Y", "description": "Customer zip code"},
                {"name": "REGISTRATION_DATE", "data_type": "DATE", "nullable": "N", "description": "Customer registration date"}
            ],
            "constraints": [
                {"name": "PK_CUSTOMERS", "constraint_type": "P", "columns": ["CUSTOMER_ID"]},
                {"name": "UK_CUSTOMERS_EMAIL", "constraint_type": "U", "columns": ["EMAIL"]}
            ],
            "estimated_rows": 100000,
            "table_size": "15MB"
        },
        
        # Financial
        "TRANSACTIONS": {
            "table_name": "TRANSACTIONS",
            "description": "Financial transactions table containing payment records, transaction amounts, dates, and status information for accounting and financial reporting.",
            "table_type": "TABLE",
            "columns": [
                {"name": "TRANSACTION_ID", "data_type": "NUMBER(15)", "nullable": "N", "description": "Unique transaction identifier"},
                {"name": "ACCOUNT_ID", "data_type": "NUMBER(10)", "nullable": "N", "description": "Reference to account table"},
                {"name": "TRANSACTION_TYPE", "data_type": "VARCHAR2(20)", "nullable": "N", "description": "Type of transaction (debit, credit, transfer)"},
                {"name": "AMOUNT", "data_type": "NUMBER(10,2)", "nullable": "N", "description": "Transaction amount"},
                {"name": "CURRENCY", "data_type": "VARCHAR2(3)", "nullable": "N", "description": "Transaction currency (USD, EUR, etc.)"},
                {"name": "TRANSACTION_DATE", "data_type": "DATE", "nullable": "N", "description": "Transaction date and time"},
                {"name": "STATUS", "data_type": "VARCHAR2(20)", "nullable": "N", "description": "Transaction status (pending, completed, failed)"},
                {"name": "DESCRIPTION", "data_type": "VARCHAR2(200)", "nullable": "Y", "description": "Transaction description"},
                {"name": "REFERENCE_NUMBER", "data_type": "VARCHAR2(50)", "nullable": "Y", "description": "External reference number"}
            ],
            "constraints": [
                {"name": "PK_TRANSACTIONS", "constraint_type": "P", "columns": ["TRANSACTION_ID"]},
                {"name": "FK_TRANSACTIONS_ACCOUNT", "constraint_type": "R", "columns": ["ACCOUNT_ID"], "referenced_table": "ACCOUNTS", "referenced_columns": ["ACCOUNT_ID"]}
            ],
            "estimated_rows": 500000,
            "table_size": "50MB"
        }
    }
    
    schema.update(domain_tables)
    
    logger.info(f"Created mock schema with {len(schema)} tables")
    return schema

def test_schema_chunking_manager(mock_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Test the schema chunking manager"""
    logger.info("Testing Schema Chunking Manager...")
    
    try:
        from schema_managers.schema_chunking import SchemaChunkingManager
        
        manager = SchemaChunkingManager(max_tokens_per_query=5000)
        
        # Test queries
        test_queries = [
            "Show me all drugs in the DRUGS table",
            "What is the data quality of the PRICING table?",
            "Find all customers from New York",
            "Analyze the structure of the PRODUCTS table",
            "Compare URA calculations between different quarters"
        ]
        
        results = []
        for query in test_queries:
            start_time = time.time()
            result = manager.get_relevant_schema(query, list(mock_schema.keys()), mock_schema)
            end_time = time.time()
            
            results.append({
                "query": query,
                "result": result,
                "execution_time": end_time - start_time
            })
        
        return {
            "manager": "Schema Chunking",
            "status": "success",
            "results": results,
            "cache_stats": manager.get_cache_stats()
        }
        
    except Exception as e:
        logger.error(f"Schema chunking test failed: {e}")
        return {
            "manager": "Schema Chunking",
            "status": "failed",
            "error": str(e)
        }

def test_hierarchical_schema_manager(mock_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Test the hierarchical schema manager"""
    logger.info("Testing Hierarchical Schema Manager...")
    
    try:
        from schema_managers.hierarchical_schema import HierarchicalSchemaManager
        
        manager = HierarchicalSchemaManager()
        
        # Test queries
        test_queries = [
            "Count total records in DRUGS table",
            "Analyze the relationship between DRUGS and PRICING tables",
            "Get comprehensive schema for URA calculation analysis"
        ]
        
        results = []
        for query in test_queries:
            start_time = time.time()
            result = manager.get_adaptive_schema(query, mock_schema, max_tokens=5000)
            end_time = time.time()
            
            results.append({
                "query": query,
                "result": result,
                "execution_time": end_time - start_time
            })
        
        return {
            "manager": "Hierarchical Schema",
            "status": "success",
            "results": results,
            "cache_stats": manager.get_cache_stats()
        }
        
    except Exception as e:
        logger.error(f"Hierarchical schema test failed: {e}")
        return {
            "manager": "Hierarchical Schema",
            "status": "failed",
            "error": str(e)
        }

def test_intent_based_schema_manager(mock_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Test the intent-based schema manager"""
    logger.info("Testing Intent-Based Schema Manager...")
    
    try:
        from schema_managers.intent_based_schema import IntentBasedSchemaSelector
        
        manager = IntentBasedSchemaSelector()
        
        # Test queries
        test_queries = [
            "How many drugs are in the DRUGS table?",
            "Analyze the data quality of the PRICING table",
            "Find discrepancies in URA calculations",
            "Compare AMP vs BP pricing data"
        ]
        
        results = []
        for query in test_queries:
            start_time = time.time()
            result = manager.select_schema_by_intent(query, list(mock_schema.keys()), mock_schema)
            end_time = time.time()
            
            results.append({
                "query": query,
                "result": result,
                "execution_time": end_time - start_time
            })
        
        return {
            "manager": "Intent-Based Schema",
            "status": "success",
            "results": results,
            "intent_stats": manager.get_intent_statistics()
        }
        
    except Exception as e:
        logger.error(f"Intent-based schema test failed: {e}")
        return {
            "manager": "Intent-Based Schema",
            "status": "failed",
            "error": str(e)
        }

def test_vector_schema_manager(mock_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Test the vector schema manager"""
    logger.info("Testing Vector Schema Manager...")
    
    try:
        from schema_managers.vector_schema import VectorSchemaStore
        
        manager = VectorSchemaStore()
        
        # Create embeddings
        start_time = time.time()
        manager.create_schema_embeddings(mock_schema)
        embedding_time = time.time() - start_time
        
        # Test queries
        test_queries = [
            "Find tables related to drug pricing",
            "Show me schema for URA calculations",
            "What tables contain customer information?"
        ]
        
        results = []
        for query in test_queries:
            start_time = time.time()
            result = manager.find_relevant_schema(query, top_k=5, similarity_threshold=0.3)
            end_time = time.time()
            
            results.append({
                "query": query,
                "result": result,
                "execution_time": end_time - start_time
            })
        
        return {
            "manager": "Vector Schema",
            "status": "success",
            "results": results,
            "embedding_time": embedding_time,
            "schema_summary": manager.get_schema_summary()
        }
        
    except Exception as e:
        logger.error(f"Vector schema test failed: {e}")
        return {
            "manager": "Vector Schema",
            "status": "failed",
            "error": str(e)
        }

def test_progressive_schema_manager(mock_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Test the progressive schema manager"""
    logger.info("Testing Progressive Schema Manager...")
    
    try:
        from schema_managers.progressive_schema import ProgressiveSchemaLoader
        
        manager = ProgressiveSchemaLoader()
        
        # Test queries
        test_queries = [
            "Count records in DRUGS table",
            "Analyze pricing data for URA calculations",
            "Get comprehensive schema for discrepancy analysis"
        ]
        
        results = []
        for query in test_queries:
            start_time = time.time()
            result = manager.load_schema_progressively(query, max_tokens=5000)
            end_time = time.time()
            
            results.append({
                "query": query,
                "result": result,
                "execution_time": end_time - start_time
            })
        
        return {
            "manager": "Progressive Schema",
            "status": "success",
            "results": results,
            "load_statistics": manager.get_load_statistics()
        }
        
    except Exception as e:
        logger.error(f"Progressive schema test failed: {e}")
        return {
            "manager": "Progressive Schema",
            "status": "failed",
            "error": str(e)
        }

def test_llm_dynamic_schema_manager(mock_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Test the LLM dynamic schema manager"""
    logger.info("Testing LLM Dynamic Schema Manager...")
    
    try:
        from schema_managers.llm_dynamic_schema import LLMDynamicSchemaManager, LLMProvider
        
        # Test with different LLM providers
        providers = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]
        results = []
        
        for provider in providers:
            try:
                manager = LLMDynamicSchemaManager(
                    llm_provider=provider,
                    max_tokens_per_query=5000
                )
                
                # Test queries
                test_queries = [
                    "How many drugs are in the DRUGS table?",
                    "Analyze customer data for marketing campaigns",
                    "Find financial transaction patterns",
                    "Compare different business domains"
                ]
                
                for query in test_queries:
                    start_time = time.time()
                    result = manager.get_adaptive_schema(
                        query, list(mock_schema.keys()), mock_schema
                    )
                    end_time = time.time()
                    
                    results.append({
                        "provider": provider.value,
                        "query": query,
                        "result": result,
                        "execution_time": end_time - start_time
                    })
                    
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {e}")
                continue
        
        return {
            "manager": "LLM Dynamic Schema",
            "status": "success" if results else "failed",
            "results": results,
            "cache_stats": manager.get_cache_stats() if 'manager' in locals() else {}
        }
        
    except Exception as e:
        logger.error(f"LLM dynamic schema test failed: {e}")
        return {
            "manager": "LLM Dynamic Schema",
            "status": "failed",
            "error": str(e)
        }

def analyze_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze and compare all test results"""
    logger.info("Analyzing test results...")
    
    analysis = {
        "summary": {},
        "performance_comparison": {},
        "recommendations": []
    }
    
    # Count successes and failures
    successful_managers = [r for r in all_results if r["status"] == "success"]
    failed_managers = [r for r in all_results if r["status"] == "failed"]
    
    analysis["summary"] = {
        "total_managers": len(all_results),
        "successful": len(successful_managers),
        "failed": len(failed_managers),
        "success_rate": f"{(len(successful_managers)/len(all_results))*100:.1f}%"
    }
    
    # Analyze performance for successful managers
    for result in successful_managers:
        manager_name = result["manager"]
        analysis["performance_comparison"][manager_name] = {
            "status": "success",
            "total_queries": len(result.get("results", [])),
            "average_execution_time": 0,
            "token_efficiency": 0,
            "cache_effectiveness": 0
        }
        
        # Calculate average execution time
        execution_times = [r["execution_time"] for r in result.get("results", [])]
        if execution_times:
            analysis["performance_comparison"][manager_name]["average_execution_time"] = sum(execution_times) / len(execution_times)
        
        # Analyze token efficiency
        for query_result in result.get("results", []):
            if "result" in query_result and "metadata" in query_result["result"]:
                metadata = query_result["result"]["metadata"]
                if "efficiency" in metadata:
                    efficiency_str = metadata["efficiency"]
                    try:
                        efficiency = float(efficiency_str.replace("%", ""))
                        analysis["performance_comparison"][manager_name]["token_efficiency"] = efficiency
                    except:
                        pass
    
    # Generate recommendations
    if successful_managers:
        # Find best performing manager
        best_manager = min(successful_managers, 
                         key=lambda x: analysis["performance_comparison"][x["manager"]]["average_execution_time"])
        
        analysis["recommendations"].append(f"Best performance: {best_manager['manager']}")
        
        # Token efficiency recommendations
        efficient_managers = [r for r in successful_managers 
                           if analysis["performance_comparison"][r["manager"]]["token_efficiency"] > 70]
        if efficient_managers:
            analysis["recommendations"].append(f"Best token efficiency: {', '.join([r['manager'] for r in efficient_managers])}")
    
    return analysis

def main():
    """Main test execution"""
    print("🧪 Schema Manager Testing Suite")
    print("=" * 50)
    
    # Create mock schema
    mock_schema = create_mock_schema()
    
    # Test all managers
    test_functions = [
        test_schema_chunking_manager,
        test_hierarchical_schema_manager,
        test_intent_based_schema_manager,
        test_vector_schema_manager,
        test_progressive_schema_manager,
        test_llm_dynamic_schema_manager
    ]
    
    all_results = []
    
    for test_func in test_functions:
        try:
            result = test_func(mock_schema)
            all_results.append(result)
            
            if result["status"] == "success":
                print(f"✅ {result['manager']}: PASSED")
            else:
                print(f"❌ {result['manager']}: FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {test_func.__name__}: ERROR - {e}")
            all_results.append({
                "manager": test_func.__name__,
                "status": "failed",
                "error": str(e)
            })
    
    print("\n" + "=" * 50)
    print("📊 Analysis Results")
    print("=" * 50)
    
    # Analyze results
    analysis = analyze_results(all_results)
    
    # Print summary
    summary = analysis["summary"]
    print(f"Total Managers: {summary['total_managers']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']}")
    
    # Print performance comparison
    print("\n📈 Performance Comparison:")
    for manager, perf in analysis["performance_comparison"].items():
        if perf["status"] == "success":
            print(f"\n{manager}:")
            print(f"  Execution Time: {perf['average_execution_time']:.4f}s")
            print(f"  Token Efficiency: {perf['token_efficiency']:.1f}%")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    for rec in analysis["recommendations"]:
        print(f"  • {rec}")
    
    # Save detailed results
    output_file = "schema_manager_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_results": all_results,
            "analysis": analysis,
            "timestamp": time.time()
        }, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    return 0 if analysis["summary"]["successful"] > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 