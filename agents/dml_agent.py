import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from database.connection import db_connection

logger = logging.getLogger(__name__)

class DMLAgent(BaseAgent):
    """Agent responsible for Data Manipulation Language (INSERT, UPDATE, DELETE) operations"""
    
    def __init__(self):
        super().__init__("DML")
        self.system_prompt = self._get_dml_system_prompt()
    
    def _get_dml_system_prompt(self) -> str:
        """Get the system prompt for the DML agent"""
        return """You are a Database DML (Data Manipulation Language) Agent. Your role is to:

1. Generate safe and efficient INSERT, UPDATE, DELETE statements
2. Validate data before modification operations
3. Ensure referential integrity and constraints
4. Provide rollback and recovery strategies
5. Handle batch operations and transactions

You should:
- Always use parameterized queries to prevent SQL injection
- Validate data types and constraints before execution
- Consider the impact of operations on related data
- Provide clear feedback on operation results
- Suggest appropriate transaction boundaries

Always respond with:
- The generated DML statement
- Data validation results
- Impact analysis
- Safety recommendations
- Rollback procedures"""

    def generate_insert_statement(self, table_name: str, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate INSERT statement for the given table and data"""
        try:
            # Get table schema for validation
            schema = db_connection.get_table_schema(table_name)
            if schema.empty:
                return {"error": f"Table {table_name} not found or no schema information available"}
            
            # Validate data against schema
            validation_result = self._validate_insert_data(data, schema)
            if not validation_result["is_valid"]:
                return {
                    "error": "Data validation failed",
                    "validation_errors": validation_result["errors"]
                }
            
            # Generate INSERT statement
            columns = list(data.keys())
            placeholders = [f":{col}" for col in columns]
            
            insert_sql = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            
            # Store the generated SQL
            self.store_memory(f"insert_sql_{table_name}_{hash(str(data))}", insert_sql)
            
            return {
                "operation": "INSERT",
                "table": table_name,
                "sql": insert_sql.strip(),
                "data": data,
                "validation": validation_result,
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate INSERT statement: {e}")
            raise
    
    def generate_update_statement(self, table_name: str, update_data: Dict[str, Any], 
                                where_conditions: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate UPDATE statement for the given table and conditions"""
        try:
            # Get table schema for validation
            schema = db_connection.get_table_schema(table_name)
            if schema.empty:
                return {"error": f"Table {table_name} not found or no schema information available"}
            
            # Validate update data against schema
            validation_result = self._validate_update_data(update_data, schema)
            if not validation_result["is_valid"]:
                return {
                    "error": "Data validation failed",
                    "validation_errors": validation_result["errors"]
                }
            
            # Generate UPDATE statement
            set_clause = ", ".join([f"{col} = :{col}" for col in update_data.keys()])
            where_clause = " AND ".join([f"{col} = :where_{col}" for col in where_conditions.keys()])
            
            update_sql = f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE {where_clause}
            """
            
            # Prepare parameters
            params = {**update_data, **{f"where_{k}": v for k, v in where_conditions.items()}}
            
            # Store the generated SQL
            self.store_memory(f"update_sql_{table_name}_{hash(str(update_data))}", update_sql)
            
            return {
                "operation": "UPDATE",
                "table": table_name,
                "sql": update_sql.strip(),
                "update_data": update_data,
                "where_conditions": where_conditions,
                "parameters": params,
                "validation": validation_result,
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate UPDATE statement: {e}")
            raise
    
    def generate_delete_statement(self, table_name: str, where_conditions: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate DELETE statement for the given table and conditions"""
        try:
            # Get table schema for validation
            schema = db_connection.get_table_schema(table_name)
            if schema.empty:
                return {"error": f"Table {table_name} not found or no schema information available"}
            
            # Generate DELETE statement
            where_clause = " AND ".join([f"{col} = :{col}" for col in where_conditions.keys()])
            
            delete_sql = f"""
            DELETE FROM {table_name}
            WHERE {where_clause}
            """
            
            # Store the generated SQL
            self.store_memory(f"delete_sql_{table_name}_{hash(str(where_conditions))}", delete_sql)
            
            return {
                "operation": "DELETE",
                "table": table_name,
                "sql": delete_sql.strip(),
                "where_conditions": where_conditions,
                "parameters": where_conditions,
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate DELETE statement: {e}")
            raise
    
    def execute_dml_operation(self, dml_statement: Dict[str, Any], transaction_mode: str = "auto") -> Dict[str, Any]:
        """Execute DML operation with transaction management"""
        try:
            operation = dml_statement.get("operation")
            sql = dml_statement.get("sql")
            parameters = dml_statement.get("parameters", {})
            
            if not sql:
                return {"error": "No SQL statement provided"}
            
            # Store execution attempt
            self.store_memory(f"dml_execution_{operation}_{hash(sql)}", {
                "sql": sql,
                "parameters": parameters,
                "timestamp": self._get_current_timestamp()
            })
            
            # Execute the DML operation
            if transaction_mode == "manual":
                # Manual transaction mode - user controls commit/rollback
                result = self._execute_with_manual_transaction(sql, parameters)
            else:
                # Auto transaction mode - auto-commit
                result = self._execute_with_auto_transaction(sql, parameters)
            
            # Store execution result
            self.store_memory(f"dml_result_{operation}_{hash(sql)}", result)
            
            return result
            
        except Exception as e:
            logger.error(f"DML execution failed: {e}")
            return {
                "error": str(e),
                "operation": dml_statement.get("operation", "unknown"),
                "timestamp": self._get_current_timestamp()
            }
    
    def _execute_with_auto_transaction(self, sql: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DML with automatic transaction management"""
        try:
            affected_rows = db_connection.execute_dml(sql, parameters)
            
            return {
                "success": True,
                "sql": sql,
                "parameters": parameters,
                "affected_rows": affected_rows,
                "transaction_mode": "auto",
                "execution_timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Auto transaction execution failed: {e}")
            return {
                "success": False,
                "sql": sql,
                "parameters": parameters,
                "error": str(e),
                "transaction_mode": "auto",
                "execution_timestamp": self._get_current_timestamp()
            }
    
    def _execute_with_manual_transaction(self, sql: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DML with manual transaction management"""
        try:
            # This would require connection-level transaction control
            # For now, fall back to auto transaction
            logger.warning("Manual transaction mode not fully implemented, using auto mode")
            return self._execute_with_auto_transaction(sql, parameters)
            
        except Exception as e:
            logger.error(f"Manual transaction execution failed: {e}")
            return {
                "success": False,
                "sql": sql,
                "parameters": parameters,
                "error": str(e),
                "transaction_mode": "manual",
                "execution_timestamp": self._get_current_timestamp()
            }
    
    def _validate_insert_data(self, data: Dict[str, Any], schema: pd.DataFrame) -> Dict[str, Any]:
        """Validate data for INSERT operation against table schema"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check if all required columns are provided
            required_columns = schema[schema['NULLABLE'] == 'N']['COLUMN_NAME'].tolist()
            provided_columns = set(data.keys())
            
            missing_required = set(required_columns) - provided_columns
            if missing_required:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Missing required columns: {list(missing_required)}")
            
            # Check data types (basic validation)
            for column, value in data.items():
                if column in schema['COLUMN_NAME'].values:
                    col_schema = schema[schema['COLUMN_NAME'] == column].iloc[0]
                    data_type = col_schema['DATA_TYPE']
                    
                    # Basic type validation
                    if data_type in ['NUMBER', 'FLOAT', 'BINARY_FLOAT', 'BINARY_DOUBLE']:
                        if not isinstance(value, (int, float)) and not str(value).replace('.', '').replace('-', '').isdigit():
                            validation_result["warnings"].append(f"Column {column} expects numeric value, got {type(value).__name__}")
                    
                    elif data_type in ['DATE', 'TIMESTAMP']:
                        if not isinstance(value, str):  # Basic check - could be enhanced
                            validation_result["warnings"].append(f"Column {column} expects date/timestamp, got {type(value).__name__}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    def _validate_update_data(self, data: Dict[str, Any], schema: pd.DataFrame) -> Dict[str, Any]:
        """Validate data for UPDATE operation against table schema"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check if columns exist in schema
            schema_columns = set(schema['COLUMN_NAME'].values)
            update_columns = set(data.keys())
            
            invalid_columns = update_columns - schema_columns
            if invalid_columns:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Invalid columns: {list(invalid_columns)}")
            
            # Check data types (basic validation)
            for column, value in data.items():
                if column in schema_columns:
                    col_schema = schema[schema['COLUMN_NAME'] == column].iloc[0]
                    data_type = col_schema['DATA_TYPE']
                    
                    # Basic type validation
                    if data_type in ['NUMBER', 'FLOAT', 'BINARY_FLOAT', 'BINARY_DOUBLE']:
                        if not isinstance(value, (int, float)) and not str(value).replace('.', '').replace('-', '').isdigit():
                            validation_result["warnings"].append(f"Column {column} expects numeric value, got {type(value).__name__}")
                    
                    elif data_type in ['DATE', 'TIMESTAMP']:
                        if not isinstance(value, str):  # Basic check - could be enhanced
                            validation_result["warnings"].append(f"Column {column} expects date/timestamp, got {type(value).__name__}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Update data validation failed: {e}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    def batch_insert(self, table_name: str, data_list: List[Dict[str, Any]], 
                    batch_size: int = 1000) -> Dict[str, Any]:
        """Execute batch INSERT operations"""
        try:
            if not data_list:
                return {"error": "No data provided for batch insert"}
            
            # Validate first record to get schema
            schema = db_connection.get_table_schema(table_name)
            if schema.empty:
                return {"error": f"Table {table_name} not found"}
            
            # Validate all data
            validation_results = []
            for i, data in enumerate(data_list):
                validation = self._validate_insert_data(data, schema)
                validation_results.append({
                    "record_index": i,
                    "validation": validation
                })
                
                if not validation["is_valid"]:
                    return {
                        "error": f"Validation failed for record {i}",
                        "validation_errors": validation["errors"]
                    }
            
            # Execute batch insert
            total_affected = 0
            batch_results = []
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                # Generate batch INSERT statement
                if len(batch) == 1:
                    # Single record insert
                    insert_result = self.generate_insert_statement(table_name, batch[0])
                    if "error" in insert_result:
                        return insert_result
                    
                    execution_result = self.execute_dml_operation(insert_result)
                    if execution_result["success"]:
                        total_affected += execution_result["affected_rows"]
                        batch_results.append(execution_result)
                else:
                    # Multiple records insert
                    batch_result = self._execute_batch_insert(table_name, batch, schema)
                    if "error" in batch_result:
                        return batch_result
                    
                    total_affected += batch_result["affected_rows"]
                    batch_results.append(batch_result)
            
            return {
                "success": True,
                "operation": "BATCH_INSERT",
                "table": table_name,
                "total_records": len(data_list),
                "total_affected": total_affected,
                "batch_size": batch_size,
                "batch_results": batch_results,
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return {
                "error": str(e),
                "operation": "BATCH_INSERT",
                "table": table_name,
                "timestamp": self._get_current_timestamp()
            }
    
    def _execute_batch_insert(self, table_name: str, batch: List[Dict[str, Any]], 
                             schema: pd.DataFrame) -> Dict[str, Any]:
        """Execute batch insert for multiple records"""
        try:
            # Generate batch INSERT statement
            columns = list(batch[0].keys())
            placeholders = [f":{col}" for col in columns]
            
            # Oracle batch insert using UNION ALL
            values_clauses = []
            all_parameters = {}
            
            for i, record in enumerate(batch):
                record_placeholders = [f":{col}_{i}" for col in columns]
                values_clauses.append(f"({', '.join(record_placeholders)})")
                
                for col in columns:
                    all_parameters[f"{col}_{i}"] = record[col]
            
            batch_sql = f"""
            INSERT ALL
            {' '.join([f'INTO {table_name} ({", ".join(columns)}) VALUES {clause}' for clause in values_clauses])}
            SELECT * FROM DUAL
            """
            
            # Execute batch insert
            affected_rows = db_connection.execute_dml(batch_sql, all_parameters)
            
            return {
                "success": True,
                "sql": batch_sql.strip(),
                "affected_rows": affected_rows,
                "batch_size": len(batch),
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Batch insert execution failed: {e}")
            return {
                "error": str(e),
                "batch_size": len(batch),
                "timestamp": self._get_current_timestamp()
            }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return DML operation results"""
        operation_type = input_data.get("operation", "").upper()
        table_name = input_data.get("table_name", "")
        data = input_data.get("data", {})
        where_conditions = input_data.get("where_conditions", {})
        context = input_data.get("context", {})
        
        try:
            if operation_type == "INSERT":
                if isinstance(data, list):
                    return self.batch_insert(table_name, data, context.get("batch_size", 1000))
                else:
                    insert_statement = self.generate_insert_statement(table_name, data, context)
                    if "error" in insert_statement:
                        return insert_statement
                    return self.execute_dml_operation(insert_statement)
            
            elif operation_type == "UPDATE":
                update_statement = self.generate_update_statement(table_name, data, where_conditions, context)
                if "error" in update_statement:
                    return update_statement
                return self.execute_dml_operation(update_statement)
            
            elif operation_type == "DELETE":
                delete_statement = self.generate_delete_statement(table_name, where_conditions, context)
                if "error" in delete_statement:
                    return delete_statement
                return self.execute_dml_operation(delete_statement)
            
            else:
                return {
                    "error": f"Unsupported operation type: {operation_type}",
                    "supported_operations": ["INSERT", "UPDATE", "DELETE"],
                    "timestamp": self._get_current_timestamp()
                }
                
        except Exception as e:
            logger.error(f"DML processing failed: {e}")
            return {
                "error": str(e),
                "operation": operation_type,
                "table_name": table_name,
                "timestamp": self._get_current_timestamp()
            }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_dml_history(self) -> List[Dict[str, Any]]:
        """Get history of recent DML operations"""
        memory_summary = self.get_memory_summary()
        operations = []
        
        for key, data in memory_summary.items():
            if any(prefix in key for prefix in ["insert_sql_", "update_sql_", "delete_sql_", "dml_execution_", "dml_result_"]):
                operations.append({
                    "key": key,
                    "data": data.get("value"),
                    "timestamp": data.get("timestamp")
                })
        
        return sorted(operations, key=lambda x: x["timestamp"], reverse=True) 