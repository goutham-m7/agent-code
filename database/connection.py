import oracledb
import pandas as pd
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OracleConnection:
    """Modern Oracle database connection using oracledb package"""
    
    def __init__(self):
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool with modern oracledb"""
        try:
            # Configure oracledb
            oracledb.init_oracle_client()
            
            # Create connection pool
            self.connection_pool = oracledb.create_pool(
                user=settings.ORACLE_USER,
                password=settings.ORACLE_PASSWORD,
                dsn=f"{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/{settings.ORACLE_SERVICE}",
                min=2,
                max=10,
                increment=1,
                encoding="UTF-8",
                nencoding="UTF-8",
                threaded=True,
                getmode=oracledb.POOL_GETMODE_WAIT
            )
            
            logger.info("Oracle connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Oracle connection pool: {e}")
            self.connection_pool = None
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context management"""
        connection = None
        try:
            if self.connection_pool:
                connection = self.connection_pool.acquire()
                yield connection
            else:
                # Fallback to direct connection
                connection = oracledb.connect(
                    user=settings.ORACLE_USER,
                    password=settings.ORACLE_PASSWORD,
                    dsn=f"{settings.ORACLE_HOST}:{settings.ORACLE_PORT}/{settings.ORACLE_SERVICE}"
                )
                yield connection
        except Exception as e:
            logger.error(f"Error acquiring connection: {e}")
            raise
        finally:
            if connection:
                try:
                    if self.connection_pool:
                        self.connection_pool.release(connection)
                    else:
                        connection.close()
                except Exception as e:
                    logger.warning(f"Error releasing connection: {e}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SELECT query and return results as DataFrame"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Fetch column names
                columns = [col[0] for col in cursor.description] if cursor.description else []
                
                # Fetch all rows
                rows = cursor.fetchall()
                
                cursor.close()
                
                # Create DataFrame
                if rows and columns:
                    return pd.DataFrame(rows, columns=columns)
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def execute_dml(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """Execute INSERT, UPDATE, DELETE queries and return affected rows"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                affected_rows = cursor.rowcount
                connection.commit()
                cursor.close()
                
                return affected_rows
                
        except Exception as e:
            logger.error(f"Error executing DML: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get detailed schema information for a table"""
        try:
            query = """
            SELECT 
                column_name,
                data_type,
                data_length,
                data_precision,
                data_scale,
                nullable,
                column_id,
                data_default,
                comments as description
            FROM user_col_comments ucc
            RIGHT JOIN user_tab_columns utc ON ucc.column_name = utc.column_name 
                AND ucc.table_name = utc.table_name
            WHERE utc.table_name = :table_name
            ORDER BY utc.column_id
            """
            
            return self.execute_query(query, {"table_name": table_name.upper()})
            
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            raise
    
    def get_all_tables(self) -> pd.DataFrame:
        """Get list of all user tables"""
        try:
            query = """
            SELECT 
                table_name,
                tablespace_name,
                num_rows as estimated_rows,
                last_analyzed,
                table_type,
                comments as description
            FROM user_tab_comments
            WHERE table_type = 'TABLE'
            ORDER BY table_name
            """
            
            return self.execute_query(query)
            
        except Exception as e:
            logger.error(f"Error getting all tables: {e}")
            raise
    
    def get_table_constraints(self, table_name: str) -> pd.DataFrame:
        """Get constraints for a table"""
        try:
            query = """
            SELECT 
                constraint_name,
                constraint_type,
                search_condition,
                r_owner,
                r_constraint_name
            FROM user_constraints
            WHERE table_name = :table_name
            ORDER BY constraint_name
            """
            
            return self.execute_query(query, {"table_name": table_name.upper()})
            
        except Exception as e:
            logger.error(f"Error getting table constraints: {e}")
            raise
    
    def get_table_indexes(self, table_name: str) -> pd.DataFrame:
        """Get indexes for a table"""
        try:
            query = """
            SELECT 
                index_name,
                index_type,
                uniqueness,
                compression,
                prefix_length
            FROM user_indexes
            WHERE table_name = :table_name
            ORDER BY index_name
            """
            
            return self.execute_query(query, {"table_name": table_name.upper()})
            
        except Exception as e:
            logger.error(f"Error getting table indexes: {e}")
            raise
    
    def get_foreign_keys(self, table_name: str) -> pd.DataFrame:
        """Get foreign key relationships for a table"""
        try:
            query = """
            SELECT 
                a.constraint_name,
                a.column_name,
                c.table_name as referenced_table,
                c.column_name as referenced_column
            FROM user_cons_columns a
            JOIN user_constraints b ON a.constraint_name = b.constraint_name
            JOIN user_cons_columns c ON b.r_constraint_name = c.constraint_name
            WHERE b.constraint_type = 'R' 
            AND b.table_name = :table_name
            ORDER BY a.constraint_name, a.position
            """
            
            return self.execute_query(query, {"table_name": table_name.upper()})
            
        except Exception as e:
            logger.error(f"Error getting foreign keys: {e}")
            raise
    
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table statistics"""
        try:
            stats = {}
            
            # Basic table info
            table_query = """
            SELECT 
                num_rows,
                blocks,
                avg_row_len,
                last_analyzed,
                sample_size
            FROM user_tables
            WHERE table_name = :table_name
            """
            
            table_stats = self.execute_query(table_query, {"table_name": table_name.upper()})
            if not table_stats.empty:
                stats.update(table_stats.iloc[0].to_dict())
            
            # Column statistics
            col_query = """
            SELECT 
                column_name,
                num_distinct,
                num_nulls,
                avg_col_len,
                low_value,
                high_value
            FROM user_tab_columns
            WHERE table_name = :table_name
            """
            
            col_stats = self.execute_query(col_query, {"table_name": table_name.upper()})
            if not col_stats.empty:
                stats['column_statistics'] = col_stats.to_dict('records')
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            return {}
    
    def close_pool(self):
        """Close the connection pool"""
        if self.connection_pool:
            try:
                self.connection_pool.close()
                logger.info("Oracle connection pool closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
    
    def test_connection(self) -> bool:
        """Test if database connection is working"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                cursor.close()
                return result[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

# Global instance
db_connection = OracleConnection() 