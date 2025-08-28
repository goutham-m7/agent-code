import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class SchemaChunk:
    """Represents a chunk of schema information with embedding"""
    id: str
    table_name: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    token_count: int = 0

class VectorSchemaStore:
    """Manages schema using vector embeddings for semantic search"""
    
    def __init__(self, embedding_model: str = "sentence-transformers", cache_dir: str = "./cache/embeddings"):
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.schema_chunks = []
        self.embeddings_loaded = False
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            if self.embedding_model == "sentence-transformers":
                # Try to import sentence-transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Sentence transformers model loaded successfully")
                except ImportError:
                    logger.warning("Sentence transformers not available, using fallback")
                    self.model = None
            else:
                self.model = None
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def create_schema_embeddings(self, full_schema: Dict[str, Any], chunk_size: int = 1000):
        """Create embeddings for schema chunks"""
        try:
            logger.info("Creating schema embeddings...")
            
            # Check if embeddings already exist
            cache_file = os.path.join(self.cache_dir, "schema_embeddings.pkl")
            if os.path.exists(cache_file):
                logger.info("Loading existing embeddings from cache")
                self._load_embeddings_from_cache(cache_file)
                return
            
            # Create chunks from schema
            self.schema_chunks = self._create_schema_chunks(full_schema, chunk_size)
            
            # Generate embeddings for each chunk
            if self.model:
                self._generate_embeddings()
            else:
                # Fallback: create simple hash-based embeddings
                self._create_fallback_embeddings()
            
            # Save embeddings to cache
            self._save_embeddings_to_cache(cache_file)
            
            logger.info(f"Created embeddings for {len(self.schema_chunks)} schema chunks")
            
        except Exception as e:
            logger.error(f"Failed to create schema embeddings: {e}")
            # Fallback to simple chunking without embeddings
            self._create_fallback_chunks(full_schema)
    
    def _create_schema_chunks(self, full_schema: Dict[str, Any], chunk_size: int) -> List[SchemaChunk]:
        """Create chunks from the full schema"""
        chunks = []
        
        for table_name, table_data in full_schema.items():
            # Create different types of chunks for each table
            
            # 1. Table overview chunk
            overview_content = self._create_table_overview(table_name, table_data)
            overview_chunk = SchemaChunk(
                id=f"{table_name}_overview",
                table_name=table_name,
                content=overview_content,
                metadata={"type": "overview", "table": table_name},
                token_count=len(overview_content.split())
            )
            chunks.append(overview_chunk)
            
            # 2. Column details chunk
            if "columns" in table_data:
                column_content = self._create_column_details(table_name, table_data["columns"])
                column_chunk = SchemaChunk(
                    id=f"{table_name}_columns",
                    table_name=table_name,
                    content=column_content,
                    metadata={"type": "columns", "table": table_name},
                    token_count=len(column_content.split())
                )
                chunks.append(column_chunk)
            
            # 3. Relationships chunk
            if "foreign_keys" in table_data or "relationships" in table_data:
                relationship_content = self._create_relationship_details(table_name, table_data)
                relationship_chunk = SchemaChunk(
                    id=f"{table_name}_relationships",
                    table_name=table_name,
                    content=relationship_content,
                    metadata={"type": "relationships", "table": table_name},
                    token_count=len(relationship_content.split())
                )
                chunks.append(relationship_chunk)
            
            # 4. Constraints chunk
            if "constraints" in table_data or "indexes" in table_data:
                constraint_content = self._create_constraint_details(table_name, table_data)
                constraint_chunk = SchemaChunk(
                    id=f"{table_name}_constraints",
                    table_name=table_name,
                    content=constraint_content,
                    metadata={"type": "constraints", "table": table_name},
                    token_count=len(constraint_content.split())
                )
                chunks.append(constraint_chunk)
        
        return chunks
    
    def _create_table_overview(self, table_name: str, table_data: Dict[str, Any]) -> str:
        """Create overview content for a table"""
        overview_parts = [
            f"Table: {table_name}",
            f"Description: {table_data.get('description', 'No description available')}",
            f"Type: {table_data.get('table_type', 'TABLE')}"
        ]
        
        if "estimated_rows" in table_data:
            overview_parts.append(f"Estimated rows: {table_data['estimated_rows']}")
        
        if "last_analyzed" in table_data:
            overview_parts.append(f"Last analyzed: {table_data['last_analyzed']}")
        
        return " | ".join(overview_parts)
    
    def _create_column_details(self, table_name: str, columns: List[Dict[str, Any]]) -> str:
        """Create column details content"""
        column_parts = [f"Columns in {table_name}:"]
        
        for col in columns:
            col_info = [
                col.get("name", ""),
                col.get("data_type", ""),
                "NULL" if col.get("nullable") == "Y" else "NOT NULL"
            ]
            
            if col.get("description"):
                col_info.append(col.get("description"))
            
            column_parts.append(" - ".join(col_info))
        
        return " | ".join(column_parts)
    
    def _create_relationship_details(self, table_name: str, table_data: Dict[str, Any]) -> str:
        """Create relationship details content"""
        relationship_parts = [f"Relationships for {table_name}:"]
        
        # Foreign keys
        if "foreign_keys" in table_data:
            for fk in table_data["foreign_keys"]:
                fk_info = [
                    f"FK: {fk.get('columns', [])}",
                    f"-> {fk.get('referenced_table', '')}",
                    f"({fk.get('referenced_columns', [])})"
                ]
                relationship_parts.append(" | ".join(fk_info))
        
        # Other relationships
        if "relationships" in table_data:
            for rel in table_data["relationships"]:
                rel_info = [
                    f"REL: {rel.get('type', '')}",
                    f"-> {rel.get('target_table', '')}"
                ]
                relationship_parts.append(" | ".join(rel_info))
        
        return " | ".join(relationship_parts)
    
    def _create_constraint_details(self, table_name: str, table_data: Dict[str, Any]) -> str:
        """Create constraint details content"""
        constraint_parts = [f"Constraints for {table_name}:"]
        
        # Primary keys
        if "primary_keys" in table_data:
            constraint_parts.append(f"PK: {table_data['primary_keys']}")
        
        # Unique constraints
        if "unique_constraints" in table_data:
            for constraint in table_data["unique_constraints"]:
                constraint_parts.append(f"UNIQUE: {constraint.get('columns', [])}")
        
        # Indexes
        if "indexes" in table_data:
            for idx in table_data["indexes"]:
                idx_info = [
                    f"INDEX: {idx.get('name', '')}",
                    f"on {idx.get('columns', [])}",
                    f"({idx.get('type', 'NORMAL')})"
                ]
                constraint_parts.append(" | ".join(idx_info))
        
        return " | ".join(constraint_parts)
    
    def _generate_embeddings(self):
        """Generate embeddings using the loaded model"""
        try:
            for chunk in self.schema_chunks:
                chunk.embedding = self.model.encode(chunk.content)
            
            self.embeddings_loaded = True
            logger.info("Generated embeddings for all schema chunks")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self._create_fallback_embeddings()
    
    def _create_fallback_embeddings(self):
        """Create fallback embeddings when model is not available"""
        logger.info("Creating fallback embeddings using hash-based approach")
        
        for chunk in self.schema_chunks:
            # Create a simple hash-based embedding
            hash_value = hashlib.md5(chunk.content.encode()).hexdigest()
            # Convert hash to a simple numerical representation
            embedding = np.array([int(hash_value[i:i+2], 16) for i in range(0, 32, 2)], dtype=np.float32)
            chunk.embedding = embedding
        
        self.embeddings_loaded = True
        logger.info("Created fallback embeddings for all schema chunks")
    
    def _create_fallback_chunks(self, full_schema: Dict[str, Any]):
        """Create simple chunks without embeddings when everything fails"""
        logger.info("Creating fallback chunks without embeddings")
        
        self.schema_chunks = []
        for table_name, table_data in full_schema.items():
            chunk = SchemaChunk(
                id=f"{table_name}_fallback",
                table_name=table_name,
                content=f"Table {table_name}: {table_data.get('description', 'No description')}",
                metadata={"type": "fallback", "table": table_name},
                token_count=len(table_name.split()) + 10
            )
            self.schema_chunks.append(chunk)
    
    def _save_embeddings_to_cache(self, cache_file: str):
        """Save embeddings to cache file"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.schema_chunks, f)
            logger.info(f"Saved embeddings to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save embeddings to cache: {e}")
    
    def _load_embeddings_from_cache(self, cache_file: str):
        """Load embeddings from cache file"""
        try:
            with open(cache_file, 'rb') as f:
                self.schema_chunks = pickle.load(f)
            self.embeddings_loaded = True
            logger.info(f"Loaded embeddings from cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to load embeddings from cache: {e}")
            self.schema_chunks = []
    
    def find_relevant_schema(self, query: str, top_k: int = 5, 
                           similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """Find most relevant schema chunks using vector similarity"""
        try:
            if not self.embeddings_loaded:
                return {
                    "error": "Embeddings not loaded",
                    "schema": {},
                    "metadata": {"fallback": True}
                }
            
            # Generate query embedding
            if self.model:
                query_embedding = self.model.encode(query)
            else:
                # Fallback: create hash-based embedding for query
                hash_value = hashlib.md5(query.encode()).hexdigest()
                query_embedding = np.array([int(hash_value[i:i+2], 16) for i in range(0, 32, 2)], dtype=np.float32)
            
            # Calculate similarities
            similarities = []
            for chunk in self.schema_chunks:
                if chunk.embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                    similarities.append((similarity, chunk))
            
            # Sort by similarity and filter by threshold
            similarities.sort(reverse=True)
            relevant_chunks = [
                chunk for similarity, chunk in similarities 
                if similarity >= similarity_threshold
            ][:top_k]
            
            # Group chunks by table
            grouped_schema = {}
            total_tokens = 0
            
            for chunk in relevant_chunks:
                table_name = chunk.table_name
                if table_name not in grouped_schema:
                    grouped_schema[table_name] = {
                        "chunks": [],
                        "total_tokens": 0
                    }
                
                grouped_schema[table_name]["chunks"].append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "type": chunk.metadata.get("type", "unknown"),
                    "similarity": next(sim for sim, ch in similarities if ch.id == chunk.id)
                })
                grouped_schema[table_name]["total_tokens"] += chunk.token_count
                total_tokens += chunk.token_count
            
            # Create final schema structure
            final_schema = {}
            for table_name, table_info in grouped_schema.items():
                final_schema[table_name] = {
                    "chunks": table_info["chunks"],
                    "total_tokens": table_info["total_tokens"],
                    "chunk_count": len(table_info["chunks"])
                }
            
            return {
                "schema": final_schema,
                "metadata": {
                    "query": query,
                    "total_chunks": len(relevant_chunks),
                    "total_tokens": total_tokens,
                    "similarity_threshold": similarity_threshold,
                    "top_k": top_k,
                    "tables_found": list(final_schema.keys()),
                    "embedding_model": self.embedding_model
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to find relevant schema: {e}")
            return {
                "error": str(e),
                "schema": {},
                "metadata": {"fallback": True}
            }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Normalize vectors
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def semantic_search(self, query: str, table_name: str = None, 
                       chunk_type: str = None) -> List[Dict[str, Any]]:
        """Perform semantic search within specific constraints"""
        try:
            # Get all relevant chunks
            result = self.find_relevant_schema(query, top_k=20, similarity_threshold=0.1)
            
            if "error" in result:
                return []
            
            # Filter results
            filtered_chunks = []
            for table, table_info in result["schema"].items():
                # Filter by table name if specified
                if table_name and table != table_name:
                    continue
                
                for chunk in table_info["chunks"]:
                    # Filter by chunk type if specified
                    if chunk_type and chunk["type"] != chunk_type:
                        continue
                    
                    filtered_chunks.append({
                        "table": table,
                        "chunk_id": chunk["id"],
                        "content": chunk["content"],
                        "type": chunk["type"],
                        "similarity": chunk["similarity"],
                        "tokens": chunk.get("token_count", 0)
                    })
            
            # Sort by similarity
            filtered_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_schema_summary(self, table_name: str = None) -> Dict[str, Any]:
        """Get summary of schema chunks"""
        if table_name:
            # Filter by specific table
            table_chunks = [chunk for chunk in self.schema_chunks if chunk.table_name == table_name]
            return {
                "table_name": table_name,
                "chunk_count": len(table_chunks),
                "chunks": [
                    {
                        "id": chunk.id,
                        "type": chunk.metadata.get("type", "unknown"),
                        "token_count": chunk.token_count
                    }
                    for chunk in table_chunks
                ]
            }
        else:
            # Overall summary
            table_summary = {}
            for chunk in self.schema_chunks:
                if chunk.table_name not in table_summary:
                    table_summary[chunk.table_name] = {"chunk_count": 0, "total_tokens": 0}
                
                table_summary[chunk.table_name]["chunk_count"] += 1
                table_summary[chunk.table_name]["total_tokens"] += chunk.token_count
            
            return {
                "total_chunks": len(self.schema_chunks),
                "total_tables": len(table_summary),
                "embeddings_loaded": self.embeddings_loaded,
                "embedding_model": self.embedding_model,
                "table_summary": table_summary
            }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        cache_file = os.path.join(self.cache_dir, "schema_embeddings.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info("Embedding cache cleared")
        
        self.schema_chunks = []
        self.embeddings_loaded = False
    
    def update_schema_chunk(self, table_name: str, chunk_id: str, new_content: str):
        """Update a specific schema chunk"""
        for chunk in self.schema_chunks:
            if chunk.table_name == table_name and chunk.id == chunk_id:
                chunk.content = new_content
                chunk.token_count = len(new_content.split())
                
                # Regenerate embedding if model is available
                if self.model:
                    chunk.embedding = self.model.encode(new_content)
                else:
                    # Update fallback embedding
                    hash_value = hashlib.md5(new_content.encode()).hexdigest()
                    chunk.embedding = np.array([int(hash_value[i:i+2], 16) for i in range(0, 32, 2)], dtype=np.float32)
                
                logger.info(f"Updated schema chunk: {chunk_id}")
                break 