from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Oracle Database Settings
    ORACLE_HOST: str = "localhost"
    ORACLE_PORT: int = 1521
    ORACLE_SERVICE: str = "XE"
    ORACLE_USER: str = ""
    ORACLE_PASSWORD: str = ""
    
    # AWS Settings
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_SESSION_TOKEN: str = ""
    AWS_REGION: str = "us-east-1"
    
    # AWS Bedrock Settings
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    BEDROCLE_ENDPOINT_URL: str = ""
    
    # LLM API Keys for Dynamic Schema Management
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # Agent Settings
    AGENT_MEMORY_TTL: int = 3600  # 1 hour
    AGENT_MAX_MEMORY_ITEMS: int = 1000
    
    # File System Settings
    DATA_DIR: str = "./data"
    QUERY_CACHE_DIR: str = "./cache/queries"
    AGENT_MEMORY_DIR: str = "./cache/memory"
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # Schema Manager Settings
    SCHEMA_MAX_TOKENS_PER_QUERY: int = 5000
    SCHEMA_CACHE_TTL: int = 7200  # 2 hours
    SCHEMA_LLM_PROVIDER: str = "anthropic"  # openai, anthropic, bedrock
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def create_directories():
    """Create necessary directories"""
    settings = Settings()
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.QUERY_CACHE_DIR, exist_ok=True)
    os.makedirs(settings.AGENT_MEMORY_DIR, exist_ok=True)
    os.makedirs("./cache/embeddings", exist_ok=True)

settings = Settings()
create_directories() 