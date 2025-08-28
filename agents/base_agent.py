import boto3
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import redis
from config import settings

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the deep agent system"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.bedrock_client = self._initialize_bedrock()
        self.redis_client = self._initialize_redis()
        self.memory_key = f"agent_memory:{agent_name}"
    
    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client with session token"""
        try:
            session = boto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                aws_session_token=settings.AWS_SESSION_TOKEN,
                region_name=settings.AWS_REGION
            )
            return session.client('bedrock-runtime')
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def _initialize_redis(self):
        """Initialize Redis client for memory management"""
        try:
            return redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            return None
    
    def invoke_bedrock(self, prompt: str, system_prompt: str = "") -> str:
        """Invoke AWS Bedrock model with the given prompt"""
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
            
            body = json.dumps({
                "prompt": full_prompt,
                "max_tokens_to_sample": settings.MAX_TOKENS,
                "temperature": settings.TEMPERATURE,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"]
            })
            
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=settings.BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('completion', '').strip()
            
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            raise
    
    def store_memory(self, key: str, value: Any, ttl: int = None):
        """Store information in agent memory"""
        if not self.redis_client:
            return
        
        try:
            memory_data = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'agent': self.agent_name
            }
            
            if ttl is None:
                ttl = settings.AGENT_MEMORY_TTL
            
            self.redis_client.setex(
                f"{self.memory_key}:{key}",
                ttl,
                json.dumps(memory_data)
            )
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
    
    def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent memory"""
        if not self.redis_client:
            return None
        
        try:
            memory_data = self.redis_client.get(f"{self.memory_key}:{key}")
            if memory_data:
                data = json.loads(memory_data)
                return data.get('value')
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return None
    
    def clear_memory(self, key: str = None):
        """Clear specific memory key or all memory for this agent"""
        if not self.redis_client:
            return
        
        try:
            if key:
                self.redis_client.delete(f"{self.memory_key}:{key}")
            else:
                # Clear all memory for this agent
                pattern = f"{self.memory_key}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of all stored memory for this agent"""
        if not self.redis_client:
            return {}
        
        try:
            pattern = f"{self.memory_key}:*"
            keys = self.redis_client.keys(pattern)
            summary = {}
            
            for key in keys:
                memory_data = self.redis_client.get(key)
                if memory_data:
                    data = json.loads(memory_data)
                    summary[key.replace(f"{self.memory_key}:", "")] = {
                        'value': data.get('value'),
                        'timestamp': data.get('timestamp')
                    }
            
            return summary
        except Exception as e:
            logger.error(f"Failed to get memory summary: {e}")
            return {}
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return results - to be implemented by subclasses"""
        pass
    
    def __str__(self):
        return f"{self.agent_name} Agent"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.agent_name}')" 