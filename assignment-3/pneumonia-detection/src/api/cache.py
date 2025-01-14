import redis
import json
import hashlib
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, host: str, port: int, ttl: int = 3600):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True
        )
        self.ttl = ttl
    
    def generate_key(self, image_data: bytes) -> str:
        """Generate a cache key from image data"""
        return hashlib.sha256(image_data).hexdigest()
    
    def get_prediction(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction result"""
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
    
    def set_prediction(self, key: str, prediction: Dict[str, Any]) -> bool:
        """Cache prediction result"""
        try:
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(prediction)
            )
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False