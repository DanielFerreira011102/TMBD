from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """API settings loaded from environment variables"""
    model_path: str = "models/best_model.pth"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    redis_host: str = "redis"
    redis_port: int = 6379
    cache_ttl: int = 3600  # Cache results for 1 hour
    
    class Config:
        env_file = ".env"

settings = Settings()