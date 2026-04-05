import os
from fastapi import HTTPException, Header
from dotenv import load_dotenv
from logger_config import logger

# Load environment variables
load_dotenv()

# Get API keys from environment
API_KEYS = os.getenv("API_KEYS", "").split(",")
API_KEYS = [key.strip() for key in API_KEYS if key.strip()]

logger.info(f"Loaded {len(API_KEYS)} API keys")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail)


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Dependency to verify API key from header.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        The valid API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not x_api_key:
        logger.warning("Request received without API key")
        raise _unauthorized("API key missing. Use X-API-Key header.")
    
    if x_api_key not in API_KEYS:
        logger.warning(f"Request received with invalid API key: {x_api_key[:10]}...")
        raise _unauthorized("Invalid API key")
    
    logger.debug(f"API key validation successful for key: {x_api_key[:10]}...")
    return x_api_key
