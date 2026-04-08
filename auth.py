import os
from pathlib import Path
from fastapi import HTTPException, Header
from dotenv import load_dotenv
from logger_config import logger

# Load .env from project directory so startup cwd does not matter.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def _get_configured_api_keys() -> list[str]:
    raw = os.getenv("API_KEYS") or os.getenv("API_KEY") or ""
    keys = []
    for item in raw.split(","):
        cleaned = item.strip().strip('"').strip("'")
        if cleaned:
            keys.append(cleaned)
    return keys

logger.info(f"Loaded {len(_get_configured_api_keys())} API keys")


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

    api_keys = _get_configured_api_keys()
    if not api_keys:
        logger.error("No API keys configured. Set API_KEYS in environment or .env")
        raise HTTPException(status_code=500, detail="Server API key configuration missing")
    
    if x_api_key not in api_keys:
        logger.warning(f"Request received with invalid API key: {x_api_key[:10]}...")
        raise _unauthorized("Invalid API key")
    
    logger.debug(f"API key validation successful for key: {x_api_key[:10]}...")
    return x_api_key
