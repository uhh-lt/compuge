import logging
import os
import time
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext

# Configuration
SECRET_KEY = "I4X2mFx3Q-Kz9z5Yh8GqKRA87MvqEr9NeZyHsSx3P2fP_FK4-bSRK8_2DctRxFObhYzpHwvM9hYgUEs-VWJD0g"
ALGORITHM = "HS256"
PASSWORD = os.environ.get(" ADMIN_PASSWORD", "IrinaIsTheBestSupervisor")
RATE_LIMIT_INTERVAL = 1  # 1 second

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
hashed_password = pwd_context.hash(PASSWORD)

# Track last authentication time for rate limiting
last_auth_time = 0

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def verify_password(plain_password: str) -> bool:
    """Verify a plain password against the stored hashed password."""
    try:
        is_valid = pwd_context.verify(plain_password, hashed_password)
        if is_valid:
            logger.info("Password verification successful.")
        else:
            logger.warning("Password verification failed.")
        return is_valid
    except Exception as e:
        logger.error(f"Error during password verification: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password verification failed")


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create a JWT access token with an optional expiration time."""
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(tz=timezone.utc) + expires_delta
        else:
            expire = datetime.now(tz=timezone.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"Access token created for user: {data.get('sub')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error during token creation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token creation failed")


def authenticate_password(password: str) -> bool:
    """Authenticate the user by verifying the password and enforcing rate limits."""
    global last_auth_time
    try:
        current_time = time.time()
        if current_time - last_auth_time < RATE_LIMIT_INTERVAL:
            logger.warning("Rate limit exceeded during authentication attempt.")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please wait and try again."
            )
        last_auth_time = current_time
        is_authenticated = verify_password(password)
        if is_authenticated:
            logger.info("User authenticated successfully.")
        else:
            logger.warning("Authentication failed due to incorrect password.")
        return is_authenticated
    except HTTPException as he:
        raise he  # Re-raise the HTTPException to handle it as expected by FastAPI
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Authentication failed")


def decode_token(token: str):
    """Decode a JWT token and return its payload if valid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info("Token decoded successfully.")
        return payload  # Return the entire payload, including "sub"
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        logger.warning("Invalid token provided.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except Exception as e:
        logger.error(f"Error during token decoding: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token decoding failed")
