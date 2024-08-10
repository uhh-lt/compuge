from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
import time

# Configuration
SECRET_KEY = "XBBXBBXBBX"  # Replace with a strong secret key
ALGORITHM = "HS256"
PASSWORD = "IrinaIsTheBestSupervisor"  # Replace with your actual password
RATE_LIMIT_INTERVAL = 1  # 1 second

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
hashed_password = pwd_context.hash(PASSWORD)

# Track last authentication time for rate limiting
last_auth_time = 0


def verify_password(plain_password: str) -> bool:
    """Verify a plain password against the stored hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create a JWT access token with an optional expiration time."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(tz=timezone.utc) + expires_delta
    else:
        expire = datetime.now(tz=timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_password(password: str) -> bool:
    """Authenticate the user by verifying the password and enforcing rate limits."""
    global last_auth_time
    current_time = time.time()

    if current_time - last_auth_time < RATE_LIMIT_INTERVAL:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait and try again."
        )

    last_auth_time = current_time

    return verify_password(password)


def decode_token(token: str):
    """Decode a JWT token and return its payload if valid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  # Return the entire payload, including "sub"
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
