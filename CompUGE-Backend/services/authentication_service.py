from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import HTTPException, status
import time

# Configuration
SECRET_KEY = "XBBXBBXBBX"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
PASSWORD = "IrinaIsTheBestSupervisor"
RATE_LIMIT_INTERVAL = 1  # 5 seconds

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
hashed_password = pwd_context.hash(PASSWORD)

# Track last authentication time for rate limiting
last_auth_time = 0


def verify_password(plain_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(tz=timezone.utc) + expires_delta
    else:
        expire = datetime.now(tz=timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_password(password: str) -> bool:
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
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
