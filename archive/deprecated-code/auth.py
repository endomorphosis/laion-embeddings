"""
Authentication and authorization module for LAION Embeddings API.
Provides JWT-based authentication with role-based access control.
"""

import os
import jwt
import hashlib
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, List
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Security scheme
security = HTTPBearer()

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class UserLogin(BaseModel):
    username: str
    password: str

class TokenData(BaseModel):
    username: str
    role: str = "user"

# In-memory user store (in production, use a database)
users_db = {
    "admin": {
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin"
    },
    "user": {
        "username": "user",
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user"
    }
}

# Role permissions
ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage"],
    "user": ["read", "write"],
    "guest": []
}

def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(plain_password) == hashed_password

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user and return user data if valid."""
    user = users_db.get(username)
    if user and verify_password(password, user["password_hash"]):
        return user
    return None

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role", "user")
        
        if username is None:
            return None
        
        return TokenData(username=username, role=role)
    except jwt.PyJWTError:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get the current authenticated user from JWT token."""
    token = credentials.credentials
    token_data = verify_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data

def require_permission(permission: str):
    """Decorator to require specific permission for an endpoint."""
    def permission_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return current_user
    
    return permission_checker

def require_role(role: str):
    """Decorator to require specific role for an endpoint."""
    def role_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {role}"
            )
        
        return current_user
    
    return role_checker

# Admin-only permission checker
require_admin = require_role("admin")

# Write permission checker (admin or users with write access)
require_write = require_permission("write")

# Read permission checker (most users should have this)
require_read = require_permission("read")
