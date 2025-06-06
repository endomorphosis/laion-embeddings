# src/mcp_server/tools/auth_tools.py

import logging
from typing import Dict, Any, Optional
from ..tool_registry import ClaudeMCPTool
from ..validators import validator

logger = logging.getLogger(__name__)

class AuthenticationTool(ClaudeMCPTool):
    """
    Tool for user authentication and JWT token management.
    """
    
    def __init__(self, auth_service=None):
        super().__init__()
        self.name = "authenticate_user"
        self.description = "Authenticate user credentials and return JWT access token."
        self.category = "authentication"
        self.tags = ["auth", "login", "jwt", "security"]
        self.input_schema = {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Username for authentication",
                    "minLength": 1,
                    "maxLength": 50
                },
                "password": {
                    "type": "string",
                    "description": "Password for authentication",
                    "minLength": 1
                }
            },
            "required": ["username", "password"]
        }
        self.auth_service = auth_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute user authentication."""
        try:
            username = validator.validate_text_input(parameters.get("username", ""))
            password = parameters.get("password", "")
            
            if self.auth_service:
                # Use actual auth service
                result = await self.auth_service.authenticate(username, password)
            else:
                # Mock authentication for testing
                result = {
                    "success": True,
                    "access_token": f"mock_token_for_{username}",
                    "token_type": "bearer",
                    "role": "user",
                    "expires_in": 3600
                }
            
            return {
                "type": "authentication",
                "result": result,
                "message": "Authentication completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {
                "type": "authentication",
                "result": {"success": False, "error": str(e)},
                "message": f"Authentication failed: {str(e)}"
            }


class UserInfoTool(ClaudeMCPTool):
    """
    Tool for retrieving current user information from JWT token.
    """
    
    def __init__(self, auth_service=None):
        super().__init__()
        self.name = "get_user_info"
        self.description = "Get current authenticated user information from JWT token."
        self.category = "authentication"
        self.tags = ["auth", "user", "jwt", "profile"]
        self.input_schema = {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "JWT access token",
                    "minLength": 1
                }
            },
            "required": ["token"]
        }
        self.auth_service = auth_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute user info retrieval."""
        try:
            token = parameters.get("token", "")
            
            if self.auth_service:
                # Use actual auth service
                user_info = await self.auth_service.get_user_from_token(token)
            else:
                # Mock user info for testing
                user_info = {
                    "username": "test_user",
                    "role": "user",
                    "permissions": ["read", "write"]
                }
            
            return {
                "type": "user_info",
                "result": user_info,
                "message": "User information retrieved successfully"
            }
            
        except Exception as e:
            logger.error(f"User info retrieval failed: {e}")
            return {
                "type": "user_info",
                "result": {"error": str(e)},
                "message": f"User info retrieval failed: {str(e)}"
            }


class TokenValidationTool(ClaudeMCPTool):
    """
    Tool for validating JWT tokens and checking permissions.
    """
    
    def __init__(self, auth_service=None):
        super().__init__()
        self.name = "validate_token"
        self.description = "Validate JWT token and check user permissions."
        self.category = "authentication"
        self.tags = ["auth", "jwt", "validation", "permissions"]
        self.input_schema = {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "JWT access token to validate",
                    "minLength": 1
                },
                "required_permission": {
                    "type": "string",
                    "description": "Required permission to check (optional)",
                    "enum": ["read", "write", "delete", "manage"]
                }
            },
            "required": ["token"]
        }
        self.auth_service = auth_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute token validation."""
        try:
            token = parameters.get("token", "")
            required_permission = parameters.get("required_permission")
            
            if self.auth_service:
                # Use actual auth service
                validation_result = await self.auth_service.validate_token(token, required_permission)
            else:
                # Mock validation for testing
                validation_result = {
                    "valid": True,
                    "username": "test_user",
                    "role": "user",
                    "permissions": ["read", "write"],
                    "has_required_permission": True if not required_permission else required_permission in ["read", "write"]
                }
            
            return {
                "type": "token_validation",
                "result": validation_result,
                "message": "Token validation completed"
            }
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return {
                "type": "token_validation",
                "result": {"valid": False, "error": str(e)},
                "message": f"Token validation failed: {str(e)}"
            }
