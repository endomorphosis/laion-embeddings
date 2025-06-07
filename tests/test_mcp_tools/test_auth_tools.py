# tests/test_mcp_tools/test_auth_tools.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

try:
    import jwt
except ImportError:
    jwt = None

from src.mcp_server.tools.auth_tools import (
    AuthenticationTool,
    UserInfoTool,
    TokenValidationTool
)
from src.mcp_server.error_handlers import MCPError, ValidationError


class TestAuthenticationTool:
    """Test cases for AuthenticationTool."""

    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        service = Mock()
        service.authenticate = AsyncMock(return_value={
            "user_id": "user123",
            "username": "testuser",
            "access_token": "mock_jwt_token",
            "refresh_token": "mock_refresh_token",
            "expires_in": 3600
        })
        service.validate_credentials = AsyncMock(return_value=True)
        return service

    @pytest.fixture
    def auth_tool(self, mock_auth_service):
        """Create AuthenticationTool instance."""
        return AuthenticationTool(auth_service=mock_auth_service)

    @pytest.mark.asyncio
    async def test_successful_authentication(self, auth_tool):
        """Test successful user authentication."""
        parameters = {
            "username": "testuser",
            "password": "testpass123"
        }

        result = await auth_tool.execute(parameters)

        assert result["type"] == "authentication"
        assert "result" in result
        assert result["message"] == "Authentication completed successfully"
        # Check the nested result structure
        auth_result = result["result"]
        assert auth_result["user_id"] == "user123"
        assert auth_result["username"] == "testuser"
        assert "access_token" in auth_result
        assert "refresh_token" in auth_result
        assert auth_result["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_authentication_with_invalid_credentials(self, auth_tool):
        """Test authentication with invalid credentials."""
        auth_tool.auth_service.authenticate.side_effect = Exception("Invalid credentials")

        parameters = {
            "username": "testuser",
            "password": "wrongpass"
        }

        result = await auth_tool.execute(parameters)

        assert result["type"] == "authentication"
        assert result["result"]["success"] == False
        assert "Invalid credentials" in result["result"]["error"]

    @pytest.mark.asyncio
    async def test_authentication_with_empty_username(self, auth_tool):
        """Test authentication with empty username."""
        parameters = {
            "username": "",
            "password": "testpass123"
        }

        result = await auth_tool.execute(parameters)
        # The tool should handle this gracefully and return an error result
        assert result["type"] == "authentication"
        assert result["result"]["success"] == False

    @pytest.mark.asyncio
    async def test_authentication_with_missing_password(self, auth_tool):
        """Test authentication with missing password."""
        parameters = {
            "username": "testuser"
        }

        with pytest.raises(ValidationError):
            await auth_tool.execute(parameters)

    @pytest.mark.asyncio
    async def test_mock_authentication_without_service(self):
        """Test mock authentication when no service is provided."""
        tool = AuthenticationTool()
        
        parameters = {
            "username": "testuser",
            "password": "testpass123"
        }

        result = await tool.execute(parameters)

        assert result["status"] == "success"
        assert result["username"] == "testuser"
        assert "access_token" in result
        assert "mock_token" in result["access_token"]

    @pytest.mark.asyncio
    async def test_username_validation(self, auth_tool):
        """Test username validation."""
        parameters = {
            "username": "a" * 60,  # Too long
            "password": "testpass123"
        }

        with pytest.raises(ValidationError):
            await auth_tool.execute(parameters)


# class TestAuthorizationTool:
#     """Test cases for AuthorizationTool."""

#     @pytest.fixture
#     def mock_auth_service(self):
#         """Mock authorization service."""
#         service = Mock()
#         service.check_permission = AsyncMock(return_value=True)
#         service.get_user_roles = AsyncMock(return_value=["user", "admin"])
#         return service

#     @pytest.fixture
#     def auth_tool(self, mock_auth_service):
#         """Create AuthorizationTool instance."""
#         return AuthorizationTool(auth_service=mock_auth_service)

#     @pytest.mark.asyncio
#     async def test_permission_check_allowed(self, auth_tool):
#         """Test permission check for allowed action."""
#         parameters = {
#             "user_id": "user123",
#             "resource": "embeddings",
#             "action": "read"
#         }

#         result = await auth_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["allowed"] is True
#         assert result["user_id"] == "user123"
#         assert result["resource"] == "embeddings"
#         assert result["action"] == "read"

#     @pytest.mark.asyncio
#     async def test_permission_check_denied(self, auth_tool):
#         """Test permission check for denied action."""
#         auth_tool.auth_service.check_permission.return_value = False

#         parameters = {
#             "user_id": "user123",
#             "resource": "admin",
#             "action": "delete"
#         }

#         result = await auth_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["allowed"] is False

#     @pytest.mark.asyncio
#     async def test_role_based_authorization(self, auth_tool):
#         """Test role-based authorization."""
#         parameters = {
#             "user_id": "user123",
#             "required_roles": ["admin"],
#             "resource": "system",
#             "action": "configure"
#         }

#         result = await auth_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert "user_roles" in result
#         assert "admin" in result["user_roles"]

#     @pytest.mark.asyncio
#     async def test_authorization_with_invalid_user(self, auth_tool):
#         """Test authorization with invalid user ID."""
#         auth_tool.auth_service.check_permission.side_effect = Exception("User not found")

#         parameters = {
#             "user_id": "invalid_user",
#             "resource": "embeddings",
#             "action": "read"
#         }

#         result = await auth_tool.execute(parameters)

#         assert result["status"] == "error"
#         assert "User not found" in result["error"]


class TestTokenValidationTool:
    """Test cases for TokenValidationTool."""

    @pytest.fixture
    def mock_token_service(self):
        """Mock token service."""
        service = Mock()
        service.validate_token = AsyncMock(return_value={
            "valid": True,
            "user_id": "user123",
            "username": "testuser",
            "expires_at": datetime.now() + timedelta(hours=1)
        })
        service.decode_token = AsyncMock(return_value={
            "user_id": "user123",
            "username": "testuser",
            "exp": (datetime.now() + timedelta(hours=1)).timestamp()
        })
        return service

    @pytest.fixture
    def token_tool(self, mock_token_service):
        """Create TokenValidationTool instance."""
        return TokenValidationTool(auth_service=mock_token_service)

    @pytest.mark.asyncio
    async def test_valid_token_validation(self, token_tool):
        """Test validation of valid JWT token."""
        parameters = {
            "token": "valid_jwt_token"
        }

        result = await token_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["valid"] is True
        assert result["user_id"] == "user123"
        assert result["username"] == "testuser"
        assert "expires_at" in result

    @pytest.mark.asyncio
    async def test_invalid_token_validation(self, token_tool):
        """Test validation of invalid JWT token."""
        token_tool.token_service.validate_token.return_value = {
            "valid": False,
            "error": "Token expired"
        }

        parameters = {
            "token": "expired_jwt_token"
        }

        result = await token_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["valid"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_malformed_token(self, token_tool):
        """Test validation of malformed token."""
        token_tool.token_service.validate_token.side_effect = Exception("Malformed token")

        parameters = {
            "token": "malformed_token"
        }

        result = await token_tool.execute(parameters)

        assert result["status"] == "error"
        assert "Malformed token" in result["error"]

    @pytest.mark.asyncio
    async def test_token_refresh(self, token_tool):
        """Test token refresh functionality."""
        token_tool.token_service.refresh_token = AsyncMock(return_value={
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600
        })

        parameters = {
            "token": "valid_refresh_token",
            "action": "refresh"
        }

        result = await token_tool.execute(parameters)

        assert result["status"] == "success"
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_token_decode(self, token_tool):
        """Test token decoding functionality."""
        parameters = {
            "token": "valid_jwt_token",
            "action": "decode"
        }

        result = await token_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["user_id"] == "user123"
        assert result["username"] == "testuser"
        assert "exp" in result


# class TestUserManagementTool:
#     """Test cases for UserManagementTool."""

#     @pytest.fixture
#     def mock_user_service(self):
#         """Mock user service."""
#         service = Mock()
#         service.create_user = AsyncMock(return_value={
#             "user_id": "user123",
#             "username": "newuser",
#             "email": "newuser@example.com",
#             "created_at": datetime.now()
#         })
#         service.get_user = AsyncMock(return_value={
#             "user_id": "user123",
#             "username": "testuser",
#             "email": "testuser@example.com",
#             "roles": ["user"],
#             "created_at": datetime.now()
#         })
#         service.update_user = AsyncMock(return_value=True)
#         service.delete_user = AsyncMock(return_value=True)
#         return service

#     @pytest.fixture
#     def user_tool(self, mock_user_service):
#         """Create UserManagementTool instance."""
#         return UserManagementTool(user_service=mock_user_service)

#     @pytest.mark.asyncio
#     async def test_create_user(self, user_tool):
#         """Test user creation."""
#         parameters = {
#             "action": "create",
#             "username": "newuser",
#             "email": "newuser@example.com",
#             "password": "securepass123",
#             "roles": ["user"]
#         }

#         result = await user_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["user_id"] == "user123"
#         assert result["username"] == "newuser"
#         assert result["email"] == "newuser@example.com"

#     @pytest.mark.asyncio
#     async def test_get_user(self, user_tool):
#         """Test user retrieval."""
#         parameters = {
#             "action": "get",
#             "user_id": "user123"
#         }

#         result = await user_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["user_id"] == "user123"
#         assert result["username"] == "testuser"
#         assert result["email"] == "testuser@example.com"
#         assert "roles" in result

#     @pytest.mark.asyncio
#     async def test_update_user(self, user_tool):
#         """Test user update."""
#         parameters = {
#             "action": "update",
#             "user_id": "user123",
#             "email": "updated@example.com",
#             "roles": ["user", "admin"]
#         }

#         result = await user_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["updated"] is True

#     @pytest.mark.asyncio
#     async def test_delete_user(self, user_tool):
#         """Test user deletion."""
#         parameters = {
#             "action": "delete",
#             "user_id": "user123"
#         }

#         result = await user_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["deleted"] is True

#     @pytest.mark.asyncio
#     async def test_invalid_action(self, user_tool):
#         """Test invalid action handling."""
#         parameters = {
#             "action": "invalid_action",
#             "user_id": "user123"
#         }

#         with pytest.raises(ValidationError):
#             await user_tool.execute(parameters)

#     @pytest.mark.asyncio
#     async def test_create_user_duplicate_username(self, user_tool):
#         """Test creating user with duplicate username."""
#         user_tool.user_service.create_user.side_effect = Exception("Username already exists")

#         parameters = {
#             "action": "create",
#             "username": "existinguser",
#             "email": "existing@example.com",
#             "password": "securepass123"
#         }

#         result = await user_tool.execute(parameters)

#         assert result["status"] == "error"
#         assert "Username already exists" in result["error"]

#     @pytest.mark.asyncio
#     async def test_get_nonexistent_user(self, user_tool):
#         """Test getting non-existent user."""
#         user_tool.user_service.get_user.side_effect = Exception("User not found")

#         parameters = {
#             "action": "get",
#             "user_id": "nonexistent"
#         }

#         result = await user_tool.execute(parameters)

#         assert result["status"] == "error"
#         assert "User not found" in result["error"]

#     @pytest.mark.parametrize("action,required_params", [
#         ("create", ["username", "email", "password"]),
#         ("get", ["user_id"]),
#         ("update", ["user_id"]),
#         ("delete", ["user_id"]),
#     ])
#     @pytest.mark.asyncio
#     async def test_required_parameters(self, user_tool, action, required_params):
#         """Test required parameters for different actions."""
#         parameters = {"action": action}
        
#         # Test with missing required parameters
#         with pytest.raises(ValidationError):
#             await user_tool.execute(parameters)

#         # Test with all required parameters
#         for param in required_params:
#             if param == "user_id":
#                 parameters[param] = "user123"
#             elif param == "username":
#                 parameters[param] = "testuser"
#             elif param == "email":
#                 parameters[param] = "test@example.com"
#             elif param == "password":
#                 parameters[param] = "testpass123"

#         # This should not raise an exception
#         result = await user_tool.execute(parameters)
#         assert result["status"] in ["success", "error"]
