# src/mcp_server/session_manager.py

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set
from weakref import WeakSet

from .config import MCPConfig
from .error_handlers import SessionError

logger = logging.getLogger(__name__)

@dataclass
class MCPSession:
    """Represents an MCP session with metadata and state."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    client_info: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    tool_results_cache: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def update_access_time(self):
        """Update the last accessed timestamp."""
        self.last_accessed = datetime.utcnow()
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if the session has expired."""
        expiry_time = self.last_accessed + timedelta(seconds=timeout_seconds)
        return datetime.utcnow() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "client_info": self.client_info,
            "context": self.context,
            "is_active": self.is_active,
            "cache_size": len(self.tool_results_cache)
        }

class SessionManager:
    """
    Manages sessions for the MCP server with automatic cleanup and monitoring.
    """
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.sessions: Dict[str, MCPSession] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.stats = {
            "total_sessions_created": 0,
            "total_sessions_expired": 0,
            "total_sessions_deleted": 0,
            "active_sessions": 0
        }
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the automatic cleanup task."""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def _cleanup_expired_sessions(self):
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.config.session_cleanup_interval)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    async def create_session(self, client_info: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session and return the session ID."""
        if len(self.sessions) >= self.config.max_sessions:
            # Clean up expired sessions first
            await self.cleanup_expired_sessions()
            
            # If still at max capacity, raise error
            if len(self.sessions) >= self.config.max_sessions:
                raise SessionError("", "Maximum number of sessions reached")
        
        session_id = str(uuid.uuid4())
        session = MCPSession(
            session_id=session_id,
            client_info=client_info or {}
        )
        
        self.sessions[session_id] = session
        self.stats["total_sessions_created"] += 1
        self.stats["active_sessions"] = len(self.sessions)
        
        logger.info(f"Created session {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[MCPSession]:
        """Get a session by ID and update its access time."""
        session = self.sessions.get(session_id)
        if session:
            if session.is_expired(self.config.session_timeout):
                await self.delete_session(session_id)
                return None
            
            session.update_access_time()
            return session
        return None
    
    async def update_session_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update session context data."""
        session = await self.get_session(session_id)
        if session:
            session.context.update(context)
            logger.debug(f"Updated context for session {session_id}")
            return True
        return False
    
    async def cache_tool_result(self, session_id: str, tool_name: str, 
                               args_hash: str, result: Any) -> bool:
        """Cache a tool result in the session."""
        session = await self.get_session(session_id)
        if session:
            cache_key = f"{tool_name}:{args_hash}"
            session.tool_results_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Limit cache size per session
            if len(session.tool_results_cache) > 100:
                # Remove oldest entries
                sorted_items = sorted(
                    session.tool_results_cache.items(),
                    key=lambda x: x[1]["timestamp"]
                )
                # Keep only the 50 most recent
                session.tool_results_cache = dict(sorted_items[-50:])
            
            return True
        return False
    
    async def get_cached_tool_result(self, session_id: str, tool_name: str, 
                                    args_hash: str) -> Optional[Any]:
        """Get a cached tool result from the session."""
        session = await self.get_session(session_id)
        if session:
            cache_key = f"{tool_name}:{args_hash}"
            cached_item = session.tool_results_cache.get(cache_key)
            if cached_item:
                # Check if cache entry is not too old (1 hour)
                cached_time = datetime.fromisoformat(cached_item["timestamp"])
                if datetime.utcnow() - cached_time < timedelta(hours=1):
                    return cached_item["result"]
                else:
                    # Remove expired cache entry
                    del session.tool_results_cache[cache_key]
        return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats["total_sessions_deleted"] += 1
            self.stats["active_sessions"] = len(self.sessions)
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up all expired sessions and return the number cleaned up."""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(self.config.session_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        if expired_sessions:
            self.stats["total_sessions_expired"] += len(expired_sessions)
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    async def list_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active sessions with their metadata."""
        result = {}
        for session_id, session in self.sessions.items():
            if not session.is_expired(self.config.session_timeout):
                result[session_id] = session.to_dict()
        return result
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        return {
            **self.stats,
            "current_active_sessions": len(self.sessions),
            "cleanup_interval": self.config.session_cleanup_interval,
            "session_timeout": self.config.session_timeout,
            "max_sessions": self.config.max_sessions
        }
    
    async def shutdown(self):
        """Shutdown the session manager and cleanup tasks."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all sessions
        self.sessions.clear()
        logger.info("Session manager shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            return True
        return False
