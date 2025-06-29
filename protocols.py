"""
Protocol definitions for server communication in HeteroSpotLLMServe.

This module defines the communication protocols used between different components
of the system, including TensorStore servers and API servers.
"""

from enum import Enum


class TensorStoreRequest(Enum):
    """Request commands for TensorStore server."""
    STATUS_CHECK = b'0'  # Check if server is ready
    SHUTDOWN = b'1'      # Request server shutdown
    
    
class TensorStoreResponse(Enum):
    """Response codes from TensorStore server."""
    NOT_READY = b'0'     # Server is not ready (weights not loaded)
    READY = b'1'         # Server is ready (weights loaded)
    OK = b'2'            # Command accepted successfully
    ERROR = b'3'         # Unknown command or error
    

class APIServerHTTPResponse(Enum):
    """HTTP response messages from API server."""
    SHUTDOWN_ACCEPTED = {"status": "shutdown_accepted", "message": "Server will shutdown gracefully"}