"""
Utility helpers for Tiangong AI Workspace tooling.

This package exposes lightly opinionated building blocks such as response
schemas, tool registries, and external service wrappers that agents can reuse.
"""

from .registry import ToolDescriptor, list_registered_tools
from .responses import ResponsePayload, WorkspaceResponse

__all__ = [
    "ResponsePayload",
    "WorkspaceResponse",
    "ToolDescriptor",
    "list_registered_tools",
]
