"""
In-process tool registry for agent discoverability.

The registry keeps a lightweight catalogue of workflows and integrations so that
agents (and humans) can quickly inspect what the workspace exposes without
reading source code. CLI commands use this module for structured listings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Tuple

__all__ = ["ToolDescriptor", "register_tool", "list_registered_tools"]


@dataclass(slots=True, frozen=True)
class ToolDescriptor:
    """Metadata describing an agent-facing workflow or integration."""

    name: str
    description: str
    category: str
    entrypoint: str
    tags: Tuple[str, ...] = ()


_TOOL_REGISTRY: MutableMapping[str, ToolDescriptor] = {}


def register_tool(descriptor: ToolDescriptor) -> None:
    """Register a tool descriptor, replacing any existing entry with the same name."""
    _TOOL_REGISTRY[descriptor.name] = descriptor


def register_many(descriptors: Iterable[ToolDescriptor]) -> None:
    """Bulk register multiple descriptors."""
    for descriptor in descriptors:
        register_tool(descriptor)


def list_registered_tools() -> Mapping[str, ToolDescriptor]:
    """Return an immutable view of the current tool registry."""
    return dict(_TOOL_REGISTRY)


# Pre-populate with the core document-oriented workflows and research utilities.
register_many(
    [
        ToolDescriptor(
            name="docs.report",
            description="Generate structured business or technical reports using the document workflow agent.",
            category="workflow",
            entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
            tags=("document", "report"),
        ),
        ToolDescriptor(
            name="docs.patent_disclosure",
            description="Draft patent disclosure sheets with optional web research.",
            category="workflow",
            entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
            tags=("document", "patent"),
        ),
        ToolDescriptor(
            name="docs.plan",
            description="Create project or execution plans with milestones and resource breakdowns.",
            category="workflow",
            entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
            tags=("document", "planning"),
        ),
        ToolDescriptor(
            name="docs.project_proposal",
            description="Prepare project proposal drafts optimised for internal reviews.",
            category="workflow",
            entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
            tags=("document", "proposal"),
        ),
        ToolDescriptor(
            name="research.tavily",
            description="Query the Tavily MCP service for live internet research.",
            category="integration",
            entrypoint="tiangong_ai_workspace.tooling.tavily.TavilySearchClient.search",
            tags=("research", "search"),
        ),
        ToolDescriptor(
            name="agents.deep",
            description="Workspace autonomous agent built with LangGraph (shell, Python, Tavily, document workflows).",
            category="agent",
            entrypoint="tiangong_ai_workspace.agents.deep_agent.build_workspace_deep_agent",
            tags=("langgraph", "planner"),
        ),
        ToolDescriptor(
            name="runtime.shell",
            description="Shell executor that returns structured stdout/stderr for commands.",
            category="runtime",
            entrypoint="tiangong_ai_workspace.tooling.executors.ShellExecutor.run",
            tags=("shell", "commands"),
        ),
        ToolDescriptor(
            name="runtime.python",
            description="Python executor for dynamic scripting with captured stdout/stderr.",
            category="runtime",
            entrypoint="tiangong_ai_workspace.tooling.executors.PythonExecutor.run",
            tags=("python", "scripting"),
        ),
    ]
)
