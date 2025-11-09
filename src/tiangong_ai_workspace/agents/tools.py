"""
LangChain tool definitions that expose workspace capabilities to agents.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_core.tools import tool

from ..tooling.executors import PythonExecutor, ShellExecutor
from ..tooling.tavily import TavilySearchClient, TavilySearchError
from .workflows import DocumentWorkflowConfig, DocumentWorkflowType, run_document_workflow

__all__ = [
    "create_document_tool",
    "create_python_tool",
    "create_shell_tool",
    "create_tavily_tool",
]


def create_shell_tool(executor: Optional[ShellExecutor] = None, *, name: str = "run_shell") -> Any:
    exec_instance = executor or ShellExecutor()

    @tool(name)
    def run_shell(command: str) -> Mapping[str, Any]:
        """Execute a shell command inside the workspace environment."""

        result = exec_instance.run(command)
        return result.to_dict()

    return run_shell


def create_python_tool(executor: Optional[PythonExecutor] = None, *, name: str = "run_python") -> Any:
    exec_instance = executor or PythonExecutor()

    @tool(name)
    def run_python(code: str) -> Mapping[str, Any]:
        """Execute Python code using the shared workspace interpreter."""

        result = exec_instance.run(code)
        return result.to_dict()

    return run_python


def create_tavily_tool(client: Optional[TavilySearchClient] = None, *, name: str = "tavily_search") -> Any:
    tavily_client = client or TavilySearchClient()

    @tool(name)
    def tavily_search(query: str) -> Mapping[str, Any]:
        """Search the internet using the configured Tavily MCP service."""

        try:
            result = tavily_client.search(query)
        except TavilySearchError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "success", "data": result}

    return tavily_search


def create_document_tool(*, name: str = "generate_document") -> Any:
    @tool(name)
    def generate_document(
        workflow: str,
        topic: str,
        instructions: str | None = None,
        audience: str | None = None,
        language: str = "zh",
        skip_research: bool = False,
    ) -> Mapping[str, Any]:
        """Generate a structured document using the LangGraph workflow."""

        try:
            workflow_type = DocumentWorkflowType(workflow)
        except ValueError:
            return {
                "status": "error",
                "message": f"Unsupported workflow '{workflow}'.",
            }

        config = DocumentWorkflowConfig(
            workflow=workflow_type,
            topic=topic,
            instructions=instructions,
            audience=audience,
            language=language,
            include_research=not skip_research,
        )
        result = run_document_workflow(config)
        return {"status": "success", "data": result}

    return generate_document
