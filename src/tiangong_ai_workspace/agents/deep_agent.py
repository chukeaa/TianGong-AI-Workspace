"""
Opinionated DeepAgents factory for the TianGong AI Workspace.

This module builds a Deep Agent that is aware of the local shell, Python
runtime, Tavily MCP search, and LangGraph-powered document workflows. The
resulting agent can orchestrate complex tasks without bespoke wiring for each
project.
"""

from __future__ import annotations

from typing import Any, List, MutableSequence, Sequence

from deepagents import create_deep_agent
from langchain_core.language_models import BaseLanguageModel

from ..tooling.executors import PythonExecutor, ShellExecutor
from ..tooling.llm import OpenAIModelFactory
from ..tooling.tavily import TavilySearchClient, TavilySearchError
from .tools import create_document_tool, create_python_tool, create_shell_tool, create_tavily_tool

DEFAULT_SYSTEM_PROMPT = """\
You are the TianGong Workspace Deep Agent.
- Plan multi-step tasks, maintain checklists, and use specialists when helpful.
- Use shell commands for filesystem inspection, package tooling, and running CLIs.
- Use Python for data processing, plotting (matplotlib, seaborn), and quick experimentation.
- Use Tavily search to gather fresh context when required.
- Use the document generator to produce reports, plans, patent disclosures, and proposals.
- Prefer structured outputs that downstream agents can parse easily.
"""

__all__ = ["build_workspace_deep_agent"]


def build_workspace_deep_agent(
    *,
    model: BaseLanguageModel | str | None = None,
    include_shell: bool = True,
    include_python: bool = True,
    include_tavily: bool = True,
    include_document_agent: bool = True,
    extra_tools: Sequence[Any] | None = None,
    subagents: Sequence[Any] | None = None,
    system_prompt: str | None = None,
) -> Any:
    """
    Construct a Deep Agent preloaded with workspace-aware tools.
    """

    if model is None:
        factory = OpenAIModelFactory()
        agent_model = factory.create_chat_model(purpose="deep_research", temperature=0.2)
    else:
        agent_model = model

    tools: MutableSequence[Any] = []

    if include_shell:
        shell_tool = create_shell_tool(ShellExecutor())
        tools.append(shell_tool)
    else:
        shell_tool = None

    if include_python:
        python_tool = create_python_tool(PythonExecutor())
        tools.append(python_tool)
    else:
        python_tool = None

    tavily_tool = None
    if include_tavily:
        try:
            tavily_tool = create_tavily_tool(TavilySearchClient())
            tools.append(tavily_tool)
        except TavilySearchError:
            tavily_tool = None  # Secrets missing; silently omit

    document_tool = None
    if include_document_agent:
        document_tool = create_document_tool()
        tools.append(document_tool)

    if extra_tools:
        tools.extend(extra_tools)

    default_subagents: List[Any] = []
    if document_tool is not None:
        default_subagents.append(
            {
                "name": "document-specialist",
                "description": "Generates reports, plans, patent disclosures, and proposals.",
                "system_prompt": "You are a documentation specialist. Always return polished markdown drafts.",
                "tools": [document_tool],
                "model": agent_model,
            }
        )
    if tavily_tool is not None:
        default_subagents.append(
            {
                "name": "research-specialist",
                "description": "Performs deep internet research via Tavily.",
                "system_prompt": "You gather current data via Tavily search and summarise findings.",
                "tools": [tavily_tool],
                "model": agent_model,
            }
        )
    if python_tool is not None or shell_tool is not None:
        specialist_tools = [tool for tool in (python_tool, shell_tool) if tool is not None]
        default_subagents.append(
            {
                "name": "execution-specialist",
                "description": "Runs Python blocks and shell commands safely.",
                "system_prompt": "Execute code snippets and commands. Return stdout/stderr and summarise key results.",
                "tools": specialist_tools,
                "model": agent_model,
            }
        )

    system_prompt_value = system_prompt or DEFAULT_SYSTEM_PROMPT
    agent = create_deep_agent(
        tools=list(tools),
        system_prompt=system_prompt_value,
        model=agent_model,
        subagents=list(default_subagents) + list(subagents or []),
    )
    return agent
