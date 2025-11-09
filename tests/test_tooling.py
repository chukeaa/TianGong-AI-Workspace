from __future__ import annotations

import json
from typing import Any

import pytest

from tiangong_ai_workspace.agents.deep_agent import build_workspace_deep_agent
from tiangong_ai_workspace.secrets import MCPServerSecrets, Secrets
from tiangong_ai_workspace.tooling import PythonExecutor, ShellExecutor, WorkspaceResponse, list_registered_tools
from tiangong_ai_workspace.tooling.tavily import TavilySearchClient, TavilySearchError


def test_workspace_response_json_roundtrip() -> None:
    response = WorkspaceResponse.ok(payload={"value": 42}, message="All good", request_id="abc123")
    payload = json.loads(response.to_json())
    assert payload["status"] == "success"
    assert payload["payload"] == {"value": 42}
    assert payload["metadata"]["request_id"] == "abc123"


def test_tool_registry_contains_core_workflows() -> None:
    registry = list_registered_tools()
    assert "docs.report" in registry
    assert registry["docs.report"].category == "workflow"
    assert "agents.deep" in registry
    assert "runtime.shell" in registry
    assert "runtime.python" in registry


def test_tavily_client_missing_service_raises() -> None:
    secrets = Secrets(openai=None, mcp_servers={})
    with pytest.raises(TavilySearchError):
        TavilySearchClient(secrets=secrets)


def test_tavily_client_custom_service_is_loaded() -> None:
    secrets = Secrets(
        openai=None,
        mcp_servers={
            "custom": MCPServerSecrets(
                service_name="custom",
                transport="streamable_http",
                url="https://example.com",
            )
        },
    )
    client = TavilySearchClient(secrets=secrets, service_name="custom")
    assert client.service_name == "custom"


def test_shell_executor_runs_command() -> None:
    executor = ShellExecutor()
    result = executor.run("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout.lower()


def test_python_executor_captures_output() -> None:
    executor = PythonExecutor()
    result = executor.run("print('hi')")
    assert "hi" in result.stdout
    assert result.stderr == ""


def test_build_workspace_deep_agent_invokes_create_deep_agent(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(*, tools: Any, system_prompt: str, model: Any, subagents: Any):
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["model"] = model
        captured["subagents"] = subagents
        return {"agent": "ok"}

    monkeypatch.setattr("tiangong_ai_workspace.agents.deep_agent.create_deep_agent", fake_create_deep_agent)

    agent = build_workspace_deep_agent(model="fake-model", include_tavily=False)
    assert agent == {"agent": "ok"}
    assert captured["tools"]
    assert captured["subagents"]
