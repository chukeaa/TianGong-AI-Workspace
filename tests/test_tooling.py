from __future__ import annotations

import json

import pytest

from tiangong_ai_workspace.secrets import MCPServerSecrets, Secrets
from tiangong_ai_workspace.tooling import WorkspaceResponse, list_registered_tools
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
