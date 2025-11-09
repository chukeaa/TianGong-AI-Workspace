"""
Wrapper around the Tavily MCP service for resilient web research.

The implementation relies on the existing :class:`MCPToolClient` to communicate
with remote MCP servers while adding retry logic and structured responses so
agents can consume results predictably.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..mcp_client import MCPToolClient
from ..secrets import MCPServerSecrets, Secrets, load_secrets

LOGGER = logging.getLogger(__name__)

__all__ = ["TavilySearchClient", "TavilySearchError"]


class TavilySearchError(RuntimeError):
    """Raised when the Tavily search integration fails."""


@dataclass(slots=True)
class TavilySearchClient:
    """
    Helper around the Tavily MCP tooling.

    Parameters
    ----------
    secrets:
        Optional :class:`Secrets` instance. When omitted the client's constructor
        loads secrets from disk using :func:`load_secrets`.
    service_name:
        Name of the MCP service entry; defaults to ``"tavily"`` which matches
        `.sercrets/secrets.example.toml`.
    tool_name:
        The tool name exposed by the Tavily MCP provider. ``"search"`` is used
        by default and can be overridden if the remote is configured differently.
    """

    secrets: Optional[Secrets] = None
    service_name: str = "tavily"
    tool_name: str = "search"
    _service_registry: MutableMapping[str, MCPServerSecrets] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loaded = self.secrets or load_secrets()
        object.__setattr__(self, "secrets", loaded)
        config = self._resolve_config(loaded)
        object.__setattr__(self, "_service_registry", {self.service_name: config})

    def _resolve_config(self, secrets: Secrets) -> MCPServerSecrets:
        configs = secrets.mcp_servers
        if self.service_name not in configs:
            available = ", ".join(sorted(configs)) or "none"
            raise TavilySearchError(f"MCP service '{self.service_name}' is not configured. Available services: {available}.")
        return configs[self.service_name]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=12),
        retry=retry_if_exception_type(TavilySearchError),
    )
    def search(self, query: str, *, options: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """
        Execute a Tavily search request.

        Parameters
        ----------
        query:
            Natural language query string.
        options:
            Optional dictionary forwarding additional parameters to the MCP tool.
        """

        payload: MutableMapping[str, Any] = {"query": query}
        if options:
            payload.update(dict(options))

        LOGGER.debug("Invoking Tavily MCP search with payload: %s", payload)
        try:
            with MCPToolClient(self._service_registry) as client:
                result, attachments = client.invoke_tool(self.service_name, self.tool_name, payload)
        except Exception as exc:  # pragma: no cover - the wrapper converts to a typed error
            LOGGER.exception("Tavily MCP tool invocation failed")
            raise TavilySearchError(str(exc)) from exc

        LOGGER.debug("Tavily MCP search succeeded")
        response: MutableMapping[str, Any] = {
            "query": query,
            "result": result,
        }
        if attachments:
            response["attachments"] = attachments
        return response
