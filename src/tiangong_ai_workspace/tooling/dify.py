"""Dify knowledge base client with structured responses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

import httpx

from ..secrets import DifyKnowledgeBaseSecrets, Secrets, load_secrets

LOGGER = logging.getLogger(__name__)

__all__ = ["DifyKnowledgeBaseClient", "DifyKnowledgeBaseError"]


class DifyKnowledgeBaseError(RuntimeError):
    """Raised when the Dify knowledge base API request fails."""


@dataclass(slots=True)
class DifyKnowledgeBaseClient:
    """Lightweight wrapper around the Dify dataset retrieval API."""

    secrets: Optional[Secrets] = None
    timeout: float = 15.0
    http_client: Optional[httpx.Client] = None
    _config: DifyKnowledgeBaseSecrets = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loaded = self.secrets or load_secrets()
        config = loaded.dify_knowledge_base
        if config is None:
            raise DifyKnowledgeBaseError("Dify knowledge base secrets are not configured.")
        object.__setattr__(self, "secrets", loaded)
        object.__setattr__(self, "_config", config)

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Retrieve knowledge chunks for the given query."""

        if not query.strip():
            raise DifyKnowledgeBaseError("Query cannot be empty.")

        payload: MutableMapping[str, Any] = {"query": query}
        if top_k is not None:
            if top_k <= 0:
                raise DifyKnowledgeBaseError("top_k must be greater than zero.")
            payload["top_k"] = int(top_k)
        if options:
            payload.update(dict(options))

        url = f"{self._config.api_base_url}/datasets/{self._config.dataset_id}/retrieve"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        LOGGER.debug("Calling Dify knowledge base: %s", url)
        try:
            response = self._post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.exception("Dify knowledge base request failed")
            raise DifyKnowledgeBaseError(f"HTTP error querying Dify knowledge base: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise DifyKnowledgeBaseError("Dify knowledge base returned invalid JSON.") from exc

        return {
            "query": query,
            "result": data,
        }

    def _post(self, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, headers=headers, json=json, timeout=self.timeout)
        return httpx.post(url, headers=headers, json=json, timeout=self.timeout)
