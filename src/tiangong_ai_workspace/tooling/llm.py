"""
Model configuration helpers for LangChain / LangGraph workflows.

The module centralises how OpenAI credentials are loaded and how default model
names are selected for different workflow purposes. This avoids scattering API
key lookups across the codebase and makes it easier to swap providers later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from langchain_openai import ChatOpenAI

from ..secrets import OpenAISecrets, Secrets, load_secrets

__all__ = ["ModelPurpose", "OpenAIModelFactory"]

ModelPurpose = Literal["general", "deep_research", "creative"]


@dataclass(slots=True)
class OpenAIModelFactory:
    """Factory that builds LangChain `ChatOpenAI` instances from secrets."""

    secrets: Optional[Secrets] = None

    def __post_init__(self) -> None:
        self._secrets = self.secrets or load_secrets()
        if not self._secrets.openai:
            raise RuntimeError("OpenAI credentials are not configured. Populate `.sercrets/secrets.toml` based on the example file.")

    def _select_model(self, purpose: ModelPurpose) -> str:
        creds: OpenAISecrets = self._secrets.openai  # type: ignore[assignment]
        if purpose == "deep_research" and creds.deep_research_model:
            return creds.deep_research_model
        if purpose == "creative" and creds.chat_model:
            return creds.chat_model
        if creds.chat_model:
            return creds.chat_model
        if creds.model:
            return creds.model
        # Default to the deep research model if available; otherwise fallback to a versatile small model.
        if creds.deep_research_model:
            return creds.deep_research_model
        return "o4-mini-deep-research"

    def create_chat_model(
        self,
        *,
        purpose: ModelPurpose = "general",
        temperature: float = 0.4,
        timeout: int | None = None,
        model_override: str | None = None,
    ) -> ChatOpenAI:
        """
        Construct a configured `ChatOpenAI` client with the requested purpose.

        Parameters
        ----------
        purpose:
            Workflow purpose hint used to decide which model name to select.
        temperature:
            Sampling temperature; defaults to a stable 0.4 suitable for drafting.
        timeout:
            Optional timeout (seconds) forwarded to the underlying client.
        """

        model_name = model_override or self._select_model(purpose)
        creds: OpenAISecrets = self._secrets.openai  # type: ignore[assignment]
        return ChatOpenAI(
            api_key=creds.api_key,
            model=model_name,
            temperature=temperature,
            timeout=timeout,
        )
