from __future__ import annotations

from typing import Any, Iterable, List

from langchain_core.runnables import Runnable

from tiangong_ai_workspace.agents import DocumentWorkflowConfig, DocumentWorkflowType, run_document_workflow
from tiangong_ai_workspace.tooling.tavily import TavilySearchError


class SequentialLLM(Runnable):
    """Minimal Runnable that returns pre-seeded responses."""

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses: List[str] = list(responses)

    def invoke(self, _: Any, config: Any | None = None) -> str:  # type: ignore[override]
        if not self._responses:
            raise RuntimeError("SequentialLLM has no responses left")
        return self._responses.pop(0)


class FailingTavilyClient:
    def search(self, _: str, *, options: Any | None = None) -> Any:
        raise TavilySearchError("network unavailable")


def test_document_workflow_without_research() -> None:
    llm = SequentialLLM(["OUTLINE", "DRAFT"])
    config = DocumentWorkflowConfig(
        workflow=DocumentWorkflowType.REPORT,
        topic="测试主题",
        include_research=False,
        language="zh",
    )
    result = run_document_workflow(config, llm=llm)
    assert result["outline"] == "OUTLINE"
    assert result["draft"] == "DRAFT"
    assert result["research"] == []


def test_document_workflow_handles_research_failure() -> None:
    llm = SequentialLLM(["OUTLINE", "DRAFT"])
    config = DocumentWorkflowConfig(
        workflow=DocumentWorkflowType.REPORT,
        topic="测试主题",
        include_research=True,
    )
    result = run_document_workflow(config, llm=llm, tavily=FailingTavilyClient())  # type: ignore[arg-type]
    assert result["outline"] == "OUTLINE"
    assert result["draft"] == "DRAFT"
    assert result["research"][0]["summary"] == "Research step failed"
