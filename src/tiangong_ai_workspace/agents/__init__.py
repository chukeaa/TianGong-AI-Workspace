"""
LangChain / LangGraph agent workflows for Tiangong AI Workspace.

The package currently focuses on document-oriented workflows tailored to
day-to-day paperwork such as reports, patent disclosures, plans, and proposals.
"""

from .workflows import DocumentWorkflowConfig, DocumentWorkflowType, run_document_workflow

__all__ = ["DocumentWorkflowConfig", "DocumentWorkflowType", "run_document_workflow"]
