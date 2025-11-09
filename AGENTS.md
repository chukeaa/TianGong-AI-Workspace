# TianGong AI Workspace — Agent Guide

## Overview
- Unified developer workspace for coordinating Codex, Gemini, Claude Code, and document-centric AI workflows.
- Python 3.12+ project managed完全 by `uv`; avoid `pip`, `poetry`, `conda`.
- Primary entry point: `uv run tiangong-workspace`, featuring LangChain/LangGraph document agents and Tavily MCP research.

## Repository Layout
- `src/tiangong_ai_workspace/cli.py`: Typer CLI with `docs`, `research`, and `mcp` subcommands plus structured JSON output support.
- `src/tiangong_ai_workspace/agents/`: LangChain/LangGraph document workflows (`run_document_workflow`, templates for reports, plans, patent, proposals).
- `src/tiangong_ai_workspace/tooling/`: Utilities shared by agents.
  - `responses.py`: `WorkspaceResponse` envelope for deterministic outputs.
  - `registry.py`: Tool metadata registry surfaced via `tiangong-workspace tools --catalog`.
  - `llm.py`: OpenAI model factory (selects chat vs deep research models).
  - `tavily.py`: Tavily MCP client with retry + structured payloads.
- `src/tiangong_ai_workspace/templates/`: Markdown scaffolds referenced by workflows.
- `.sercrets/secrets.toml`: Local-only secrets (copy from `.sercrets/secrets.example.toml`).

## Tooling Workflow
Run everything through `uv`:

```bash
uv sync
uv run tiangong-workspace --help
```

After **every** code change run, in order:

```bash
uv run black .
uv run ruff check
uv run pytest
```

All three must pass before sharing updates.

## CLI Quick Reference
- `uv run tiangong-workspace info` — workspace summary.
- `uv run tiangong-workspace check` — validate Python/uv/Node + registered CLIs.
- `uv run tiangong-workspace tools --catalog` — list internal agent workflows from the registry.
- `uv run tiangong-workspace docs list` — supported document workflows.
- `uv run tiangong-workspace docs run <workflow> --topic ...` — generate drafts (supports `--json`, `--skip-research`, `--purpose`, etc.).
- `uv run tiangong-workspace research "<query>"` — invoke Tavily MCP search (also supports `--json`).
- `uv run tiangong-workspace mcp services|tools|invoke` — inspect and call configured MCP services.

Use `--json` for machine-readable responses suitable for chaining agents.

## Secrets
- Populate `.sercrets/secrets.toml` using the example file.
- Required: `openai.api_key`. Optional: `model`, `chat_model`, `deep_research_model`.
- Tavily section needs `service_name`, `url`, and `api_key` (`Authorization: Bearer` header).
- Secrets stay local; never commit `.sercrets/`.

## Maintenance Rules
- Modify program code → update both `AGENTS.md` and `README.md`.
- Respect dependency declarations in `pyproject.toml`; use `uv add/remove`.
- Prefer ASCII in source files unless the file already uses other encodings.
- Structured outputs (`WorkspaceResponse`) keep agent integrations predictable—adhere to them when adding new commands.

## Helpful Notes
- To stub LLM calls in tests, inject a custom `Runnable` when calling `run_document_workflow`.
- Tavily wrapper retries transient failures; propagate explicit `TavilySearchError` for agents to handle.
- Register new workflows via `tooling.registry.register_tool` for discoverability.
- Keep logs redaction-aware if adding persistence; avoid leaking API keys.

