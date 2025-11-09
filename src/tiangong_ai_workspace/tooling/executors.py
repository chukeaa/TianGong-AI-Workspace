"""
Execution utilities that expose controlled shell and Python runtimes.

These helpers allow higher-level agents (Codex, LangChain, workspace agent, etc.) to
reliably run commands while capturing structured outputs that can be surfaced to
callers or persisted for auditing.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

__all__ = [
    "PythonExecutionResult",
    "PythonExecutor",
    "ShellExecutionResult",
    "ShellExecutor",
]


@dataclass(slots=True)
class ShellExecutionResult:
    """Structured result for shell command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    cwd: Path

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "cwd": str(self.cwd),
        }


@dataclass(slots=True)
class ShellExecutor:
    """
    Small wrapper around `subprocess.run` with predictable output structure.
    """

    workdir: Path = Path.cwd()

    def run(self, command: str, *, timeout: Optional[int] = None) -> ShellExecutionResult:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.workdir,
            timeout=timeout,
        )
        return ShellExecutionResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cwd=self.workdir,
        )


@dataclass(slots=True)
class PythonExecutionResult:
    """Structured result for arbitrary Python code execution."""

    code: str
    stdout: str
    stderr: str
    globals_used: Mapping[str, Any]

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class PythonExecutor:
    """
    Executes Python source strings in a constrained namespace.

    The executor captures stdout / stderr and surfaces them alongside any
    exceptions. Agents can reuse a single executor instance to preserve global
    state between invocations if desired.
    """

    def __init__(self, *, shared_globals: Optional[MutableMapping[str, Any]] = None) -> None:
        self._globals: MutableMapping[str, Any] = shared_globals or {"__name__": "__agent_exec__"}

    def run(self, code: str, *, locals_override: Optional[MutableMapping[str, Any]] = None) -> PythonExecutionResult:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        local_vars: MutableMapping[str, Any] = locals_override or {}

        compiled_code = compile(code, "<agent-python>", "exec")

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                exec(compiled_code, self._globals, local_vars)
            except Exception:  # pragma: no cover - error text captured for caller
                traceback.print_exc(file=stderr_buffer)

        stdout_value = stdout_buffer.getvalue()
        stderr_value = stderr_buffer.getvalue()
        return PythonExecutionResult(
            code=textwrap.dedent(code),
            stdout=stdout_value,
            stderr=stderr_value,
            globals_used=dict(self._globals),
        )
