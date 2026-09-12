"""Generic, dependency-free execution boundary for external governance layers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolExecutor


@dataclass(slots=True)
class ExecutionBoundaryDecision:
    """Decision returned immediately before ToolExecutor dispatches a tool call."""

    allow: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionBoundary(Protocol):
    """Minimal contract implemented by an external consequence boundary."""

    def before_execute(
        self,
        *,
        agent_id: str,
        tool_call: ToolCall,
        params: Dict[str, Any],
        tool: BaseTool,
    ) -> ExecutionBoundaryDecision: ...

    def after_execute(
        self,
        *,
        agent_id: str,
        tool_call: ToolCall,
        params: Dict[str, Any],
        tool: BaseTool,
        result: ToolResult,
    ) -> None: ...


class ExecutionBoundaryToolExecutor(ToolExecutor):
    """ToolExecutor with an optional fail-closed external execution boundary.

    With no boundary configured this is behaviorally identical to ToolExecutor.
    A configured boundary can deny a call before any tool effect occurs and can
    observe the resulting ToolResult for lineage, continuity, or evidence.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        bus=None,
        *,
        execution_boundary: Optional[ExecutionBoundary] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(tools, bus, **kwargs)
        self._execution_boundary = execution_boundary

    def execute(self, tool_call: ToolCall) -> ToolResult:
        boundary = self._execution_boundary
        if boundary is None:
            return super().execute(tool_call)

        tool = self._tools.get(tool_call.name)
        if tool is None:
            return super().execute(tool_call)

        try:
            params = json.loads(tool_call.arguments) if tool_call.arguments else {}
        except json.JSONDecodeError:
            return super().execute(tool_call)
        if not isinstance(params, dict):
            return super().execute(tool_call)

        before = getattr(boundary, "before_execute", None)
        if callable(before):
            try:
                decision = before(
                    agent_id=self._agent_id,
                    tool_call=tool_call,
                    params=dict(params),
                    tool=tool,
                )
            except Exception as exc:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=f"Execution boundary failed closed: {exc}",
                    success=False,
                    metadata={"execution_boundary": "error"},
                )

            if not isinstance(decision, ExecutionBoundaryDecision):
                return ToolResult(
                    tool_name=tool_call.name,
                    content="Execution boundary failed closed: invalid decision.",
                    success=False,
                    metadata={"execution_boundary": "invalid_decision"},
                )
            if not decision.allow:
                return ToolResult(
                    tool_name=tool_call.name,
                    content=decision.reason or "Execution denied by boundary.",
                    success=False,
                    metadata={
                        "execution_boundary": "denied",
                        **decision.metadata,
                    },
                )

        result = super().execute(tool_call)

        after = getattr(boundary, "after_execute", None)
        if callable(after):
            try:
                after(
                    agent_id=self._agent_id,
                    tool_call=tool_call,
                    params=dict(params),
                    tool=tool,
                    result=result,
                )
            except Exception as exc:
                # The effect may already have happened. Preserve the real result
                # and make the evidence failure explicit rather than pretending
                # execution failed or attempting an unsafe implicit rollback.
                result.metadata["execution_boundary_after_error"] = str(exc)

        return result


__all__ = [
    "ExecutionBoundary",
    "ExecutionBoundaryDecision",
    "ExecutionBoundaryToolExecutor",
]
