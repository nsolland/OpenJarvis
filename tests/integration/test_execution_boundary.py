from __future__ import annotations

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.integration.execution_boundary import (
    ExecutionBoundaryDecision,
    ExecutionBoundaryToolExecutor,
)
from openjarvis.tools._stubs import BaseTool, ToolSpec


class _CountingTool(BaseTool):
    tool_id = "counting"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="counting",
            description="Count executions.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        )

    def execute(self, **params) -> ToolResult:
        self.calls += 1
        return ToolResult(
            tool_name="counting",
            content=params.get("value", ""),
            success=True,
        )


class _Boundary:
    def __init__(self, allow: bool = True) -> None:
        self.allow = allow
        self.before = []
        self.after = []

    def before_execute(self, **kwargs):
        self.before.append(kwargs)
        return ExecutionBoundaryDecision(
            allow=self.allow,
            reason="policy denied" if not self.allow else "",
            metadata={"decision_id": "d-1"},
        )

    def after_execute(self, **kwargs):
        self.after.append(kwargs)


def test_no_boundary_preserves_tool_executor_behavior():
    tool = _CountingTool()
    executor = ExecutionBoundaryToolExecutor([tool])

    result = executor.execute(
        ToolCall(id="1", name="counting", arguments='{"value":"ok"}')
    )

    assert result.success is True
    assert result.content == "ok"
    assert tool.calls == 1


def test_boundary_can_deny_before_effect():
    tool = _CountingTool()
    boundary = _Boundary(allow=False)
    executor = ExecutionBoundaryToolExecutor(
        [tool], execution_boundary=boundary, agent_id="agent-7"
    )

    result = executor.execute(
        ToolCall(id="2", name="counting", arguments='{"value":"blocked"}')
    )

    assert result.success is False
    assert result.content == "policy denied"
    assert result.metadata["execution_boundary"] == "denied"
    assert result.metadata["decision_id"] == "d-1"
    assert tool.calls == 0
    assert boundary.before[0]["agent_id"] == "agent-7"
    assert boundary.after == []


def test_boundary_observes_verified_result_after_effect():
    tool = _CountingTool()
    boundary = _Boundary(allow=True)
    executor = ExecutionBoundaryToolExecutor(
        [tool], execution_boundary=boundary, agent_id="agent-8"
    )

    result = executor.execute(
        ToolCall(id="3", name="counting", arguments='{"value":"done"}')
    )

    assert result.success is True
    assert tool.calls == 1
    assert boundary.after[0]["result"] is result
    assert boundary.after[0]["params"] == {"value": "done"}


def test_boundary_exception_fails_closed_before_effect():
    class BrokenBoundary:
        def before_execute(self, **kwargs):
            raise RuntimeError("unavailable")

    tool = _CountingTool()
    executor = ExecutionBoundaryToolExecutor(
        [tool], execution_boundary=BrokenBoundary()
    )

    result = executor.execute(ToolCall(id="4", name="counting", arguments="{}"))

    assert result.success is False
    assert "failed closed" in result.content
    assert tool.calls == 0


def test_after_failure_does_not_rewrite_actual_effect_result():
    class BrokenEvidenceBoundary:
        def before_execute(self, **kwargs):
            return ExecutionBoundaryDecision(allow=True)

        def after_execute(self, **kwargs):
            raise RuntimeError("evidence sink unavailable")

    tool = _CountingTool()
    executor = ExecutionBoundaryToolExecutor(
        [tool], execution_boundary=BrokenEvidenceBoundary()
    )

    result = executor.execute(ToolCall(id="5", name="counting", arguments="{}"))

    assert result.success is True
    assert tool.calls == 1
    assert result.metadata["execution_boundary_after_error"] == (
        "evidence sink unavailable"
    )
