from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import react_agent as agent  # noqa: E402


def hub_message(seq: int, sender: str, content: str) -> dict[str, object]:
    return {
        "seq": seq,
        "agent_name": sender,
        "timestamp": f"2026-05-27T13:{seq:02d}:00Z",
        "content": content,
    }


def broad_calculator_request(seq: int = 1) -> dict[str, object]:
    return hub_message(
        seq,
        "Emil (human)",
        "all agents, create a simple calculator with basic +, -, *, / functions.",
    )


def coordinator_request(seq: int = 1) -> dict[str, object]:
    return hub_message(
        seq,
        "Igor (human)",
        (
            "emil-flyghed-agent, You are the manager/coordinator for this task; "
            "build a simple calculator supporting +, -, *, and / with no unnecessary features, "
            "delegate implementation, testing, code review, bug checking, and documentation tasks "
            "across all other agents, collect status reports, verify tests pass, and provide final run instructions."
        ),
    )


class HubBehaviorTests(unittest.TestCase):
    def test_broad_build_deferral_answer_is_rejected(self) -> None:
        answer = (
            "I can start drafting the calculator functions if assigned. "
            "Please confirm task assignments so we avoid duplication."
        )

        reason = agent.hub_contribution_guard_reason(answer, [], [broad_calculator_request()])

        self.assertEqual(reason, "broad build request received a deferral or status-only answer")

    def test_incomplete_calculator_code_without_divide_is_rejected(self) -> None:
        answer = """Here are the files:

# calculator.py
```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
```
"""

        reason = agent.hub_contribution_guard_reason(answer, [], [broad_calculator_request()])

        self.assertEqual(reason, "calculator implementation omitted a requested operation")

    def test_complete_calculator_code_is_concrete_contribution(self) -> None:
        answer = """I drafted the calculator implementation:

# file: calculator.py
```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```
"""

        reason = agent.hub_contribution_guard_reason(answer, [], [broad_calculator_request()])

        self.assertIsNone(reason)

    def test_status_after_another_complete_artifact_is_rejected(self) -> None:
        history = [
            broad_calculator_request(1),
            hub_message(
                2,
                "other-agent",
                """Done.

# file: calculator.py
```python
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return a / b
```
""",
            ),
        ]
        answer = "The calculator.py and test_calculator.py are complete and tested. Please assign a clear next task."

        reason = agent.hub_contribution_guard_reason(answer, history, [])

        self.assertEqual(reason, "broad build request received a deferral or status-only answer")

    def test_specific_review_finding_is_allowed_after_broad_request(self) -> None:
        history = [broad_calculator_request(1)]
        answer = "Review finding: the divide function should raise ValueError when b is 0."

        reason = agent.hub_contribution_guard_reason(answer, history, [])

        self.assertIsNone(reason)

    def test_readme_content_counts_as_artifact(self) -> None:
        request = hub_message(1, "Emil (human)", "all agents, write a README for the calculator project.")
        answer = """# Calculator Project

## Usage
Import the calculator functions and call them directly.

## Testing
Run `pytest`.
"""

        reason = agent.hub_contribution_guard_reason(answer, [], [request])

        self.assertIsNone(reason)

    def test_run_hub_decision_reprompts_deferral_then_passes(self) -> None:
        replies = iter(
            [
                json.dumps(
                    {
                        "action": "final",
                        "answer": "I can draft this code if assigned. Please confirm task assignments.",
                    }
                ),
                json.dumps({"action": "pass", "reason": "another agent already has it"}),
            ]
        )
        config = agent.HubConfig(
            url="https://example.invalid",
            password=None,
            agent_name="emil-flyghed-agent",
            max_messages=5,
            token_budget=10_000,
            poll_seconds=1.0,
            settle_seconds=0.0,
            user_agent="test",
            dry_run=True,
        )
        state = agent.HubRuntimeState(
            max_messages=5,
            token_budget=10_000,
            poll_seconds=1.0,
            settle_seconds=0.0,
        )

        decision = agent.run_hub_decision(
            history=[],
            new_messages=[broad_calculator_request()],
            config=config,
            state=state,
            system_prompt="system",
            complete=lambda _messages: next(replies),
        )

        self.assertEqual(decision.action, "pass")
        self.assertEqual(decision.content, "another agent already has it")

    def test_run_hub_decision_stops_before_exceeding_token_budget(self) -> None:
        config = agent.HubConfig(
            url="https://example.invalid",
            password=None,
            agent_name="emil-flyghed-agent",
            max_messages=5,
            token_budget=100,
            poll_seconds=1.0,
            settle_seconds=0.0,
            user_agent="test",
            dry_run=True,
        )
        state = agent.HubRuntimeState(
            max_messages=5,
            token_budget=100,
            poll_seconds=1.0,
            settle_seconds=0.0,
        )

        def fail_if_called(_messages: list[dict[str, str]]) -> str:
            raise AssertionError("LLM should not be called after token budget is exhausted")

        decision = agent.run_hub_decision(
            history=[],
            new_messages=[broad_calculator_request()],
            config=config,
            state=state,
            system_prompt="system",
            complete=fail_if_called,
        )

        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.content, "token budget reached before next LLM request")
        self.assertTrue(state.snapshot()["stop_requested"])

    def test_direct_coordinator_assignment_is_not_suppressed_as_status(self) -> None:
        messages = [
            coordinator_request(1),
            hub_message(2, "emil-hjaertfors-agent", "I am ready to take testing tasks for the calculator project."),
        ]

        decision = agent.hub_duplicate_status_decision(messages, "emil-flyghed-agent")

        self.assertIsNone(decision)

    def test_direct_other_target_with_group_words_is_passed(self) -> None:
        history = [hub_message(1, "emil-hjaertfors-agent", "Hej, jag ar emil-hjaertfors-agent")]
        message = hub_message(
            2,
            "Emil (human)",
            (
                "@emil-hjaertfors-agent, You are the manager/coordinator for this task; "
                "delegate implementation and testing tasks across all other agents."
            ),
        )

        decision = agent.hub_target_guard_decision(history, [message], "emil-flyghed-agent")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "pass")
        self.assertIn("emil-hjaertfors-agent", decision.content)

    def test_unknown_handle_target_is_passed(self) -> None:
        message = hub_message(1, "Stefan (human)", "@scd do the summary of conversation")

        decision = agent.hub_target_guard_decision([], [message], "emil-flyghed-agent")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "pass")

    def test_bare_first_name_address_is_passed(self) -> None:
        for content in ("Emil, are you here?", "Emil, quote Monty Python for me", "@emil do this"):
            with self.subTest(content=content):
                message = hub_message(1, "Emil F (human)", content)

                decision = agent.hub_target_guard_decision([], [message], "emil-flyghed-agent")

                self.assertIsNotNone(decision)
                self.assertEqual(decision.action, "pass")

    def test_full_handle_still_reaches_agent(self) -> None:
        message = hub_message(1, "Emil F (human)", "emil-flyghed-agent, please add unit tests for divide.")

        decision = agent.hub_target_guard_decision([], [message], "emil-flyghed-agent")

        self.assertIsNone(decision)

    def test_presence_ping_with_task_claim_is_not_noise(self) -> None:
        message = hub_message(
            1,
            "lullo-swe-agent",
            "IDENTIFIED\n\nlullo-swe-agent is online. I can take the README/demo instructions task: "
            "usage steps, examples, and a short run/demo section.",
        )

        self.assertFalse(agent.is_presence_noise(message["content"]))
        decision = agent.hub_target_guard_decision([], [message], "emil-flyghed-agent")
        self.assertIsNone(decision)

    def test_pure_presence_ping_is_still_noise(self) -> None:
        for content in ("lullo-swe-agent is online and ready to help.", "I am here", "redo att hjalpa"):
            with self.subTest(content=content):
                self.assertTrue(agent.is_presence_noise(content))

    def test_coordinator_answer_inviting_claims_is_rejected(self) -> None:
        answer = (
            "I propose splitting tasks into implementation, tests, review, bug checking, and documentation. "
            "I invite agents to claim one task each and review existing progress before starting. "
            "Please confirm your task choice."
        )

        reason = agent.hub_coordinator_guard_reason(
            answer,
            [coordinator_request()],
            [],
            "emil-flyghed-agent",
        )

        self.assertEqual(reason, "coordinator answer invited claims instead of assigning named tasks")

    def test_concrete_named_coordinator_assignments_are_allowed(self) -> None:
        history = [
            coordinator_request(1),
            hub_message(2, "lullo-swe-agent", "I can contribute."),
            hub_message(3, "emil-hjaertfors-agent", "I am ready for implementation, testing, or documentation."),
        ]
        answer = (
            "@lullo-swe-agent take implementation for calculator.py. "
            "@emil-hjaertfors-agent take unit tests for add, subtract, multiply, and divide. "
            "I will cover final run instructions if documentation remains open."
        )

        coordinator_reason = agent.hub_coordinator_guard_reason(answer, history, [], "emil-flyghed-agent")
        contribution_reason = agent.hub_contribution_guard_reason(answer, history, [], "emil-flyghed-agent")

        self.assertIsNone(coordinator_reason)
        self.assertIsNone(contribution_reason)

    def test_reassigning_claimed_tests_without_resolution_is_rejected(self) -> None:
        history = [
            coordinator_request(1),
            hub_message(2, "emil-hjaertfors-agent", "I am ready for implementation, testing, or documentation."),
            hub_message(3, "stefan-code-disaster (@scd)", "For Task 2 (Unit tests), I'll take that on."),
        ]
        answer = "@emil-hjaertfors-agent please claim Task 2: Unit tests."

        reason = agent.hub_coordinator_guard_reason(answer, history, [], "emil-flyghed-agent")

        self.assertEqual(reason, "coordinator answer invited claims instead of assigning named tasks")

    def test_after_tests_pass_coordinator_moves_to_review_and_docs(self) -> None:
        history = [
            coordinator_request(1),
            hub_message(2, "emil-hjaertfors-agent", "Klar med: Task 2: Unit tests. Tester: 4 passed in 0.08s."),
            hub_message(3, "stefan-code-disaster (@scd)", "I can help with remaining tasks."),
        ]
        answer = (
            "Task 2 tests are done. "
            "@emil-hjaertfors-agent take Task 3: Code review. "
            "@stefan-code-disaster switch to Task 5: Documentation and final run instructions."
        )

        reason = agent.hub_coordinator_guard_reason(answer, history, [], "emil-flyghed-agent")

        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
