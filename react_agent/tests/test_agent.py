import agent
from agent import run, MAX_ITERS
from llm import MockLLM


def test_loop_returns_final_answer_immediately():
    llm = MockLLM(["Thought: easy\nFinal Answer: 42"])
    out = run(llm, "what is the answer?")
    assert "42" in out


def test_loop_executes_action_then_final():
    script = [
        "Thought: I'll echo it.\nAction: bash\nAction Input: echo seven",
        "Thought: got it.\nFinal Answer: the value was seven",
    ]
    llm = MockLLM(script)
    out = run(llm, "echo seven please")
    assert "seven" in out


def test_loop_can_trace_model_and_observation():
    events = []
    script = [
        "Thought: I'll echo it.\nAction: bash\nAction Input: echo seven",
        "Thought: got it.\nFinal Answer: the value was seven",
    ]
    llm = MockLLM(script)

    out = run(llm, "echo seven please", on_step=lambda kind, text: events.append((kind, text)))

    assert "seven" in out
    assert ("observation", "seven\n") in events
    assert events[0][0] == "model"


def test_loop_recovers_from_malformed_response():
    script = [
        "I forgot the format entirely",
        "Thought: ok now\nFinal Answer: recovered",
    ]
    llm = MockLLM(script)
    out = run(llm, "task")
    assert "recovered" in out


def test_loop_rejects_unknown_action(monkeypatch):
    def fail_if_called(command):
        raise AssertionError(f"unexpected bash execution: {command}")

    monkeypatch.setattr(agent, "run_bash", fail_if_called)
    script = [
        "Thought: wrong tool\nAction: python\nAction Input: print('no')",
        "Thought: corrected\nFinal Answer: recovered",
    ]
    llm = MockLLM(script)
    out = run(llm, "task")
    assert "recovered" in out


def test_loop_recovers_from_multiline_action_input(monkeypatch):
    def fail_if_called(command):
        raise AssertionError(f"unexpected bash execution: {command}")

    monkeypatch.setattr(agent, "run_bash", fail_if_called)
    script = [
        "Thought: multiline\nAction: bash\nAction Input: echo one\necho two",
        "Thought: corrected\nFinal Answer: recovered",
    ]
    llm = MockLLM(script)
    out = run(llm, "task")
    assert "recovered" in out


def test_loop_caps_iterations():
    action_resp = "Thought: loop\nAction: bash\nAction Input: echo x"
    # Provide more than MAX_ITERS so the cap, not the script, ends the loop.
    llm = MockLLM([action_resp] * (MAX_ITERS + 5))
    out = run(llm, "task")
    assert "max iterations" in out
