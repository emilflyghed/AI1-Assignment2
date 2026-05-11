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


def test_loop_recovers_from_malformed_response():
    script = [
        "I forgot the format entirely",
        "Thought: ok now\nFinal Answer: recovered",
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
