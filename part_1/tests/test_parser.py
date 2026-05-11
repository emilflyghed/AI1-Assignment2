from parser import parse


def test_action_basic():
    text = "Thought: I should look at the files.\nAction: bash\nAction Input: ls -la"
    r = parse(text)
    assert r.kind == "action"
    assert r.thought == "I should look at the files."
    assert r.action == "bash"
    assert r.action_input == "ls -la"


def test_final_basic():
    text = "Thought: done.\nFinal Answer: 42"
    r = parse(text)
    assert r.kind == "final"
    assert r.thought == "done."
    assert r.final_answer == "42"


def test_final_preferred_over_action():
    text = "Thought: nope\nAction: bash\nAction Input: ls\nFinal Answer: actually we are done"
    r = parse(text)
    assert r.kind == "final"
    assert "done" in r.final_answer


def test_extra_whitespace_tolerated():
    text = (
        "   Thought:    reasoning here   \n\n"
        "  Action:   bash   \n"
        "  Action Input:    echo hello   \n"
    )
    r = parse(text)
    assert r.kind == "action"
    assert r.thought == "reasoning here"
    assert r.action == "bash"
    assert r.action_input == "echo hello"


def test_case_insensitive_labels():
    text = "thought: t\nACTION: bash\naction input: pwd"
    r = parse(text)
    assert r.kind == "action"
    assert r.action_input == "pwd"


def test_multiline_final_answer():
    text = "Thought: ok\nFinal Answer: line1\nline2\nline3"
    r = parse(text)
    assert r.kind == "final"
    assert "line1" in r.final_answer
    assert "line2" in r.final_answer
    assert "line3" in r.final_answer


def test_empty_input_is_error():
    r = parse("")
    assert r.kind == "error"
    assert r.error


def test_whitespace_only_is_error():
    r = parse("   \n\n  ")
    assert r.kind == "error"


def test_no_labels_is_error():
    r = parse("just some chatty text without any labels")
    assert r.kind == "error"


def test_action_without_input_is_error():
    r = parse("Thought: hmm\nAction: bash")
    assert r.kind == "error"
    assert "Action Input" in r.error


def test_action_input_without_action_is_error():
    r = parse("Thought: hmm\nAction Input: ls")
    assert r.kind == "error"


def test_thought_optional_for_action():
    # Action + Action Input alone should still parse as action.
    text = "Action: bash\nAction Input: pwd"
    r = parse(text)
    assert r.kind == "action"
    assert r.action_input == "pwd"


def test_action_input_with_pipes_and_quotes():
    text = 'Thought: count\nAction: bash\nAction Input: find . -name "*.py" | wc -l'
    r = parse(text)
    assert r.kind == "action"
    assert r.action_input == 'find . -name "*.py" | wc -l'


def test_thought_does_not_swallow_following_action():
    text = "Thought: I will list files\nAction: bash\nAction Input: ls"
    r = parse(text)
    assert r.kind == "action"
    assert r.thought == "I will list files"
    assert r.action == "bash"
