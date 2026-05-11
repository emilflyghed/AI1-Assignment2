import tools
from tools import run_bash


def test_echo_stdout():
    out = run_bash("echo hello")
    assert "hello" in out


def test_stderr_captured():
    out = run_bash("echo oops 1>&2")
    assert "oops" in out


def test_nonzero_exit_reported():
    out = run_bash('python -c "import sys; sys.exit(7)"')
    assert "exit code" in out
    assert "7" in out


def test_timeout(monkeypatch):
    monkeypatch.setattr(tools, "TIMEOUT", 1)
    out = tools.run_bash('python -c "import time; time.sleep(5)"')
    assert "timed out" in out


def test_truncation():
    out = run_bash('python -c "print(\'x\' * 3000)"')
    assert "truncated" in out
    # Body capped to MAX_OUTPUT plus a small marker.
    assert len(out) < tools.MAX_OUTPUT + len(tools.TRUNC_MARKER) + 1


def test_empty_command():
    out = run_bash("")
    assert "error" in out.lower()


def test_does_not_raise_on_garbage():
    # Whatever weird input, executor must return a string, not raise.
    out = run_bash("this_command_definitely_does_not_exist_xyz_123")
    assert isinstance(out, str)
    assert out  # non-empty (should mention error / non-zero exit)
