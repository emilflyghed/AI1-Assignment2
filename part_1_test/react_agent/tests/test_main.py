import main


def test_cli_uses_real_llm_by_default(monkeypatch, capsys):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    created = []

    class FakeRealLLM:
        def __init__(self, model):
            self.model = model
            self.prompt = None
            created.append(self)

        def complete(self, prompt):
            self.prompt = prompt
            return "Thought: done\nFinal Answer: ok"

    monkeypatch.setattr(main, "RealLLM", FakeRealLLM)

    code = main.main(["--provider", "gemini", "--model", "demo-model", "custom", "task"])

    assert code == 0
    assert created
    assert created[0].model == "demo-model"
    assert "User task: custom task" in created[0].prompt
    assert "ok" in capsys.readouterr().out


def test_cli_uses_lmstudio_provider(monkeypatch, capsys):
    created = []

    class FakeLMStudioLLM:
        def __init__(self, model, base_url):
            self.model = model
            self.base_url = base_url
            self.prompt = None
            created.append(self)

        def complete(self, prompt):
            self.prompt = prompt
            return "Thought: done\nFinal Answer: local ok"

    monkeypatch.setattr(main, "LMStudioLLM", FakeLMStudioLLM)

    code = main.main(
        [
            "--provider",
            "lmstudio",
            "--model",
            "google/gemma-local",
            "--base-url",
            "http://localhost:1234/v1",
            "local",
            "task",
        ]
    )

    assert code == 0
    assert created
    assert created[0].model == "google/gemma-local"
    assert created[0].base_url == "http://localhost:1234/v1"
    assert "User task: local task" in created[0].prompt
    assert "local ok" in capsys.readouterr().out


def test_cli_provider_can_come_from_environment(monkeypatch, capsys):
    created = []

    class FakeLMStudioLLM:
        def __init__(self, model, base_url):
            self.model = model
            self.base_url = base_url
            self.prompt = None
            created.append(self)

        def complete(self, prompt):
            self.prompt = prompt
            return "Thought: done\nFinal Answer: env local ok"

    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setattr(main, "LMStudioLLM", FakeLMStudioLLM)

    code = main.main(["env", "task"])

    assert code == 0
    assert created
    assert created[0].model is None
    assert created[0].base_url is None
    assert "User task: env task" in created[0].prompt
    assert "env local ok" in capsys.readouterr().out


def test_cli_mock_mode_does_not_construct_real_llm(monkeypatch, capsys):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("RealLLM should not be constructed in --mock mode")

    monkeypatch.setattr(main, "RealLLM", fail_if_called)

    code = main.main(["--mock"])

    assert code == 0
    assert "count of Python files" in capsys.readouterr().out
