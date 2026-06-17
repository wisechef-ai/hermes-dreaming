from hermes_dreaming import __version__
from hermes_dreaming.cli import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_status_command(capsys, tmp_path, monkeypatch) -> None:
    # Isolate to an empty dreaming home so output is deterministic and doesn't
    # depend on whatever real artifacts exist on the host.
    monkeypatch.setenv("HERMES_DREAMING_HOME", str(tmp_path))
    assert main(["status"]) == 0
    out = capsys.readouterr().out.strip()
    assert "dream" in out.lower()  # "No dream artifacts found."
