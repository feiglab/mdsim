import json
import shlex
import subprocess
import sys


def run_cli(cmd: str) -> str:
    exe = [sys.executable, "-m", "mdsim.cli"]
    return subprocess.check_output(exe + shlex.split(cmd), text=True)


def test_info():
    out = run_cli("info")
    d = json.loads(out)
    assert "version" in d["model"]
