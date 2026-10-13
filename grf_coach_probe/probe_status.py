"""Read-only GRF installation/API inventory; does not start a match."""

import ast
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "football"
ACTION_FILE = SOURCE / "gfootball" / "env" / "football_action_set.py"


def run(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


tree = ast.parse(ACTION_FILE.read_text(encoding="utf-8"))
v1 = None
v2_addition = None
for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    if any(isinstance(target, ast.Name) and target.id == "action_set_v1"
           for target in node.targets) and isinstance(node.value, ast.List):
        v1 = [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
    if any(isinstance(target, ast.Name) and target.id == "action_set_v2"
           for target in node.targets) and isinstance(node.value, ast.BinOp):
        if isinstance(node.value.right, ast.List):
            v2_addition = [elt.id for elt in node.value.right.elts
                           if isinstance(elt, ast.Name)]

windows_python = ROOT / ".venv" / "Scripts" / "python.exe"
linux_root_result = run("wsl", "-d", "Ubuntu", "--", "wslpath", "-a", str(ROOT))
linux_python = None
linux_install = None
linux_version = None
if linux_root_result["exit_code"] == 0:
    linux_python = linux_root_result["stdout"] + "/.venv-linux/bin/python"
    linux_install = run("wsl", "-d", "Ubuntu", "--", linux_python,
                        "-m", "pip", "show", "gfootball")
    linux_version = run("wsl", "-d", "Ubuntu", "--", linux_python, "--version")

result = {
    "source_commit": run("git", "-C", str(SOURCE), "rev-parse", "HEAD")["stdout"],
    "host_os": platform.platform(),
    "host_python": sys.version.split()[0],
    "windows_venv_python": run(str(windows_python), "--version") if windows_python.exists() else None,
    "windows_gfootball": run(str(windows_python), "-m", "pip", "show", "gfootball") if windows_python.exists() else None,
    "wsl_python": linux_version,
    "wsl_gfootball": linux_install,
    "source_v1_action_count": len(v1) if v1 is not None else None,
    "source_v2_action_count": len(v1) + len(v2_addition) if v1 is not None and v2_addition is not None else None,
    "source_v2_added_actions": v2_addition,
    "source_builtin_ai_action_id": v1.index("action_builtin_ai") if v1 is not None and "action_builtin_ai" in v1 else len(v1) if v1 is not None and v2_addition == ["action_builtin_ai"] else None,
}
(ROOT / "status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
