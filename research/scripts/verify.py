import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if sys.version_info < (3, 12):
        print("Verification requires Python 3.12 or newer.", file=sys.stderr)
        return 1

    node = shutil.which("node")
    if node is None:
        print(
            "Install Node.js 18 or newer and make node available on PATH.",
            file=sys.stderr,
        )
        return 1
    version = subprocess.check_output([node, "--version"], text=True).strip()
    match = re.fullmatch(r"v(\d+)\..+", version)
    if not match or int(match.group(1)) < 18:
        print(f"Node.js 18 or newer is required; found {version}.", file=sys.stderr)
        return 1

    modules = (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "pytest",
        "ruff",
        "mypy",
        "playwright",
    )
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing development packages: " + ", ".join(missing), file=sys.stderr)
        print(
            "Install them with: python -m pip install -r research/requirements.txt "
            "pytest==9.1.1 ruff==0.15.18 mypy==2.1.0 playwright==1.63.0",
            file=sys.stderr,
        )
        return 1

    environment = os.environ.copy()
    environment["PULSESHIFT_PYTHON"] = sys.executable
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(ROOT / "research"), environment.get("PYTHONPATH", ""))
        if part
    )
    environment["MYPYPATH"] = str(ROOT / "research")
    javascript_tests = sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("*.test.cjs")
    )
    python = [sys.executable, "-m"]
    with tempfile.TemporaryDirectory(prefix="pulseshift-checks-") as temporary:
        cache = Path(temporary)
        pytest = python + ["pytest", "-q", "-p", "no:cacheprovider", "--basetemp"]
        sources = ["research", "tests"]
        checks = [
            ("JavaScript tests", [node, "--test", *javascript_tests]),
            ("Research tests", pytest + [str(cache / "research"), "research/tests"]),
            (
                "Python lint",
                python + ["ruff", "check", "--no-cache", *sources],
            ),
            (
                "Python formatting",
                python + ["ruff", "format", "--check", "--no-cache", *sources],
            ),
            (
                "Python types",
                python
                + [
                    "mypy",
                    "--ignore-missing-imports",
                    "--explicit-package-bases",
                    "--cache-dir",
                    str(cache / "mypy"),
                    "research/pulseshift",
                    "research/scripts",
                ],
            ),
            (
                "Browser tests",
                pytest + [str(cache / "browser"), "tests/test_browser.py"],
            ),
        ]
        for name, command in checks:
            print(f"\n{name}", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode:
                print(f"{name} failed.", file=sys.stderr)
                return 1
    print("\nAll six checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
