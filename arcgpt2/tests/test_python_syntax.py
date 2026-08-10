from __future__ import annotations

from pathlib import Path


def test_all_arcgpt2_python_files_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path}: {exc}")
    assert not failures, "\n".join(failures)
