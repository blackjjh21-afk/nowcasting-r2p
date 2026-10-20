"""Check installation entry points and input-format documentation."""

import importlib
from pathlib import Path
import re
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
METADATA = tomllib.loads((ROOT / "pyproject.toml").read_text())


@pytest.mark.parametrize("target", METADATA["project"]["scripts"].values())
def test_console_entrypoint_is_importable(target: str) -> None:
    module, name = target.split(":")
    assert callable(getattr(importlib.import_module(module), name))


def test_readme_local_links_resolve() -> None:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", (ROOT / "README.md").read_text())
    assert links
    for link in links:
        if link.startswith(("http://", "https://", "#")):
            continue
        assert (ROOT / link.split("#", 1)[0]).is_file(), link


def test_input_documentation_uses_kst_nanoseconds() -> None:
    text = (ROOT / "docs" / "INPUTS.md").read_text()
    assert "KST-naive" in text
    assert "Unix nanoseconds" not in text and "Unix-nanosecond axis" not in text
