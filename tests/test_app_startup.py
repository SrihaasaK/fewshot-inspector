"""The demo app renders headlessly without loading the model (no weight download).

Runs wherever Streamlit is installed: locally, and in CI's `app-runtime` job, which installs app/requirements.txt
with uv on Linux x86_64 / Python 3.14 — the live Streamlit Community Cloud runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FEWSHOT_DEVICE", "cpu")
    at = testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info])


def test_app_renders_with_examples(app):
    assert "▶ Try an example" in [b.label for b in app.button]
    example = next(s for s in app.selectbox if s.label == "Example part")
    assert set(example.options) == {"bottle", "screw"}


def test_demo_is_labelled_as_not_benchmark_accuracy(app):
    text = _text(app).lower()
    assert "not benchmark accuracy" in text
    assert "672 px" in text  # the benchmark resolution is stated next to the demo resolution
