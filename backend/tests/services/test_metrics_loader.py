from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from backend.services.metrics_loader import MetricsLoader


@pytest.fixture
def loader():
    with patch.object(MetricsLoader, "reload_metrics", return_value=None):
        return MetricsLoader()


class TestImageProcessing:
    def test_resize_large_image(self, loader: MetricsLoader):
        large_img = Image.new("RGB", (1000, 500))
        resized = loader._resize_if_needed(large_img)
        assert max(resized.size) == loader.MAX_METRIC_SIZE

    def test_small_image_unchanged(self, loader: MetricsLoader):
        small_img = Image.new("RGB", (100, 100))
        result = loader._resize_if_needed(small_img)
        assert result.size == (100, 100)


class TestMetricComputation:
    def test_compute_single_handles_exception(self, loader: MetricsLoader):
        def faulty_metric(img, score):
            raise ValueError("Metric Crash")

        name, result = loader._compute_single(
            "faulty", faulty_metric, Image.new("RGB", (10, 10)), 0.5, None
        )

        assert name == "faulty"
        assert "error" in result
        assert "Metric Crash" in result["error"]

    def test_compute_single_unknown_returns_none(self, loader: MetricsLoader):
        result = loader.compute_single("missing_metric", Image.new("RGB", (10, 10)), 0.5)
        assert result is None

class TestMetricLoading:
    def test_missing_directory_logs_warning(self, tmp_path: Path, caplog):
        from backend.core.config import settings

        missing = tmp_path / "nonexistent"

        with patch.object(MetricsLoader, "reload_metrics", return_value=None):
            loader = MetricsLoader()

        with patch.object(settings, "METRICS_DIR", missing):
            loader.reload_metrics()

        assert any("Metrics directory not found" in r.message for r in caplog.records)

    def test_loads_valid_plugins(self, tmp_path: Path):
        from backend.core.config import settings

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir(parents=True)

        (metrics_dir / "_common.py").write_text(
            textwrap.dedent("""
                def to_python_types(x):
                    return x
            """),
            encoding="utf-8",
        )

        (metrics_dir / "plugin_ok.py").write_text(
            textwrap.dedent("""
                def calculate(image, score, context=None):
                    return {"ok": True, "score": float(score)}
            """),
            encoding="utf-8",
        )
        (metrics_dir / "plugin_bad.py").write_text("x = 1\n", encoding="utf-8")

        with patch.object(settings, "METRICS_DIR", metrics_dir):
            loader = MetricsLoader()

        names = loader.get_metric_names()
        assert "plugin_ok" in names
        assert "plugin_bad" not in names

    def test_executes_loaded_plugin(self, tmp_path: Path):
        from backend.core.config import settings

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir(parents=True)

        (metrics_dir / "test_metric.py").write_text(
            textwrap.dedent("""
                def calculate(image, score, context=None):
                    return {"result": True, "score": float(score)}
            """),
            encoding="utf-8",
        )

        with patch.object(settings, "METRICS_DIR", metrics_dir):
            loader = MetricsLoader()

        result = loader.compute_single("test_metric", Image.new("RGB", (10, 10)), 0.75)

        assert result is not None
        assert result["result"] is True
        assert result["score"] == pytest.approx(0.75)