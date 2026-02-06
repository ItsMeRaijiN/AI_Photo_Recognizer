from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def engine():
    from backend.services.ml_engine import ml_engine
    return ml_engine

class TestPredictSingle:
    def test_unknown_model_returns_error(self, engine):
        with patch.object(engine, "model_type", "unknown"):
            r = engine.predict(Image.new("RGB", (224, 224)))

        assert r.error and "Model not loaded" in r.error

    @patch("torch.no_grad")
    def test_torch_model_happy_path(self, _no_grad, engine):
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[10.0]])

        with (
            patch.object(engine, "torch_model", mock_model),
            patch.object(engine, "model_type", "torch"),
        ):
            result = engine.predict(Image.new("RGB", (224, 224)))

        assert result.error is None
        assert result.is_ai is True
        assert result.score > 0.5

    def test_onnx_model_happy_path(self, engine):
        sess = MagicMock()
        sess.get_inputs.return_value = [MagicMock(name="input0")]
        sess.get_inputs.return_value[0].name = "input0"
        sess.run.return_value = [np.array([[10.0]], dtype=np.float32)]

        with (
            patch.object(engine, "model_type", "onnx"),
            patch.object(engine, "onnx_session", sess),
            patch.object(engine, "_onnx_input_name", "input0"),
        ):
            r = engine.predict(Image.new("RGB", (224, 224)))

        assert r.error is None
        assert r.is_ai is True


class TestPredictBatch:
    def test_unknown_model_returns_errors(self, engine):
        with patch.object(engine, "model_type", "unknown"):
            rs = engine.predict_batch(
                [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))],
                return_errors=True,
            )

        assert len(rs) == 2
        assert all(r.error for r in rs)

    def test_empty_list_returns_empty(self, engine):
        with patch.object(engine, "model_type", "torch"):
            assert engine.predict_batch([], return_errors=True) == []

    def test_preprocess_error_returns_none(self, engine):
        def bad_transform(_img):
            raise ValueError("bad image")

        with (
            patch.object(engine, "model_type", "torch"),
            patch.object(engine, "transform", side_effect=bad_transform),
        ):
            out = engine.predict_batch([Image.new("RGB", (10, 10))], return_errors=False)

        assert out == [None]

    def test_torch_batch_happy_path(self, engine):
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[10.0], [0.0]])

        with (
            patch.object(engine, "torch_model", mock_model),
            patch.object(engine, "model_type", "torch"),
        ):
            results = engine.predict_batch(
                [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))],
                return_errors=True,
            )

        assert len(results) == 2
        assert results[0].is_ai is True

    def test_onnx_batch_happy_path(self, engine):
        sess = MagicMock()
        sess.run.return_value = [np.array([[10.0], [-10.0]], dtype=np.float32)]

        with (
            patch.object(engine, "model_type", "onnx"),
            patch.object(engine, "onnx_session", sess),
            patch.object(engine, "_onnx_input_name", "input0"),
        ):
            rs = engine.predict_batch(
                [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))],
                return_errors=True,
            )

        assert len(rs) == 2
        assert rs[0].is_ai is True
        assert rs[1].is_ai is False

    def test_inference_error_marks_all(self, engine, monkeypatch):
        imgs = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]

        def ok_transform(_img):
            return torch.zeros((3, 224, 224), dtype=torch.float32)

        def crash_model(_batch):
            raise RuntimeError("boom")

        monkeypatch.setattr(engine, "model_type", "torch")
        monkeypatch.setattr(engine, "torch_model", crash_model)
        monkeypatch.setattr(engine, "transform", ok_transform)

        out = engine.predict_batch(imgs, return_errors=True)

        assert len(out) == 2
        assert all("Batch inference error" in r.error for r in out)


class TestModelLoading:
    def test_missing_path_returns_none(self, engine, tmp_path: Path):
        from backend.core.config import settings

        missing = tmp_path / "nope.pt"

        with patch.object(settings, "MODEL_PATH", missing):
            ok = engine._load_model()

        assert ok is None


class TestGradCAM:
    def test_no_model_returns_none(self, engine):
        with patch.object(engine, "torch_model", None):
            assert engine._get_gradcam_target_layer() is None

    def test_non_torch_returns_none(self, engine):
        with (
            patch.object(engine, "model_type", "onnx"),
            patch.object(engine, "torch_model", None),
        ):
            assert engine.generate_heatmap(Image.new("RGB", (224, 224))) is None

    def test_no_target_layer_returns_none(self, engine):
        with (
            patch.object(engine, "model_type", "torch"),
            patch.object(engine, "torch_model", MagicMock()),
            patch.object(engine, "_get_gradcam_target_layer", return_value=None),
        ):
            assert engine.generate_heatmap(Image.new("RGB", (224, 224))) is None


    def test_heatmap_for_analysis_returns_none_when_generate_fails(self, engine, tmp_path: Path):
        from backend.core.config import settings

        with (
            patch.object(settings, "HEATMAPS_DIR", tmp_path),
            patch.object(engine, "generate_heatmap", return_value=None),
        ):
            out = engine.generate_heatmap_for_analysis(Image.new("RGB", (224, 224)), analysis_id=123)

        assert out is None

    def test_heatmap_for_analysis_saves_file(self, engine, tmp_path: Path):
        from backend.core.config import settings

        heatmap_img = Image.new("RGB", (10, 10))

        def fake_generate(_img, save_path=None):
            if save_path:
                Path(save_path).write_bytes(b"x")
            return heatmap_img

        with (
            patch.object(settings, "HEATMAPS_DIR", tmp_path),
            patch.object(engine, "generate_heatmap", side_effect=fake_generate),
        ):
            out = engine.generate_heatmap_for_analysis(Image.new("RGB", (224, 224)), analysis_id=777)

        assert out is not None
        assert Path(out).exists()


class TestModelInfo:
    def test_model_info_returns_correct_values(self, engine):
        prev_type = engine.model_type
        prev_backbone = engine.backbone_name
        prev_threshold = engine.threshold

        try:
            engine.model_type = "torch"
            engine.backbone_name = "test_backbone"
            engine.threshold = 0.33

            info = engine.model_info

            assert info["loaded"] is True
            assert info["type"] == "torch"
            assert info["backbone"] == "test_backbone"
            assert info["threshold"] == 0.33
        finally:
            engine.model_type = prev_type
            engine.backbone_name = prev_backbone
            engine.threshold = prev_threshold