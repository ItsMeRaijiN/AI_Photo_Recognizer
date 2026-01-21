from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from backend.core.config import settings

logger = logging.getLogger(__name__)

try:
    import timm
    HAS_TIMM = True
except ImportError:
    timm = None
    HAS_TIMM = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIC = True
except ImportError:
    HAS_HEIC = False


@dataclass
class PredictionResult:
    score: float
    is_ai: bool
    confidence: float
    inference_time_ms: float
    model_type: str
    backbone_name: str
    threshold_used: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "is_ai": self.is_ai,
            "confidence": self.confidence,
            "inference_time_ms": self.inference_time_ms,
            "model_type": self.model_type,
            "backbone_name": self.backbone_name,
            "threshold_used": self.threshold_used,
            "error": self.error,
        }


class MLEngine:
    _instance: "MLEngine | None" = None

    def __new__(cls) -> "MLEngine":
        if cls._instance is None:
            cls._instance = super(MLEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialize()
        self._initialized = True

    def _initialize(self) -> None:
        self.device = torch.device(
            settings.DEVICE if torch.cuda.is_available() or settings.DEVICE == "cpu"
            else "cpu"
        )

        if settings.DEVICE == "mps" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

        self.model_type = "unknown"
        self.backbone_name = "unknown"
        self.model_version: str | None = None
        self.threshold = 0.5

        self.torch_model: nn.Module | None = None
        self.onnx_session: Any = None
        self._onnx_input_name: str | None = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("ML Engine initializing...")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model path: {settings.MODEL_PATH}")
        logger.info(f"  HEIC support: {'yes' if HAS_HEIC else 'no'}")
        logger.info(f"  ONNX support: {'yes' if HAS_ONNX else 'no'}")
        logger.info(f"  timm support: {'yes' if HAS_TIMM else 'no'}")

        self._load_model()

    def _load_model(self) -> None:
        model_path = settings.MODEL_PATH

        if model_path is None or not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return

        meta_path = model_path.parent / "results.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.backbone_name = meta.get("backbone", "unknown")
                    self.threshold = meta.get("best_threshold", 0.5)
                    self.model_version = meta.get("model_version")
                    logger.info(
                        f"Loaded metadata: backbone={self.backbone_name}, "
                        f"threshold={self.threshold:.4f}"
                    )
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        if str(model_path).endswith(".onnx"):
            self._load_onnx(model_path)
        else:
            self._load_pytorch(model_path)

    def _load_pytorch(self, path: Path) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                config = checkpoint.get("config", {})
                if isinstance(config, dict):
                    if self.backbone_name == "unknown":
                        self.backbone_name = config.get("backbone", "effnetv2")
                    if self.threshold == 0.5:
                        self.threshold = checkpoint.get("best_threshold", 0.5)

            backbone = (self.backbone_name or "unknown").lower()
            logger.info(f"Loading backbone: {self.backbone_name}")

            if "convnext" in backbone:
                if not HAS_TIMM:
                    raise ImportError(
                        "ConvNeXt requires `timm` library. Install with: pip install timm"
                    )
                self.torch_model = timm.create_model(
                    "convnextv2_tiny.fcmae_ft_in22k_in1k",
                    pretrained=False,
                    num_classes=1,
                    drop_rate=0.3,
                )

            elif "effnet" in backbone or "efficientnet" in backbone:
                self.torch_model = models.efficientnet_v2_s(weights=None)
                in_features = self.torch_model.classifier[1].in_features
                self.torch_model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, 1),
                )

            else:
                raise ValueError(
                    f"Unsupported backbone '{self.backbone_name}'. Supported: convnext*, effnet*."
                )

            state = checkpoint["model_state_dict"] if (
                    isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            ) else checkpoint

            self.torch_model.load_state_dict(state)
            self.torch_model.to(self.device)
            self.torch_model.eval()
            self.model_type = "torch"

            logger.info(f"PyTorch model loaded: {self.backbone_name}")
            logger.info(f"  Threshold: {self.threshold:.4f}")

        except Exception as e:
            logger.exception(f"PyTorch load error: {e}")
            self.model_type = "unknown"
            self.torch_model = None

    def _load_onnx(self, path: Path) -> None:
        if not HAS_ONNX:
            logger.error("ONNX Runtime not installed")
            return

        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.device.type == "cuda" else ['CPUExecutionProvider']

            self.onnx_session = ort.InferenceSession(str(path), providers=providers)
            self._onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.model_type = "onnx"

            if self.backbone_name == "unknown":
                self.backbone_name = "onnx_model"

            logger.info("ONNX model loaded")

        except Exception as e:
            logger.exception(f"ONNX load error: {e}")

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _calculate_confidence(self, score: float) -> float:
        return score if score >= 0.5 else 1.0 - score

    def _create_error_result(self, error: str) -> PredictionResult:
        return PredictionResult(
            score=0.5,
            is_ai=False,
            confidence=0.5,
            inference_time_ms=0,
            model_type=self.model_type,
            backbone_name=self.backbone_name,
            threshold_used=self.threshold,
            error=error
        )

    def predict_batch(
        self,
        images: list[Image.Image],
        return_errors: bool = False
    ) -> list[PredictionResult | None]:
        if not images:
            return []

        if self.model_type == "unknown":
            error_result = self._create_error_result("Model not loaded")
            return [error_result] * len(images)

        start_time = time.perf_counter()
        results: list[PredictionResult | None] = []

        valid_tensors: list[torch.Tensor] = []
        valid_indices: list[int] = []
        errors: dict[int, str] = {}

        for i, image in enumerate(images):
            try:
                tensor = self.transform(image)
                valid_tensors.append(tensor)
                valid_indices.append(i)
            except Exception as e:
                errors[i] = str(e)

        scores: np.ndarray = np.array([])
        if valid_tensors:
            try:
                batch_tensor = torch.stack(valid_tensors)

                if self.model_type == "torch" and self.torch_model is not None:
                    batch_tensor = batch_tensor.to(self.device)
                    with torch.no_grad():
                        outputs = self.torch_model(batch_tensor)
                        scores = torch.sigmoid(outputs).cpu().numpy().flatten()

                elif self.model_type == "onnx" and self.onnx_session is not None:
                    input_arr = batch_tensor.cpu().numpy()
                    outputs = self.onnx_session.run(
                        None, {self._onnx_input_name: input_arr}
                    )
                    raw_output = np.array(outputs[0]).flatten()
                    scores = self._sigmoid(raw_output)

                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

            except Exception as e:
                for i in valid_indices:
                    errors[i] = f"Batch inference error: {e}"
                scores = np.array([])
                valid_indices = []

        inference_time = (time.perf_counter() - start_time) * 1000
        time_per_image = inference_time / len(images) if images else 0

        score_idx = 0
        for i in range(len(images)):
            if i in errors:
                if return_errors:
                    results.append(self._create_error_result(errors[i]))
                else:
                    results.append(None)
            elif i in valid_indices and score_idx < len(scores):
                score = float(scores[score_idx])
                results.append(PredictionResult(
                    score=score,
                    is_ai=bool(score >= self.threshold),
                    confidence=self._calculate_confidence(score),
                    inference_time_ms=float(time_per_image),
                    model_type=self.model_type,
                    backbone_name=self.backbone_name,
                    threshold_used=float(self.threshold),
                ))
                score_idx += 1
            else:
                if return_errors:
                    results.append(self._create_error_result("Unknown error"))
                else:
                    results.append(None)

        return results

    def predict(self, image: Image.Image) -> PredictionResult:
        if self.model_type == "unknown":
            return self._create_error_result("Model not loaded")

        start_time = time.perf_counter()

        try:
            img_tensor = self.transform(image).unsqueeze(0)

            if self.model_type == "torch" and self.torch_model is not None:
                img_tensor = img_tensor.to(self.device)
                with torch.no_grad():
                    output = self.torch_model(img_tensor)
                    score = torch.sigmoid(output).item()

            elif self.model_type == "onnx" and self.onnx_session is not None:
                input_arr = img_tensor.cpu().numpy()
                output = self.onnx_session.run(
                    None, {self._onnx_input_name: input_arr}
                )
                raw_output = np.array(output[0]).flatten()[0]
                score = float(self._sigmoid(np.array([raw_output]))[0])

            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            inference_time = (time.perf_counter() - start_time) * 1000

            return PredictionResult(
                score=float(score),
                is_ai=bool(score >= self.threshold),
                confidence=self._calculate_confidence(score),
                inference_time_ms=float(inference_time),
                model_type=self.model_type,
                backbone_name=self.backbone_name,
                threshold_used=float(self.threshold),
            )

        except Exception as e:
            return self._create_error_result(str(e))

    def _get_gradcam_target_layer(self) -> nn.Module | None:
        if self.torch_model is None:
            return None

        backbone = (self.backbone_name or "unknown").lower()

        try:
            if "convnext" in backbone:
                if hasattr(self.torch_model, "stages") and self.torch_model.stages:
                    return self.torch_model.stages[-1]
                if hasattr(self.torch_model, "features") and self.torch_model.features:
                    return self.torch_model.features[-1]

                logger.warning("GradCAM: Unknown ConvNeXt structure (no stages/features).")
                return None

            if "effnet" in backbone or "efficientnet" in backbone:
                if hasattr(self.torch_model, "features") and self.torch_model.features:
                    return self.torch_model.features[-1]

                logger.warning("GradCAM: EfficientNet structure missing 'features'.")
                return None

            logger.warning(f"GradCAM not supported for backbone: {backbone}")
            return None

        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(f"Could not find GradCAM target layer for {backbone}: {e}")
            return None

    def generate_heatmap(
        self,
        image: Image.Image,
        save_path: Path | None = None
    ) -> Image.Image | None:
        if self.model_type != "torch" or self.torch_model is None:
            return None

        target_layer = self._get_gradcam_target_layer()
        if target_layer is None:
            logger.warning(f"GradCAM not supported for backbone: {self.backbone_name}")
            return None

        self.torch_model.eval()
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        gradients: list[torch.Tensor] = []
        activations: list[torch.Tensor] = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        h_forward = target_layer.register_forward_hook(forward_hook)
        h_backward = target_layer.register_full_backward_hook(backward_hook)

        try:
            output = self.torch_model(img_tensor)
            self.torch_model.zero_grad()
            output.backward()

            if not gradients or not activations:
                return None

            grads = gradients[0].detach().cpu().numpy()[0]
            acts = activations[0].detach().cpu().numpy()[0]

            weights = np.mean(grads, axis=(1, 2))
            cam = np.sum(weights[:, None, None] * acts, axis=0)

            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            cam_resized = cv2.resize(cam, image.size)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            img_array = np.array(image)
            superimposed = np.uint8(img_array * 0.6 + heatmap * 0.4)
            result = Image.fromarray(superimposed)

            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(save_path, "JPEG", quality=90)

            return result

        except Exception as e:
            logger.warning(f"Heatmap generation error: {e}")
            return None

        finally:
            h_forward.remove()
            h_backward.remove()

    def generate_heatmap_for_analysis(
        self,
        image: Image.Image,
        analysis_id: int
    ) -> str | None:
        import uuid
        filename = f"heatmap_{analysis_id}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = settings.HEATMAPS_DIR / filename

        result = self.generate_heatmap(image, save_path)

        if result and save_path.exists():
            return str(save_path)
        return None

    @property
    def is_loaded(self) -> bool:
        return self.model_type != "unknown"

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "type": self.model_type,
            "backbone": self.backbone_name,
            "version": self.model_version,
            "threshold": self.threshold,
            "device": str(self.device),
            "loaded": self.is_loaded,
        }

ml_engine = MLEngine()