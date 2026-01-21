from __future__ import annotations

import importlib.util
import inspect
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from backend.core.config import settings

logger = logging.getLogger(__name__)


MetricFunction = Callable[..., dict[str, Any]]


class MetricsLoader:

    MAX_METRIC_SIZE: int = 512
    DEFAULT_MAX_WORKERS: int = 4

    def __init__(self) -> None:
        self._metrics: dict[str, MetricFunction] = {}
        self._metric_info: dict[str, dict[str, Any]] = {}
        self._supports_context: dict[str, bool] = {}
        self._to_python_types: Callable[[Any], Any] | None = None
        self.reload_metrics()

    def reload_metrics(self) -> None:
        self._metrics = {}
        self._metric_info = {}
        self._supports_context = {}
        self._to_python_types = None

        metrics_dir = settings.METRICS_DIR

        if not metrics_dir or not metrics_dir.exists():
            logger.warning(f"Metrics directory not found: {metrics_dir}")
            return

        logger.info(f"Loading custom metrics from: {metrics_dir}")

        self._preload_common_module(metrics_dir)

        for filename in sorted(os.listdir(metrics_dir)):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue

            metric_name = filename[:-3]
            file_path = metrics_dir / filename

            try:
                self._load_single_metric(metric_name, file_path)
            except Exception as e:
                logger.error(f"Failed to load metric '{metric_name}': {e}")

        logger.info(f"Loaded {len(self._metrics)} metrics: {list(self._metrics.keys())}")

    def _preload_common_module(self, metrics_dir: Path) -> None:
        import sys
        import types

        try:
            from backend.custom_metrics._common import to_python_types
            self._to_python_types = to_python_types
            logger.debug("Loaded to_python_types from backend.custom_metrics._common")
            return
        except ImportError:
            pass

        common_path = metrics_dir / "_common.py"
        if not common_path.exists():
            logger.debug("No _common.py found")
            return

        try:
            spec = importlib.util.spec_from_file_location("_common", common_path)
            if not spec or not spec.loader:
                return

            common_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(common_module)

            if hasattr(common_module, "to_python_types"):
                self._to_python_types = common_module.to_python_types
                logger.debug("Loaded to_python_types from manually loaded _common.py")

            if "backend" not in sys.modules:
                backend_pkg = types.ModuleType("backend")
                backend_pkg.__path__ = []
                sys.modules["backend"] = backend_pkg

            if "backend.custom_metrics" not in sys.modules:
                cm_pkg = types.ModuleType("backend.custom_metrics")
                cm_pkg.__path__ = [str(metrics_dir)]
                sys.modules["backend.custom_metrics"] = cm_pkg

            sys.modules["backend.custom_metrics._common"] = common_module

            logger.debug("Manually registered backend.custom_metrics._common in sys.modules")

        except Exception as e:
            logger.warning(f"Failed to preload _common.py: {e}")

    def _convert_result_types(self, obj: Any) -> Any:
        if self._to_python_types is not None:
            return self._to_python_types(obj)

        return self._fallback_convert_types(obj)

    @staticmethod
    def _fallback_convert_types(obj: Any) -> Any:
        try:
            import numpy as np
        except ImportError:
            return obj

        if obj is None:
            return None

        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, dict):
            return {
                MetricsLoader._fallback_convert_types(k): MetricsLoader._fallback_convert_types(v)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            converted = [MetricsLoader._fallback_convert_types(item) for item in obj]
            return type(obj)(converted) if isinstance(obj, tuple) else converted

        return obj

    def _load_single_metric(self, metric_name: str, file_path: Path) -> None:
        spec = importlib.util.spec_from_file_location(metric_name, file_path)
        if spec is None or spec.loader is None:
            logger.warning(f"Could not load spec for {metric_name}")
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "calculate"):
            logger.warning(f"Metric '{metric_name}' missing 'calculate' function")
            return

        calculate_func = module.calculate

        sig = inspect.signature(calculate_func)
        supports_context = "context" in sig.parameters

        self._metrics[metric_name] = calculate_func
        self._supports_context[metric_name] = supports_context

        if hasattr(module, "METRIC_INFO"):
            self._metric_info[metric_name] = module.METRIC_INFO
        else:
            self._metric_info[metric_name] = {
                "name": metric_name,
                "description": f"Custom metric: {metric_name}",
                "category": "custom",
            }

        logger.debug(f"Loaded metric: {metric_name} (context={'yes' if supports_context else 'no'})")

    def compute_all(
        self,
        image: Image.Image,
        score: float,
        context: Any = None,
        parallel: bool = True,
    ) -> dict[str, Any]:
        if not self._metrics:
            return {}

        image = self._resize_if_needed(image)

        if parallel and len(self._metrics) > 1:
            return self._compute_parallel(image, score, context)
        else:
            return self._compute_sequential(image, score, context)

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        if max(w, h) > self.MAX_METRIC_SIZE:
            scale = self.MAX_METRIC_SIZE / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def _compute_single(
        self,
        name: str,
        func: MetricFunction,
        image: Image.Image,
        score: float,
        context: Any,
    ) -> tuple[str, dict[str, Any]]:
        try:
            if self._supports_context.get(name, False):
                value = func(image, score, context)
            else:
                value = func(image, score)

            return name, self._convert_result_types(value)
        except Exception as e:
            logger.error(f"Metric '{name}' failed: {e}")
            return name, {"error": str(e)}

    def _compute_parallel(
        self,
        image: Image.Image,
        score: float,
        context: Any,
    ) -> dict[str, Any]:
        results = {}
        max_workers = min(len(self._metrics), self.DEFAULT_MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._compute_single, name, func, image, score, context
                ): name
                for name, func in self._metrics.items()
            }

            for future in as_completed(futures):
                name, value = future.result()
                results[name] = value

        return results

    def _compute_sequential(
        self,
        image: Image.Image,
        score: float,
        context: Any,
    ) -> dict[str, Any]:
        results = {}
        for name, func in self._metrics.items():
            _, value = self._compute_single(name, func, image, score, context)
            results[name] = value
        return results

    def compute_single(
        self,
        metric_name: str,
        image: Image.Image,
        score: float,
        context: Any = None,
    ) -> dict[str, Any] | None:
        func = self._metrics.get(metric_name)
        if func is None:
            return None

        image = self._resize_if_needed(image)
        _, value = self._compute_single(metric_name, func, image, score, context)
        return value

    def get_available_metrics(self) -> dict[str, dict[str, Any]]:
        return self._metric_info.copy()

    def get_metric_names(self) -> list[str]:
        return list(self._metrics.keys())

    @property
    def metric_count(self) -> int:
        return len(self._metrics)

    def is_metric_available(self, name: str) -> bool:
        return name in self._metrics

metrics_engine = MetricsLoader()