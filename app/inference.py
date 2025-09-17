"""
Модуль инференса YOLO-модели для детекции логотипа.

Здесь реализована обёртка над Ultralytics YOLO с возможностью:
- загрузки весов из локального пути или по публичной ссылке (в т.ч. Яндекс.Диск);
- детекции и возврата bbox в абсолютных пикселях;
- логирования времени инференса и параметров запуска.
"""

import os
import tempfile
import time
import logging
from typing import List, Dict, Any

import torch
from ultralytics import YOLO

from .models import BoundingBox, Detection
from .settings import get_settings


class YOLODetector:
    """Обёртка для загрузки модели и выполнения инференса."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logging.getLogger("yolo.inference")
        self.weights_path: str | None = None
        self.model = self._load_model()

    def _ensure_weights(self, weights_path: str) -> str:
        """Проверяет наличие весов локально и при необходимости скачивает по ссылке.

        Возвращает локальный путь к файлу весов.
        """
        if os.path.exists(weights_path):
            return weights_path

        # Если весов нет локально, но задан URL — попробуем скачать
        if self.settings.WEIGHTS_URL:
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            url = self._normalize_yadisk_public_url(self.settings.WEIGHTS_URL)
            self._download_file(url, weights_path)
            return weights_path

        raise FileNotFoundError(
            f"Weights not found at {weights_path} and WEIGHTS_URL is not set"
        )

    def _load_model(self) -> YOLO:
        """Загружает модель YOLO и переносит на выбранное устройство (CPU/GPU)."""
        weights_path = self._ensure_weights(self.settings.DEFAULT_WEIGHTS_PATH)
        self.weights_path = weights_path
        device = (
            0
            if self.settings.DEVICE == "cuda"
            else ("cuda:0" if torch.cuda.is_available() and self.settings.DEVICE == "auto" else "cpu")
        )
        model = YOLO(weights_path)
        model.to(device)
        return model

    def _download_file(self, url: str, dst_path: str) -> None:
        import json
        import urllib.request

        # Поддержка прямой загрузки и публичной ссылки Я.Диск через cloud-api
        if "cloud-api.yandex.net" in url:
            with urllib.request.urlopen(url) as r:
                meta = json.loads(r.read().decode("utf-8"))
            href = meta.get("href")
            if not href:
                raise RuntimeError(f"Yandex Disk API did not return download href. Response: {meta}")
            with urllib.request.urlopen(href) as r2, open(dst_path, "wb") as f:
                f.write(r2.read())
            return

        # Fallback: простая загрузка
        with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
            f.write(r.read())

    def _normalize_yadisk_public_url(self, url: str) -> str:
        # Для публичной ссылки Яндекс.Диск иногда требует прямую ссылку на скачивание.
        # Пользовательскую ссылку можно конвертировать в публичный «download» эндпоинт через disk.yandex.ru/d/...
        # Если ссылка уже скачиваемая — вернём как есть.
        if "yadi.sk" in url or "disk.yandex" in url:
            # API прямой загрузки: https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=<URL>
            from urllib.parse import quote
            public_key = quote(url, safe="")
            return f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={public_key}"
        return url

    def detect(self, image_bytes: bytes) -> List[Detection]:
        """Выполняет детекцию на одном изображении (байты)."""
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()

            results = self.model.predict(
                source=tmp.name,
                conf=self.settings.CONF_THRESHOLD,
                iou=self.settings.IOU_THRESHOLD,
                imgsz=self.settings.IMG_SIZE,
                verbose=False,
                device=self.model.device,
            )

        detections: List[Detection] = []
        for r in results:
            h, w = r.orig_shape
            for b in r.boxes:
                # b.xyxy — абсолютные координаты [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(max(0, v)) for v in b.xyxy[0].tolist()]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
                detections.append(Detection(bbox=bbox))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.logger.info(
            "Инференс: %d детекций, %.1f ms (device=%s, conf=%.2f, iou=%.2f, img=%d)",
            len(detections), elapsed_ms, str(self.model.device), self.settings.CONF_THRESHOLD, self.settings.IOU_THRESHOLD, self.settings.IMG_SIZE
        )
        return detections

    def runtime_info(self) -> Dict[str, Any]:
        """Диагностическая информация для эндпоинта /health."""
        return {
            "device": str(self.model.device),
            "weights_path": self.weights_path,
            "weights_exists": bool(self.weights_path and os.path.exists(self.weights_path)),
            "conf_threshold": self.settings.CONF_THRESHOLD,
            "iou_threshold": self.settings.IOU_THRESHOLD,
            "img_size": self.settings.IMG_SIZE,
        }








