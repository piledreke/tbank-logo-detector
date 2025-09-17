import os
import tempfile
from typing import List

import torch
from ultralytics import YOLO

from .models import BoundingBox, Detection
from .settings import get_settings


class YOLODetector:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = self._load_model()

    def _ensure_weights(self, weights_path: str) -> str:
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
        weights_path = self._ensure_weights(self.settings.DEFAULT_WEIGHTS_PATH)
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
            h, w = r.orig_shape  # height, width
            for b in r.boxes:
                # b.xyxy is tensor [[x1,y1,x2,y2]] in absolute floats
                x1, y1, x2, y2 = [int(max(0, v)) for v in b.xyxy[0].tolist()]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
                detections.append(Detection(bbox=bbox))

        return detections








