"""
Скрипт визуализации предсказаний: рисует bbox на изображениях и сохраняет результат.

Примеры использования:
  python scripts/visualize_predictions.py --images dataset/test/images --out out/viz --max 50
"""

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw

from app.inference import YOLODetector


def draw_bbox(im: Image.Image, bbox: Tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(im)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize predictions on images")
    p.add_argument("--images", required=True, help="Directory with input images")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--max", type=int, default=0, help="Limit number of images (0 = all)")
    args = p.parse_args()

    src_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector()

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in exts:
        images.extend(src_dir.glob(ext))
    images = sorted(images)
    if args.max > 0:
        images = images[: args.max]

    for pth in images:
        with Image.open(pth) as im:
            imc = im.convert("RGB")
            detections = detector.detect(image_bytes=_to_png_bytes(imc))
            for d in detections:
                bb = d.bbox
                draw_bbox(imc, (bb.x_min, bb.y_min, bb.x_max, bb.y_max))
            imc.save(out_dir / pth.name)


def _to_png_bytes(im: Image.Image) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()


