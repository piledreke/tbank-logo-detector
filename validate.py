import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from app.inference import YOLODetector


@dataclass
class Box:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def parse_yolo_label(label_path: Path, image_size: Tuple[int, int]) -> List[Box]:
    w, h = image_size
    boxes: List[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        # YOLO format is relative cx, cy, bw, bh in [0,1]
        abs_w = bw * w
        abs_h = bh * h
        cx_abs = cx * w
        cy_abs = cy * h
        x_min = int(max(0, cx_abs - abs_w / 2))
        y_min = int(max(0, cy_abs - abs_h / 2))
        x_max = int(min(w - 1, cx_abs + abs_w / 2))
        y_max = int(min(h - 1, cy_abs + abs_h / 2))
        boxes.append(Box(x_min, y_min, x_max, y_max))
    return boxes


def iou(box_a: Box, box_b: Box) -> float:
    inter_x1 = max(box_a.x_min, box_b.x_min)
    inter_y1 = max(box_a.y_min, box_b.y_min)
    inter_x2 = min(box_a.x_max, box_b.x_max)
    inter_y2 = min(box_a.y_max, box_b.y_max)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    area_a = (box_a.x_max - box_a.x_min + 1) * (box_a.y_max - box_a.y_min + 1)
    area_b = (box_b.x_max - box_b.x_min + 1) * (box_b.y_max - box_b.y_min + 1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_detections_to_gt(dets: List[Box], gts: List[Box], iou_thresh: float) -> Tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    for d in dets:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, g in enumerate(gts):
            if idx in matched_gt:
                continue
            cur_iou = iou(d, g)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = idx
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
    fp = len(dets) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate detector on a labeled dataset")
    parser.add_argument("--images_dir", type=str, default=r"dataset/val/images")
    parser.add_argument("--labels_dir", type=str, default=r"dataset/val/labels")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="validation_metrics.json")
    args = parser.parse_args()

    detector = YOLODetector()

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(Path(args.images_dir).glob(ext))
    image_paths = sorted(image_paths)

    total_tp = total_fp = total_fn = 0

    for img_path in image_paths:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            label_path = Path(args.labels_dir) / (img_path.stem + ".txt")
            gt_boxes = parse_yolo_label(label_path, (w, h))

            # Run detection
            from io import BytesIO

            buf = BytesIO()
            im.save(buf, format="PNG")
            detections = detector.detect(buf.getvalue())
            det_boxes = [Box(d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max) for d in detections]

            tp, fp, fn = match_detections_to_gt(det_boxes, gt_boxes, args.iou)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "images": len(image_paths),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": args.iou,
    }

    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


