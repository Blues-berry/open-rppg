"""Benchmark rPPG predictions against heart-rate overlays in livestream clips.

This script creates a reproducible offline loop:
discover/acquire clips -> extract visible HR as ground truth -> run FacePhys ->
render predictions back into the video -> evaluate per-second accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "demo"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import video_overlay_pipeline as overlay  # noqa: E402


BENCH_ROOT = DEMO_ROOT / "video_benchmark"
SOURCES_DIR = BENCH_ROOT / "sources"
RAW_DIR = BENCH_ROOT / "raw"
TRUTH_DIR = BENCH_ROOT / "truth"
PREDICTION_DIR = BENCH_ROOT / "predictions"
RENDER_DIR = BENCH_ROOT / "rendered"
REPORT_DIR = BENCH_ROOT / "reports"
PREVIEW_DIR = BENCH_ROOT / "previews"

TRUTH_MIN_BPM = 30
TRUTH_MAX_BPM = 220
TRUTH_RELIABLE_COVERAGE = 0.70
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_SAMPLE_SECONDS = 1.0
DEFAULT_SHIFT_SECONDS = 5
VIDEO_EXTENSIONS = overlay.VIDEO_EXTENSIONS


@dataclass
class OcrObservation:
    frame_index: int
    ts: float
    bpm: int | None
    confidence: float
    roi: list[int] | None
    raw_text: str
    accepted: bool


class PaddleBpmReader:
    def __init__(self):
        from paddleocr import PaddleOCR  # type: ignore

        self.ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)

    def read(self, crop: np.ndarray) -> tuple[int | None, float, str]:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(rgb, cls=False)
        texts: list[str] = []
        confidences: list[float] = []
        for page in result or []:
            for item in page or []:
                if len(item) < 2:
                    continue
                text = str(item[1][0])
                conf = overlay.safe_float(item[1][1]) or 0.0
                texts.append(text)
                confidences.append(conf)
        joined = " ".join(texts)
        bpm = choose_bpm_from_text(joined)
        if bpm is None:
            return None, 0.0, joined
        return bpm, float(np.mean(confidences)) if confidences else 0.5, joined


class CvDigitBpmReader:
    def __init__(self):
        self.templates = build_digit_templates()

    def read(self, crop: np.ndarray) -> tuple[int | None, float, str]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        masks = [red_mask(hsv), bright_digit_mask(crop)]
        best: tuple[int | None, float, str] = (None, 0.0, "")
        for mask in masks:
            bpm, confidence, raw_text = self._read_mask(mask, crop.shape[1], crop.shape[0])
            if bpm is not None and confidence > best[1]:
                best = (bpm, confidence, raw_text)
        return best

    def _read_mask(self, mask: np.ndarray, width: int, height: int) -> tuple[int | None, float, str]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        components = component_boxes(mask)
        if not components:
            return None, 0.0, ""

        line = select_digit_line(components, width, height)
        if not line:
            return None, 0.0, ""

        digits: list[str] = []
        scores: list[float] = []
        boxes = sorted(line, key=lambda item: item[0])
        for x, y, w, h, _area in boxes:
            glyph = mask[max(0, y - 1) : min(mask.shape[0], y + h + 1), max(0, x - 1) : min(mask.shape[1], x + w + 1)]
            digit, score = match_digit(glyph, self.templates)
            if digit is not None and score >= 0.45:
                digits.append(digit)
                scores.append(score)

        if not 2 <= len(digits) <= 3:
            return None, 0.0, "".join(digits)
        bpm = int("".join(digits))
        if not TRUTH_MIN_BPM <= bpm <= TRUTH_MAX_BPM:
            return None, 0.0, str(bpm)
        return bpm, float(min(scores)) if scores else 0.0, str(bpm)


def choose_bpm_from_text(text: str) -> int | None:
    candidates = [int(match) for match in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", text)]
    for value in candidates:
        if TRUTH_MIN_BPM <= value <= TRUTH_MAX_BPM:
            return value
    return None


def red_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 35, 35), (12, 255, 255)),
        cv2.inRange(hsv, (165, 35, 35), (180, 255, 255)),
    )


def bright_digit_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    return bright


def component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= 18:
            boxes.append((x, y, w, h, area))
    return boxes


def select_digit_line(
    components: list[tuple[int, int, int, int, int]],
    width: int,
    height: int,
) -> list[tuple[int, int, int, int, int]]:
    candidates = []
    for box in components:
        x, y, w, h, area = box
        aspect = w / max(h, 1)
        if h < max(10, height * 0.07):
            continue
        if h > height * 0.62:
            continue
        if aspect > 0.95:
            continue
        if x < width * 0.05 and w > h * 0.8:
            continue
        candidates.append(box)
    best: list[tuple[int, int, int, int, int]] = []
    best_score = -1.0
    for anchor in candidates:
        ax, ay, aw, ah, _ = anchor
        cy = ay + ah / 2
        line = []
        for box in candidates:
            x, y, w, h, area = box
            overlap = min(y + h, ay + ah) - max(y, ay)
            if overlap > min(h, ah) * 0.35 or abs((y + h / 2) - cy) <= max(8, ah * 0.55):
                line.append(box)
        if not 2 <= len(line) <= 4:
            continue
        xs = [item[0] for item in line]
        spread = max(xs) - min(xs)
        score = len(line) * 10 + sum(item[4] for item in line) / 100 - spread / max(width, 1)
        if score > best_score:
            best = line
            best_score = score
    return best


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    if len(xs):
        binary = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    canvas = np.zeros((48, 32), dtype=np.uint8)
    h, w = binary.shape[:2]
    scale = min(26 / max(w, 1), 42 / max(h, 1))
    resized = cv2.resize(binary, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    y = (canvas.shape[0] - resized.shape[0]) // 2
    x = (canvas.shape[1] - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def build_digit_templates() -> list[tuple[str, np.ndarray]]:
    templates: list[tuple[str, np.ndarray]] = []
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]
    for digit in range(10):
        for font in fonts:
            for font_scale in (1.25, 1.45, 1.65):
                for thickness in (2, 3):
                    canvas = np.zeros((80, 60), dtype=np.uint8)
                    cv2.putText(canvas, str(digit), (5, 60), font, font_scale, 255, thickness, cv2.LINE_AA)
                    templates.append((str(digit), normalize_glyph(canvas)))
    return templates


def match_digit(mask: np.ndarray, templates: list[tuple[str, np.ndarray]]) -> tuple[str | None, float]:
    glyph = normalize_glyph(mask)
    best_digit = None
    best_score = -1.0
    a = glyph.flatten().astype(np.float32)
    for digit, template in templates:
        b = template.flatten().astype(np.float32)
        score = correlation(a, b)
        if score > best_score:
            best_digit = digit
            best_score = score
    return best_digit, best_score


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return -1.0
    return float(np.dot(a, b) / denom)


def make_bpm_reader(prefer_paddle: bool = True):
    if prefer_paddle:
        try:
            return PaddleBpmReader(), "paddleocr"
        except Exception:
            pass
    return CvDigitBpmReader(), "opencv_digit_template"


def detect_hr_roi(path: Path, meta: overlay.VideoMeta) -> tuple[list[int] | None, dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    duration = meta.duration_s or 0.0
    sample_times = sorted({0.0, min(3.0, duration), min(10.0, duration), min(30.0, duration), max(0.0, duration - 5.0)})
    candidates = []
    for ts in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        for roi, score, kind in detect_roi_candidates(frame):
            candidates.append({"ts": ts, "roi": roi, "score": score, "kind": kind})
    cap.release()
    if not candidates:
        return None, {"sample_times": sample_times, "candidates": []}
    white_panel_candidates = [
        item
        for item in candidates
        if item["kind"] == "white_panel"
        and item["roi"][1] > meta.height * 0.25
        and item["roi"][1] < meta.height * 0.45
        and item["roi"][3] < meta.height * 0.18
    ]
    purple_badge_candidates = [item for item in candidates if item["kind"] == "purple_badge"]
    right_purple_badges = [
        item
        for item in purple_badge_candidates
        if item["roi"][0] > meta.width * 0.42 and item["roi"][1] < meta.height * 0.38
    ]
    pool = white_panel_candidates or right_purple_badges or purple_badge_candidates or candidates
    pool.sort(key=lambda item: item["score"], reverse=True)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return pool[0]["roi"], {"sample_times": sample_times, "candidates": candidates[:12], "preferred_kind": pool[0]["kind"]}


def detect_roi_candidates(frame: np.ndarray) -> list[tuple[list[int], float, str]]:
    h, w = frame.shape[:2]
    candidates: list[tuple[list[int], float, str]] = []

    white = cv2.inRange(frame, (205, 205, 205), (255, 255, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < max(900, w * h * 0.0007):
            continue
        if cw < 40 or ch < 30:
            continue
        roi = expand_roi([x, y, cw, ch], w, h, margin=12)
        crop = crop_roi(frame, roi)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red_pixels = int(np.count_nonzero(red_mask(hsv)))
        score = area / 1000 + red_pixels / 50
        if red_pixels >= 40:
            candidates.append((roi, score, "white_panel"))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = red_mask(hsv)
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)), iterations=1)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < max(600, w * h * 0.0005) or cw < 30 or ch < 18:
            continue
        roi = expand_roi([x, y, cw, ch], w, h, margin=28)
        candidates.append((roi, area / 700, "red_text"))

    purple = cv2.inRange(hsv, (125, 45, 55), (175, 255, 255))
    purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < 800 or cw < 20 or ch < 20:
            continue
        if y > h * 0.55:
            continue
        roi = expand_badge_roi([x, y, cw, ch], w, h)
        crop = crop_roi(frame, roi)
        bright_pixels = int(np.count_nonzero(bright_digit_mask(crop)))
        score = area / 120 + bright_pixels / 35
        if bright_pixels >= 25:
            candidates.append((roi, score, "purple_badge"))
    return candidates


def expand_roi(roi: list[int], frame_w: int, frame_h: int, margin: int) -> list[int]:
    x, y, w, h = roi
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_w, x + w + margin)
    y2 = min(frame_h, y + h + margin)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def expand_badge_roi(roi: list[int], frame_w: int, frame_h: int) -> list[int]:
    x, y, w, h = roi
    x1 = max(0, x - 24)
    y1 = max(0, y - 24)
    x2 = min(frame_w, x + w + max(120, int(w * 1.7)))
    y2 = min(frame_h, y + h + 28)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def crop_roi(frame: np.ndarray, roi: list[int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def extract_truth(input_path: Path, output_path: Path | None, prefer_paddle: bool, frame_step: int) -> dict:
    meta = overlay.read_video_meta(input_path)
    roi, roi_debug = detect_hr_roi(input_path, meta)
    reader, reader_name = make_bpm_reader(prefer_paddle)
    output_path = output_path or truth_output_path(input_path)
    preview_path = PREVIEW_DIR / f"{safe_stem(input_path)}_truth_roi.jpg"
    timeline_path = PREVIEW_DIR / f"{safe_stem(input_path)}_truth_timeline.jpg"
    csv_path = output_path.with_suffix(".frames.csv")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    observations: list[OcrObservation] = []
    previous_bpm: int | None = None
    frame_index = 0
    accepted_count = 0
    sampled_count = 0
    first_preview_frame = None
    started = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if first_preview_frame is None:
            first_preview_frame = frame.copy()
        if roi is not None and frame_index % max(1, frame_step) == 0:
            ts = frame_index / meta.fps if meta.fps else 0.0
            crop = crop_roi(frame, roi)
            bpm, confidence, raw_text = reader.read(crop)
            accepted = False
            if bpm is not None:
                accepted = is_continuous_bpm(previous_bpm, bpm)
                if accepted:
                    previous_bpm = bpm
                    accepted_count += 1
            sampled_count += 1
            observations.append(
                OcrObservation(
                    frame_index=frame_index,
                    ts=round(ts, 4),
                    bpm=bpm if accepted else None,
                    confidence=round(float(confidence), 4),
                    roi=roi,
                    raw_text=raw_text,
                    accepted=accepted,
                )
            )
        frame_index += 1
    cap.release()

    coverage = accepted_count / sampled_count if sampled_count else 0.0
    per_second = aggregate_truth_by_second(observations)
    status = "truth_ready" if coverage >= TRUTH_RELIABLE_COVERAGE else "truth_unreliable"

    if first_preview_frame is not None:
        draw_roi_preview(first_preview_frame, roi, preview_path)
    draw_truth_timeline(per_second, timeline_path)
    write_truth_csv(csv_path, observations)

    payload = {
        "input": str(input_path.resolve()),
        "status": status,
        "reader": reader_name,
        "metadata": asdict(meta),
        "roi": roi,
        "roi_debug": roi_debug,
        "frame_step": frame_step,
        "coverage": round(coverage, 4),
        "sampled_frames": sampled_count,
        "accepted_frames": accepted_count,
        "per_second": per_second,
        "observations": [asdict(obs) for obs in observations],
        "artifacts": {
            "roi_preview": str(preview_path.resolve()),
            "timeline": str(timeline_path.resolve()),
            "frames_csv": str(csv_path.resolve()),
        },
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    overlay.write_json(output_path, payload)
    return payload


def is_continuous_bpm(previous: int | None, current: int) -> bool:
    if not TRUTH_MIN_BPM <= current <= TRUTH_MAX_BPM:
        return False
    if previous is None:
        return True
    return abs(current - previous) <= max(35, int(previous * 0.35))


def aggregate_truth_by_second(observations: list[OcrObservation]) -> list[dict]:
    buckets: dict[int, list[OcrObservation]] = {}
    for obs in observations:
        if obs.accepted and obs.bpm is not None:
            buckets.setdefault(int(math.floor(obs.ts)), []).append(obs)
    rows = []
    for second in sorted(buckets):
        values = [obs.bpm for obs in buckets[second] if obs.bpm is not None]
        confidences = [obs.confidence for obs in buckets[second]]
        if not values:
            continue
        rows.append(
            {
                "second": second,
                "bpm": int(round(float(np.median(values)))),
                "samples": len(values),
                "confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            }
        )
    return rows


def write_truth_csv(path: Path, observations: list[OcrObservation]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "ts", "bpm", "confidence", "accepted", "raw_text", "roi"])
        writer.writeheader()
        for obs in observations:
            row = asdict(obs)
            row["roi"] = json.dumps(row["roi"], ensure_ascii=False)
            writer.writerow(row)


def draw_roi_preview(frame: np.ndarray, roi: list[int] | None, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = frame.copy()
    if roi:
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (82, 238, 190), 4, cv2.LINE_AA)
        cv2.putText(preview, "HR OCR ROI", (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (82, 238, 190), 3, cv2.LINE_AA)
    cv2.imwrite(str(path), preview)


def draw_truth_timeline(per_second: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((360, 1200, 3), (20, 25, 30), dtype=np.uint8)
    cv2.putText(canvas, "OCR Ground Truth BPM", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 242, 245), 2, cv2.LINE_AA)
    if not per_second:
        cv2.putText(canvas, "No reliable OCR samples", (36, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 120, 135), 3, cv2.LINE_AA)
        cv2.imwrite(str(path), canvas)
        return
    seconds = [row["second"] for row in per_second]
    bpms = [row["bpm"] for row in per_second]
    min_s, max_s = min(seconds), max(seconds)
    min_b, max_b = min(bpms), max(bpms)
    lo = max(TRUTH_MIN_BPM, min_b - 10)
    hi = min(TRUTH_MAX_BPM, max_b + 10)
    prev = None
    for row in per_second:
        x = int(60 + (row["second"] - min_s) / max(1, max_s - min_s) * 1080)
        y = int(310 - (row["bpm"] - lo) / max(1, hi - lo) * 220)
        if prev:
            cv2.line(canvas, prev, (x, y), (89, 214, 238), 3, cv2.LINE_AA)
        prev = (x, y)
    cv2.putText(canvas, f"{lo}-{hi} BPM | {len(per_second)} seconds", (36, 334), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 198, 205), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def predict(input_path: Path, output_path: Path | None, window_s: float, sample_s: float) -> dict:
    output_path = output_path or prediction_output_path(input_path)
    progress_path = output_path.with_suffix(".progress.json")
    meta = overlay.read_video_meta(input_path)
    analysis = overlay.analyze_video(input_path, meta, window_s, sample_s, progress_path)
    payload = {
        "input": str(input_path.resolve()),
        "metadata": asdict(meta),
        "thresholds": {
            "stable_sqi": overlay.STABLE_SQI_THRESHOLD,
            "preview_sqi": overlay.PREVIEW_SQI_THRESHOLD,
            "window_seconds": window_s,
            "sample_seconds": sample_s,
        },
        "analysis": analysis,
    }
    overlay.write_json(output_path, payload)
    return payload


def evaluate(truth_path: Path, prediction_path: Path, output_dir: Path | None, max_shift_s: int) -> dict:
    truth = read_json(truth_path)
    prediction = read_json(prediction_path)
    output_dir = output_dir or REPORT_DIR / safe_stem(Path(prediction["input"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = build_comparison_rows(truth, prediction, shift_s=0)
    raw_metrics = compute_metrics(comparisons)
    shifted = []
    for shift in range(-max_shift_s, max_shift_s + 1):
        rows = build_comparison_rows(truth, prediction, shift_s=shift)
        metrics = compute_metrics(rows)
        metrics["shift_s"] = shift
        shifted.append((metrics, rows))
    evaluable_shifted = [item for item in shifted if item[0].get("count", 0)]
    if evaluable_shifted:
        best_metrics, best_rows = min(evaluable_shifted, key=lambda item: item[0].get("MAE", float("inf")))
    else:
        best_metrics, best_rows = {"count": 0, "shift_s": None}, []

    comparison_csv = output_dir / "per_second_comparison.csv"
    write_comparison_csv(comparison_csv, comparisons, best_rows)

    status = "evaluated" if raw_metrics.get("count", 0) else "not_evaluable"
    report = {
        "status": status,
        "truth": str(truth_path.resolve()),
        "prediction": str(prediction_path.resolve()),
        "input": prediction.get("input"),
        "truth_status": truth.get("status"),
        "raw_metrics": raw_metrics,
        "best_shift_metrics": best_metrics,
        "prediction_summary": prediction.get("analysis", {}).get("summary", {}),
        "truth_coverage": truth.get("coverage", 0.0),
        "artifacts": {
            "per_second_comparison": str(comparison_csv.resolve()),
        },
        "note": "Ground truth is OCR of an existing livestream HR overlay, not a medical-grade measurement.",
    }
    report_path = output_dir / "evaluation.json"
    overlay.write_json(report_path, report)
    write_evaluation_markdown(output_dir / "evaluation.md", report)
    return report


def build_comparison_rows(truth: dict, prediction: dict, shift_s: int) -> list[dict]:
    truth_by_second = {int(row["second"]): row for row in truth.get("per_second", [])}
    pred_by_second = {}
    for sample in prediction.get("analysis", {}).get("track", []):
        second = int(round(float(sample["ts"])))
        pred_by_second[second] = sample
    rows = []
    for second in sorted(truth_by_second):
        truth_row = truth_by_second[second]
        pred = pred_by_second.get(second + shift_s)
        pred_hr = overlay.safe_float(pred.get("hr") if pred else None)
        sqi = overlay.safe_float(pred.get("SQI") if pred else None)
        error = abs(pred_hr - truth_row["bpm"]) if pred_hr is not None else None
        rows.append(
            {
                "second": second,
                "prediction_second": second + shift_s,
                "truth_bpm": truth_row["bpm"],
                "pred_bpm": pred_hr,
                "SQI": sqi,
                "status": pred.get("status") if pred else "missing",
                "abs_error": error,
                "truth_confidence": truth_row.get("confidence"),
                "truth_samples": truth_row.get("samples"),
            }
        )
    return rows


def compute_metrics(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("pred_bpm") is not None and row.get("truth_bpm") is not None]
    if not valid:
        return {"count": 0}
    errors = np.array([float(row["abs_error"]) for row in valid], dtype=np.float64)
    truths = np.array([float(row["truth_bpm"]) for row in valid], dtype=np.float64)
    preds = np.array([float(row["pred_bpm"]) for row in valid], dtype=np.float64)
    sqis = [float(row["SQI"]) for row in valid if row.get("SQI") is not None]
    metrics = {
        "count": len(valid),
        "MAE": round(float(np.mean(errors)), 3),
        "RMSE": round(float(np.sqrt(np.mean(errors ** 2))), 3),
        "MedianAE": round(float(np.median(errors)), 3),
        "MAPE": round(float(np.mean(errors / np.maximum(truths, 1)) * 100), 3),
        "within_5_bpm": round(float(np.mean(errors <= 5)), 3),
        "within_10_bpm": round(float(np.mean(errors <= 10)), 3),
        "SQI_mean": round(float(np.mean(sqis)), 3) if sqis else None,
        "SQI_max": round(float(np.max(sqis)), 3) if sqis else None,
    }
    metrics["pearson"] = round(float(np.corrcoef(truths, preds)[0, 1]), 3) if len(valid) > 1 and np.std(preds) > 0 and np.std(truths) > 0 else None
    for status in ("stable", "preview", "low_sqi"):
        subset = [float(row["abs_error"]) for row in valid if row.get("status") == status]
        metrics[f"{status}_count"] = len(subset)
        metrics[f"{status}_MAE"] = round(float(np.mean(subset)), 3) if subset else None
    return metrics


def write_comparison_csv(path: Path, raw_rows: list[dict], shifted_rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    shifted_by_second = {row["second"]: row for row in shifted_rows}
    fieldnames = [
        "second",
        "truth_bpm",
        "pred_bpm",
        "SQI",
        "status",
        "abs_error",
        "shifted_prediction_second",
        "shifted_pred_bpm",
        "shifted_abs_error",
        "truth_confidence",
        "truth_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            shifted = shifted_by_second.get(row["second"], {})
            writer.writerow(
                {
                    "second": row.get("second"),
                    "truth_bpm": row.get("truth_bpm"),
                    "pred_bpm": row.get("pred_bpm"),
                    "SQI": row.get("SQI"),
                    "status": row.get("status"),
                    "abs_error": row.get("abs_error"),
                    "truth_confidence": row.get("truth_confidence"),
                    "truth_samples": row.get("truth_samples"),
                    "shifted_prediction_second": shifted.get("prediction_second"),
                    "shifted_pred_bpm": shifted.get("pred_bpm"),
                    "shifted_abs_error": shifted.get("abs_error"),
                }
            )


def write_evaluation_markdown(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = report.get("raw_metrics", {})
    shifted = report.get("best_shift_metrics", {})
    content = [
        "# rPPG Benchmark Evaluation",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Truth status: `{report.get('truth_status')}`",
        f"- OCR coverage: `{report.get('truth_coverage')}`",
        f"- Raw MAE: `{raw.get('MAE')}` BPM",
        f"- Best shift: `{shifted.get('shift_s')}` s",
        f"- Best-shift MAE: `{shifted.get('MAE')}` BPM",
        f"- Within 10 BPM: `{shifted.get('within_10_bpm')}`",
        "",
        "The ground truth is OCR of the visible livestream HR overlay. It is not medical-grade ground truth.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def render(input_path: Path, prediction_path: Path, truth_path: Path | None, output_path: Path | None) -> dict:
    prediction = read_json(prediction_path)
    truth = read_json(truth_path) if truth_path and truth_path.exists() else None
    meta = overlay.read_video_meta(input_path)
    output_path = output_path or render_output_path(input_path)
    progress_path = output_path.with_suffix(".progress.json")
    track = [overlay.TrackSample(**sample) for sample in prediction.get("analysis", {}).get("track", [])]
    truth_by_second = {int(row["second"]): row for row in (truth or {}).get("per_second", [])}
    truth_roi = (truth or {}).get("roi")
    render_benchmark_video(input_path, meta, track, truth_by_second, truth_roi, output_path, progress_path)
    result = {
        "input": str(input_path.resolve()),
        "prediction": str(prediction_path.resolve()),
        "truth": str(truth_path.resolve()) if truth_path else None,
        "output": str(output_path.resolve()),
    }
    overlay.write_json(output_path.with_suffix(".json"), result)
    return result


def render_benchmark_video(
    path: Path,
    meta: overlay.VideoMeta,
    track: list[overlay.TrackSample],
    truth_by_second: dict[int, dict],
    truth_roi: list[int] | None,
    output_path: Path,
    progress_path: Path,
):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for render: {path}")
    temp_path = output_path.with_name(output_path.stem + ".video_only.mp4")
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot open output writer: {temp_path}")

    frame_index = 0
    track_index = 0
    started = time.perf_counter()
    last_progress_at = 0.0
    overlay_origin = choose_model_overlay_origin(meta.width, meta.height, truth_roi)
    gt_origin = choose_gt_origin(meta.width, meta.height, truth_roi, overlay_origin)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = frame_index / meta.fps if meta.fps else 0.0
        sample, track_index = overlay.sample_for_time(track, ts, track_index)
        draw_model_overlay_at(frame, sample, ts, overlay_origin)
        truth_sample = truth_by_second.get(int(math.floor(ts)))
        draw_gt_badge(frame, truth_sample, gt_origin)
        writer.write(frame)
        frame_index += 1
        now = time.perf_counter()
        if frame_index % 180 == 0 or now - last_progress_at > 8:
            overlay.write_json(
                progress_path,
                {
                    "state": "rendering",
                    "frames_rendered": frame_index,
                    "frames_total": meta.frames_total,
                    "percent": overlay.progress_percent(frame_index, meta.frames_total),
                    "elapsed_s": round(now - started, 2),
                },
            )
            last_progress_at = now
    cap.release()
    writer.release()
    overlay.mux_audio(path, temp_path, output_path)


def choose_model_overlay_origin(width: int, height: int, truth_roi: list[int] | None) -> tuple[int, int]:
    scale = max(0.78, min(1.35, width / 1280))
    margin = int(34 * scale)
    panel = [margin, margin, int(min(width - margin * 2, 520 * scale)), int(170 * scale)]
    if truth_roi and rect_iou(panel, truth_roi) > 0.03:
        return width - panel[2] - margin, margin
    return margin, margin


def choose_gt_origin(width: int, height: int, truth_roi: list[int] | None, model_origin: tuple[int, int]) -> tuple[int, int]:
    scale = max(0.78, min(1.35, width / 1280))
    badge = [width - int(238 * scale) - int(30 * scale), height - int(86 * scale) - int(34 * scale), int(238 * scale), int(86 * scale)]
    if truth_roi and rect_iou(badge, truth_roi) > 0.03:
        badge[0] = int(30 * scale)
    return badge[0], badge[1]


def rect_iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1, aw * ah)


def draw_model_overlay_at(frame: np.ndarray, sample: overlay.TrackSample, ts: float, origin: tuple[int, int]):
    h, w = frame.shape[:2]
    scale = max(0.78, min(1.35, w / 1280))
    x, y = origin
    panel_w = int(min(w - int(34 * scale) * 2, 520 * scale))
    panel_h = int(170 * scale)
    overlay.draw_glass_panel(frame, x, y, panel_w, panel_h, scale)
    center = (x + int(84 * scale), y + int(84 * scale))
    hr_for_anim = sample.hr if sample.hr is not None and 30 <= sample.hr <= 180 else 72
    confidence = sample.SQI if sample.SQI is not None else 0.0
    overlay.draw_heart(frame, center, int(34 * scale), hr_for_anim, confidence, ts, sample.status)
    text_x = x + int(152 * scale)
    color = overlay.status_color(sample.status)
    bpm_text = "--"
    if sample.hr is not None and sample.SQI is not None and sample.SQI >= overlay.PREVIEW_SQI_THRESHOLD:
        bpm_text = str(int(round(sample.hr)))
    cv2.putText(frame, "OUR RPPG", (text_x, y + int(42 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.58 * scale, color, max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(frame, bpm_text, (text_x, y + int(100 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.65 * scale, (245, 248, 252), max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.putText(frame, "BPM", (text_x + int(128 * scale), y + int(99 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (190, 206, 214), max(1, int(2 * scale)), cv2.LINE_AA)
    sqi_text = "--" if sample.SQI is None else f"{sample.SQI:.2f}"
    cv2.putText(frame, f"SQI {sqi_text}", (text_x, y + int(135 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (198, 214, 222), max(1, int(2 * scale)), cv2.LINE_AA)
    overlay.draw_sqi_bar(frame, text_x + int(118 * scale), y + int(124 * scale), int(180 * scale), int(12 * scale), confidence, scale)


def draw_gt_badge(frame: np.ndarray, truth_sample: dict | None, origin: tuple[int, int]):
    h, w = frame.shape[:2]
    scale = max(0.78, min(1.35, w / 1280))
    x, y = origin
    bw = int(238 * scale)
    bh = int(86 * scale)
    overlay.draw_glass_panel(frame, x, y, bw, bh, scale)
    value = "--" if not truth_sample else str(int(truth_sample["bpm"]))
    cv2.putText(frame, "VISIBLE GT", (x + int(22 * scale), y + int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (245, 116, 132), max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(frame, value, (x + int(22 * scale), y + int(68 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (245, 248, 252), max(2, int(3 * scale)), cv2.LINE_AA)
    cv2.putText(frame, "BPM", (x + int(118 * scale), y + int(67 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, (190, 206, 214), max(1, int(2 * scale)), cv2.LINE_AA)


def discover(query: str, limit: int, output_path: Path | None) -> dict:
    output_path = output_path or SOURCES_DIR / f"discover_{int(time.time())}.json"
    candidates = []
    errors = []
    candidates.extend(discover_bilibili(query, limit, errors))
    candidates.extend(discover_ytdlp(query, limit, errors))
    payload = {
        "query": query,
        "limit": limit,
        "generated_at": time.time(),
        "candidates": candidates[:limit],
        "errors": errors,
        "note": "Only public, tool-supported videos should be acquired. No login, DRM, cookie, or anti-scraping bypass is used.",
    }
    overlay.write_json(output_path, payload)
    return payload


def discover_bilibili(query: str, limit: int, errors: list[dict]) -> list[dict]:
    url = "https://api.bilibili.com/x/web-interface/search/type?" + urllib.parse.urlencode(
        {"search_type": "video", "keyword": query, "page": 1}
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 benchmark research",
            "Referer": "https://search.bilibili.com/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        errors.append({"engine": "bilibili_api", "error": f"{type(exc).__name__}: {exc}"})
        return []
    items = payload.get("data", {}).get("result", []) or []
    candidates = []
    for item in items[:limit]:
        url = item.get("arcurl") or f"https://www.bilibili.com/video/{item.get('bvid', '')}"
        candidates.append(
            {
                "platform": "bilibili",
                "title": strip_html(item.get("title", "")),
                "url": url,
                "id": item.get("bvid") or item.get("aid"),
                "duration": item.get("duration"),
                "source": "bilibili_api",
                "status": "candidate",
            }
        )
    return candidates


def discover_ytdlp(query: str, limit: int, errors: list[dict]) -> list[dict]:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        errors.append({"engine": "yt-dlp", "error": f"yt-dlp unavailable: {exc}"})
        return []
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as exc:
        errors.append({"engine": "yt-dlp", "error": f"{type(exc).__name__}: {exc}"})
        return []
    candidates = []
    for entry in (info or {}).get("entries", []) or []:
        candidates.append(
            {
                "platform": entry.get("extractor_key") or "ytsearch",
                "title": entry.get("title"),
                "url": entry.get("url") or entry.get("webpage_url"),
                "id": entry.get("id"),
                "duration": entry.get("duration"),
                "source": "yt-dlp",
                "status": "candidate",
            }
        )
    return candidates


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def acquire(sources_path: Path, limit: int, output_dir: Path | None) -> dict:
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = read_json(sources_path).get("candidates", [])
    results = []
    for candidate in sources[:limit]:
        url = candidate.get("url")
        if not url:
            results.append({**candidate, "status": "skipped", "error": "missing url"})
            continue
        try:
            downloaded = download_with_ytdlp(url, output_dir)
            results.append({**candidate, "status": "downloaded", "local_path": str(downloaded.resolve())})
        except Exception as exc:
            results.append({**candidate, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    payload = {"sources": str(sources_path.resolve()), "output_dir": str(output_dir.resolve()), "results": results}
    acquire_report = SOURCES_DIR / f"acquire_{int(time.time())}.json"
    overlay.write_json(acquire_report, payload)
    payload["report"] = str(acquire_report.resolve())
    return payload


def download_with_ytdlp(url: str, output_dir: Path) -> Path:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        raise RuntimeError("yt-dlp is not installed. Install requirements-benchmark.txt first.") from exc
    before = set(output_dir.glob("*"))
    options = {
        "outtmpl": str(output_dir / "%(title).80s_%(id)s.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    after = set(output_dir.glob("*"))
    new_files = [p for p in after - before if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not new_files:
        new_files = sorted(output_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not new_files:
        raise RuntimeError("download finished but no video file was found")
    return sorted(new_files, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def run_all(args) -> dict:
    inputs = select_inputs(args)
    if not inputs:
        raise SystemExit("No input videos found.")
    reports = []
    for index, input_path in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] Benchmarking {input_path}", flush=True)
        try:
            report = process_benchmark_video(input_path, args)
            reports.append(report)
            print(f"  -> {report.get('render', {}).get('output')}", flush=True)
        except Exception as exc:
            failure = {"input": str(input_path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            reports.append(failure)
            print(f"  !! {failure['error']}", flush=True)
        write_summary_reports(reports)
    return write_summary_reports(reports)


def process_benchmark_video(input_path: Path, args) -> dict:
    truth = extract_truth(input_path, None, prefer_paddle=not args.no_paddle, frame_step=args.truth_frame_step)
    prediction = predict(input_path, None, args.window_seconds, args.sample_seconds)
    truth_path = truth_output_path(input_path)
    prediction_path = prediction_output_path(input_path)
    evaluation = evaluate(truth_path, prediction_path, REPORT_DIR / safe_stem(input_path), args.max_shift_seconds)
    rendered = render(input_path, prediction_path, truth_path, None)
    return {
        "input": str(input_path.resolve()),
        "truth": str(truth_path.resolve()),
        "prediction": str(prediction_path.resolve()),
        "evaluation": evaluation,
        "render": rendered,
    }


def write_summary_reports(reports: list[dict]) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = REPORT_DIR / "benchmark_summary.csv"
    fieldnames = [
        "input",
        "status",
        "truth_status",
        "truth_coverage",
        "raw_MAE",
        "best_shift_s",
        "best_shift_MAE",
        "within_10_bpm",
        "stable_samples",
        "preview_samples",
        "render_output",
        "error",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            evaluation = report.get("evaluation", {})
            raw = evaluation.get("raw_metrics", {})
            shifted = evaluation.get("best_shift_metrics", {})
            pred_summary = evaluation.get("prediction_summary", {})
            writer.writerow(
                {
                    "input": report.get("input"),
                    "status": evaluation.get("status") or report.get("status"),
                    "truth_status": evaluation.get("truth_status"),
                    "truth_coverage": evaluation.get("truth_coverage"),
                    "raw_MAE": raw.get("MAE"),
                    "best_shift_s": shifted.get("shift_s"),
                    "best_shift_MAE": shifted.get("MAE"),
                    "within_10_bpm": shifted.get("within_10_bpm"),
                    "stable_samples": pred_summary.get("stable_samples"),
                    "preview_samples": pred_summary.get("preview_samples"),
                    "render_output": report.get("render", {}).get("output"),
                    "error": report.get("error"),
                }
            )
    summary_md = REPORT_DIR / "benchmark_report.md"
    lines = ["# rPPG Livestream Clip Benchmark", "", "| Input | Status | Truth | MAE | Best Shift | Render |", "|---|---:|---:|---:|---:|---|"]
    for report in reports:
        evaluation = report.get("evaluation", {})
        raw = evaluation.get("raw_metrics", {})
        shifted = evaluation.get("best_shift_metrics", {})
        lines.append(
            f"| {Path(report.get('input', '')).name} | {evaluation.get('status') or report.get('status')} | "
            f"{evaluation.get('truth_status')} | {raw.get('MAE')} | {shifted.get('shift_s')}s / {shifted.get('MAE')} | "
            f"{Path(report.get('render', {}).get('output', '')).name if report.get('render') else ''} |"
        )
    lines.extend(["", "Visible HR overlays are OCR-derived reference values, not medical-grade ground truth."])
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "generated_at": time.time(),
        "reports": reports,
        "artifacts": {
            "benchmark_summary": str(summary_csv.resolve()),
            "benchmark_report": str(summary_md.resolve()),
        },
    }
    overlay.write_json(REPORT_DIR / "benchmark_summary.json", payload)
    return payload


def select_inputs(args) -> list[Path]:
    if args.input:
        return overlay.dedupe_paths([Path(item).resolve() for item in args.input])
    root = Path(args.input_root)
    videos = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if args.recent:
        return videos[: args.recent]
    if args.all:
        return videos
    return videos[:1]


def truth_output_path(input_path: Path) -> Path:
    return TRUTH_DIR / f"{safe_stem(input_path)}_truth.json"


def prediction_output_path(input_path: Path) -> Path:
    return PREDICTION_DIR / f"{safe_stem(input_path)}_prediction.json"


def render_output_path(input_path: Path) -> Path:
    return RENDER_DIR / f"{safe_stem(input_path)}_benchmark_overlay.mp4"


def safe_stem(path: Path) -> str:
    stem = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_" for ch in path.stem)
    stem = "_".join(part for part in stem.split("_") if part)
    return stem[:80] or f"video_{int(time.time())}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs():
    for path in (SOURCES_DIR, RAW_DIR, TRUTH_DIR, PREDICTION_DIR, RENDER_DIR, REPORT_DIR, PREVIEW_DIR):
        path.mkdir(parents=True, exist_ok=True)


def verify_rendered_videos(root: Path) -> list[dict]:
    rows = []
    for video in sorted(root.glob("*_benchmark_overlay.mp4")):
        cap = cv2.VideoCapture(str(video))
        rows.append(
            {
                "path": str(video.resolve()),
                "opened": cap.isOpened(),
                "fps": overlay.safe_float(cap.get(cv2.CAP_PROP_FPS)),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "streams": ffprobe_streams(video),
            }
        )
        cap.release()
    return rows


def ffprobe_streams(video: Path) -> list[dict]:
    if not shutil.which("ffprobe"):
        return []
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type,codec_name", "-of", "json", str(video)],
            check=True,
            text=True,
            capture_output=True,
        )
        return json.loads(completed.stdout).get("streams", [])
    except Exception:
        return []


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Benchmark rPPG against visible livestream heart-rate overlays.")
    sub = parser.add_subparsers(dest="command", required=True)

    discover_parser = sub.add_parser("discover")
    discover_parser.add_argument("--query", default="直播 心率 bpm")
    discover_parser.add_argument("--limit", type=int, default=30)
    discover_parser.add_argument("--output")

    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument("--sources", required=True)
    acquire_parser.add_argument("--limit", type=int, default=5)
    acquire_parser.add_argument("--output-dir")

    truth_parser = sub.add_parser("extract-truth")
    truth_parser.add_argument("--input", required=True)
    truth_parser.add_argument("--output")
    truth_parser.add_argument("--truth-frame-step", type=int, default=1)
    truth_parser.add_argument("--no-paddle", action="store_true")

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output")
    predict_parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    predict_parser.add_argument("--sample-seconds", type=float, default=DEFAULT_SAMPLE_SECONDS)

    render_parser = sub.add_parser("render")
    render_parser.add_argument("--input", required=True)
    render_parser.add_argument("--prediction", required=True)
    render_parser.add_argument("--truth")
    render_parser.add_argument("--output")

    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--truth", required=True)
    evaluate_parser.add_argument("--prediction", required=True)
    evaluate_parser.add_argument("--output-dir")
    evaluate_parser.add_argument("--max-shift-seconds", type=int, default=DEFAULT_SHIFT_SECONDS)

    all_parser = sub.add_parser("all")
    all_parser.add_argument("--input", action="append")
    all_parser.add_argument("--input-root", default=str(DEMO_ROOT / "video_inputs"))
    all_parser.add_argument("--recent", type=int)
    all_parser.add_argument("--all", action="store_true")
    all_parser.add_argument("--truth-frame-step", type=int, default=1)
    all_parser.add_argument("--no-paddle", action="store_true")
    all_parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    all_parser.add_argument("--sample-seconds", type=float, default=DEFAULT_SAMPLE_SECONDS)
    all_parser.add_argument("--max-shift-seconds", type=int, default=DEFAULT_SHIFT_SECONDS)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--render-root", default=str(RENDER_DIR))

    args = parser.parse_args()
    if args.command == "discover":
        result = discover(args.query, args.limit, Path(args.output) if args.output else None)
    elif args.command == "acquire":
        result = acquire(Path(args.sources), args.limit, Path(args.output_dir) if args.output_dir else None)
    elif args.command == "extract-truth":
        result = extract_truth(Path(args.input), Path(args.output) if args.output else None, prefer_paddle=not args.no_paddle, frame_step=args.truth_frame_step)
    elif args.command == "predict":
        result = predict(Path(args.input), Path(args.output) if args.output else None, args.window_seconds, args.sample_seconds)
    elif args.command == "render":
        result = render(Path(args.input), Path(args.prediction), Path(args.truth) if args.truth else None, Path(args.output) if args.output else None)
    elif args.command == "evaluate":
        result = evaluate(Path(args.truth), Path(args.prediction), Path(args.output_dir) if args.output_dir else None, args.max_shift_seconds)
    elif args.command == "all":
        result = run_all(args)
    elif args.command == "verify":
        result = {"videos": verify_rendered_videos(Path(args.render_root))}
    else:
        raise SystemExit(f"Unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
