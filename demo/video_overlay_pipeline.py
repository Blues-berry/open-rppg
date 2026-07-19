"""Render Open-rppg heart-rate overlays into local videos.

Pipeline:
1. Decode the input video and feed frames into rppg.Model.
2. Build a time-aligned HR/SQI track from sliding BVP windows.
3. Render a polished heart-rate overlay into a new mp4.
4. Mux the original audio back when ffmpeg is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rppg  # noqa: E402


STABLE_SQI_THRESHOLD = 0.38
PREVIEW_SQI_THRESHOLD = 0.2
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_SAMPLE_SECONDS = 1.0
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class VideoMeta:
    fps: float
    frames_total: int
    duration_s: float | None
    width: int
    height: int
    size_mb: float


@dataclass
class TrackSample:
    ts: float
    hr: float | None
    SQI: float | None
    status: str


def safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def list_videos(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def select_inputs(args) -> list[Path]:
    if args.input:
        return [Path(item).resolve() for item in args.input]
    videos = list_videos(Path(args.input_root))
    if args.recent:
        return videos[: args.recent]
    if args.all:
        return videos
    return videos[:1]


def read_video_meta(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_s = frames_total / fps if frames_total and fps else None
    return VideoMeta(
        fps=fps,
        frames_total=frames_total,
        duration_s=duration_s,
        width=width,
        height=height,
        size_mb=round(path.stat().st_size / 1024 / 1024, 2),
    )


def classify_sample(hr: float | None, sqi: float | None) -> str:
    if hr is None or not 30 <= hr <= 180 or sqi is None:
        return "warming"
    if sqi >= STABLE_SQI_THRESHOLD:
        return "stable"
    if sqi >= PREVIEW_SQI_THRESHOLD:
        return "preview"
    return "low_sqi"


def configure_model(model):
    model.face_detection_threads = 1
    model.face_resampling_threads = 1
    model.face_detect_per_n = 3


def analyze_video(path: Path, meta: VideoMeta, window_s: float, sample_s: float, progress_path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    model = rppg.Model("FacePhys.rlap")
    configure_model(model)

    frame_index = 0
    first_ts = None
    started = time.perf_counter()
    last_progress_at = 0.0

    with model:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = pos_msec / 1000 if pos_msec and pos_msec > 0 else frame_index / meta.fps
            if first_ts is None:
                first_ts = ts
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            model.update_frame(rgb, ts - first_ts)
            frame_index += 1

            now = time.perf_counter()
            if frame_index % 120 == 0 or now - last_progress_at > 8:
                write_json(
                    progress_path,
                    {
                        "state": "analyzing",
                        "video": str(path),
                        "frames_processed": frame_index,
                        "frames_total": meta.frames_total,
                        "percent": progress_percent(frame_index, meta.frames_total),
                        "elapsed_s": round(now - started, 2),
                        "signal_frames": int(getattr(model, "n_signal", 0) or 0),
                        "no_face_count": int((getattr(model, "statistic", {}) or {}).get("null", 0) or 0),
                    },
                )
                last_progress_at = now

    cap.release()

    duration = meta.duration_s or (frame_index / meta.fps if meta.fps else 0)
    track = build_track(model, duration, window_s, sample_s)
    statistic = getattr(model, "statistic", {}) or {}
    return {
        "frames_processed": frame_index,
        "analysis_ms": round((time.perf_counter() - started) * 1000, 1),
        "signal_frames": int(getattr(model, "n_signal", 0) or 0),
        "no_face_count": int(statistic.get("null", 0) or 0),
        "statistic": normalize_statistic(statistic),
        "track": [asdict(sample) for sample in track],
        "summary": summarize_track(track),
    }


def build_track(model, duration_s: float, window_s: float, sample_s: float) -> list[TrackSample]:
    samples: list[TrackSample] = []
    ts = 0.0
    while ts <= max(duration_s, 0.0) + 1e-6:
        if ts < 2.0:
            samples.append(TrackSample(round(ts, 3), None, None, "warming"))
        else:
            start = max(0.0, ts - window_s)
            result = model.hr(start=start, end=ts, return_hrv=False)
            hr = safe_float(result.get("hr") if isinstance(result, dict) else None)
            sqi = safe_float(result.get("SQI") if isinstance(result, dict) else None)
            samples.append(TrackSample(round(ts, 3), hr, sqi, classify_sample(hr, sqi)))
        ts += sample_s
    return samples


def summarize_track(track: list[TrackSample]) -> dict:
    valid = [sample for sample in track if sample.hr is not None and sample.SQI is not None]
    stable = [sample for sample in valid if sample.status == "stable"]
    preview = [sample for sample in valid if sample.status == "preview"]
    hrs = [sample.hr for sample in stable] or [sample.hr for sample in preview] or [sample.hr for sample in valid]
    sqis = [sample.SQI for sample in valid]
    return {
        "samples": len(track),
        "valid_samples": len(valid),
        "stable_samples": len(stable),
        "preview_samples": len(preview),
        "stable_ratio": round(len(stable) / len(track), 3) if track else 0,
        "mean_hr": round(float(np.mean(hrs)), 2) if hrs else None,
        "median_hr": round(float(np.median(hrs)), 2) if hrs else None,
        "max_sqi": round(float(np.max(sqis)), 3) if sqis else None,
        "mean_sqi": round(float(np.mean(sqis)), 3) if sqis else None,
    }


def normalize_statistic(statistic: dict) -> dict:
    normalized = {}
    for key, value in statistic.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        normalized[key] = value
    return normalized


def progress_percent(done: int, total: int) -> float | None:
    if not total:
        return None
    return round(min(100.0, max(0.0, done / total * 100)), 2)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def output_name(path: Path, digest: str) -> str:
    stem = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_" for ch in path.stem)
    stem = "_".join(part for part in stem.split("_") if part)[:64] or "video"
    return f"{stem}_{digest[:10]}_rppg_overlay.mp4"


def sample_for_time(track: list[TrackSample], ts: float, start_index: int) -> tuple[TrackSample, int]:
    if not track:
        return TrackSample(ts=ts, hr=None, SQI=None, status="warming"), start_index
    index = start_index
    while index + 1 < len(track) and track[index + 1].ts <= ts:
        index += 1
    return track[index], index


def render_video(path: Path, meta: VideoMeta, track: list[TrackSample], output_path: Path, progress_path: Path) -> Path:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for render: {path}")

    temp_path = output_path.with_name(output_path.stem + ".video_only.mp4")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        meta.fps,
        (meta.width, meta.height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot open output writer: {temp_path}")

    started = time.perf_counter()
    frame_index = 0
    track_index = 0
    last_progress_at = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = frame_index / meta.fps if meta.fps else 0.0
        sample, track_index = sample_for_time(track, ts, track_index)
        draw_overlay(frame, sample, ts)
        writer.write(frame)
        frame_index += 1

        now = time.perf_counter()
        if frame_index % 180 == 0 or now - last_progress_at > 8:
            write_json(
                progress_path,
                {
                    "state": "rendering",
                    "video": str(path),
                    "frames_rendered": frame_index,
                    "frames_total": meta.frames_total,
                    "percent": progress_percent(frame_index, meta.frames_total),
                    "elapsed_s": round(now - started, 2),
                },
            )
            last_progress_at = now

    cap.release()
    writer.release()
    mux_audio(path, temp_path, output_path)
    return output_path


def mux_audio(source_video: Path, video_only_path: Path, output_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        replace_with_retry(video_only_path, output_path)
        return
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_only_path),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
        unlink_with_retry(video_only_path)
    except Exception:
        replace_with_retry(video_only_path, output_path)


def unlink_with_retry(path: Path, attempts: int = 8, delay_s: float = 0.25):
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == attempts - 1:
                return
            time.sleep(delay_s)


def replace_with_retry(source: Path, target: Path, attempts: int = 8, delay_s: float = 0.25):
    last_error = None
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay_s)
    if target.exists():
        return
    if last_error:
        raise last_error


def draw_overlay(frame: np.ndarray, sample: TrackSample, ts: float):
    h, w = frame.shape[:2]
    scale = max(0.78, min(1.35, w / 1280))
    margin = int(34 * scale)
    panel_w = int(min(w - margin * 2, 520 * scale))
    panel_h = int(170 * scale)
    x = margin
    y = margin

    draw_glass_panel(frame, x, y, panel_w, panel_h, scale)

    center = (x + int(84 * scale), y + int(84 * scale))
    hr_for_anim = sample.hr if sample.hr is not None and 30 <= sample.hr <= 180 else 72
    confidence = sample.SQI if sample.SQI is not None else 0.0
    draw_heart(frame, center, int(34 * scale), hr_for_anim, confidence, ts, sample.status)

    text_x = x + int(152 * scale)
    top = y + int(42 * scale)
    status_label = status_label_for(sample)
    color = status_color(sample.status)
    bpm_text = "--"
    if sample.hr is not None and sample.SQI is not None and sample.SQI >= PREVIEW_SQI_THRESHOLD:
        bpm_text = str(int(round(sample.hr)))

    cv2.putText(frame, status_label, (text_x, top), cv2.FONT_HERSHEY_SIMPLEX, 0.58 * scale, color, max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(frame, bpm_text, (text_x, y + int(100 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.65 * scale, (245, 248, 252), max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.putText(frame, "BPM", (text_x + int(128 * scale), y + int(99 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (190, 206, 214), max(1, int(2 * scale)), cv2.LINE_AA)

    sqi_text = "--" if sample.SQI is None else f"{sample.SQI:.2f}"
    cv2.putText(frame, f"SQI {sqi_text}", (text_x, y + int(135 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (198, 214, 222), max(1, int(2 * scale)), cv2.LINE_AA)
    draw_sqi_bar(frame, text_x + int(118 * scale), y + int(124 * scale), int(180 * scale), int(12 * scale), confidence, scale)


def draw_glass_panel(frame: np.ndarray, x: int, y: int, panel_w: int, panel_h: int, scale: float):
    overlay = frame.copy()
    radius = int(28 * scale)
    draw_filled_round_rect(overlay, (x, y), (x + panel_w, y + panel_h), radius, (10, 16, 22))
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    border = (68, 86, 98)
    draw_round_rect_outline(frame, (x, y), (x + panel_w, y + panel_h), radius, border, max(1, int(2 * scale)))


def draw_filled_round_rect(img, pt1, pt2, radius, color):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_round_rect_outline(img, pt1, pt2, radius, color, thickness):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_heart(frame: np.ndarray, center: tuple[int, int], base_size: int, bpm: float, confidence: float, ts: float, status: str):
    beat_period = 60.0 / max(30.0, min(180.0, bpm))
    phase = (ts % beat_period) / beat_period
    pulse = 1.0 + 0.16 * math.exp(-phase * 11.0) + 0.07 * math.exp(-((phase - 0.24) ** 2) / 0.006)
    if status not in {"stable", "preview"}:
        pulse = 1.0
    size = int(base_size * pulse)
    ring_radius = int(base_size * (1.58 + 0.22 * pulse))

    glow = frame.copy()
    cv2.circle(glow, center, ring_radius + int(12 * pulse), (38, 60, 72), -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.38, frame, 0.62, 0, frame)

    ring_color = status_color(status)
    cv2.circle(frame, center, ring_radius, (38, 50, 58), max(2, base_size // 8), cv2.LINE_AA)
    arc = max(18, int(360 * max(0.03, min(1.0, confidence))))
    cv2.ellipse(frame, center, (ring_radius, ring_radius), -90, 0, arc, ring_color, max(2, base_size // 7), cv2.LINE_AA)

    points = heart_points(center, size)
    heart_layer = frame.copy()
    cv2.fillPoly(heart_layer, [points], (76, 72, 248), cv2.LINE_AA)
    cv2.addWeighted(heart_layer, 0.92, frame, 0.08, 0, frame)
    cv2.polylines(frame, [points], True, (135, 162, 255), max(1, base_size // 13), cv2.LINE_AA)

    highlight_center = (center[0] - int(size * 0.24), center[1] - int(size * 0.32))
    cv2.ellipse(frame, highlight_center, (max(2, size // 6), max(2, size // 11)), -35, 0, 360, (190, 196, 255), -1, cv2.LINE_AA)


def heart_points(center: tuple[int, int], size: int) -> np.ndarray:
    pts = []
    for t in np.linspace(0, 2 * math.pi, 90):
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        px = center[0] + int(x * size / 18)
        py = center[1] + int(y * size / 18) + int(size * 0.08)
        pts.append((px, py))
    return np.array(pts, dtype=np.int32)


def draw_sqi_bar(frame: np.ndarray, x: int, y: int, width: int, height: int, value: float, scale: float):
    value = max(0.0, min(1.0, value or 0.0))
    draw_filled_round_rect(frame, (x, y), (x + width, y + height), max(2, height // 2), (39, 52, 60))
    fill_w = max(height, int(width * value))
    color = (102, 225, 186) if value >= STABLE_SQI_THRESHOLD else ((72, 195, 238) if value >= PREVIEW_SQI_THRESHOLD else (94, 108, 121))
    draw_filled_round_rect(frame, (x, y), (x + fill_w, y + height), max(2, height // 2), color)
    cv2.line(frame, (x + int(width * STABLE_SQI_THRESHOLD), y - int(3 * scale)), (x + int(width * STABLE_SQI_THRESHOLD), y + height + int(3 * scale)), (220, 226, 232), max(1, int(scale)), cv2.LINE_AA)


def status_label_for(sample: TrackSample) -> str:
    if sample.status == "stable":
        return "OPEN-RPPG LIVE"
    if sample.status == "preview":
        return "LOW SQI PREVIEW"
    if sample.status == "low_sqi":
        return "LOW SQI"
    return "BUILDING SIGNAL"


def status_color(status: str) -> tuple[int, int, int]:
    if status == "stable":
        return (102, 225, 186)
    if status == "preview":
        return (72, 195, 238)
    if status == "low_sqi":
        return (112, 120, 255)
    return (175, 190, 200)


def process_one(path: Path, output_dir: Path, args) -> dict:
    digest = hash_file(path)
    meta = read_video_meta(path)
    output_path = output_dir / output_name(path, digest)
    report_path = output_path.with_suffix(".json")
    progress_path = output_path.with_suffix(".progress.json")

    write_json(progress_path, {"state": "queued", "video": str(path), "output": str(output_path)})
    analysis = analyze_video(path, meta, args.window_seconds, args.sample_seconds, progress_path)
    track = [TrackSample(**sample) for sample in analysis["track"]]
    rendered = render_video(path, meta, track, output_path, progress_path)

    report = {
        "input": str(path.resolve()),
        "output": str(rendered.resolve()),
        "hash": digest,
        "metadata": asdict(meta),
        "thresholds": {
            "stable_sqi": STABLE_SQI_THRESHOLD,
            "preview_sqi": PREVIEW_SQI_THRESHOLD,
            "window_seconds": args.window_seconds,
            "sample_seconds": args.sample_seconds,
        },
        "analysis": analysis,
        "interpretation": {
            "usable_for_overlay": analysis["summary"]["stable_samples"] > 0,
            "note": "Remote rPPG pulse estimates are for interaction and content enhancement, not diagnosis or medical decision-making.",
        },
    }
    write_json(report_path, report)
    write_json(progress_path, {"state": "done", "video": str(path), "output": str(output_path), "report": str(report_path)})
    return report


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Render rPPG heart-rate overlays into videos.")
    parser.add_argument("--input", action="append", help="Input video path. Can be passed multiple times.")
    parser.add_argument("--input-root", default=str(REPO_ROOT / "demo" / "video_inputs"), help="Video root for --recent/--all.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "demo" / "video_outputs" / "rendered"), help="Output directory.")
    parser.add_argument("--recent", type=int, help="Process N newest videos from input root.")
    parser.add_argument("--all", action="store_true", help="Process all videos from input root.")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS, help="Sliding HR/SQI window.")
    parser.add_argument("--sample-seconds", type=float, default=DEFAULT_SAMPLE_SECONDS, help="Track sample interval.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = dedupe_paths(select_inputs(args))
    if not inputs:
        raise SystemExit("No input videos found.")

    reports = []
    manifest_path = output_dir / "manifest.json"
    for index, path in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] Processing {path}", flush=True)
        try:
            report = process_one(path, output_dir, args)
            reports.append(report)
            print(f"  -> {report['output']}", flush=True)
        except Exception as exc:
            failure = {"input": str(path), "error": f"{type(exc).__name__}: {exc}"}
            reports.append(failure)
            print(f"  !! {failure['error']}", flush=True)
        write_json(manifest_path, {"generated_at": time.time(), "reports": reports})

    print(json.dumps({"manifest": str(manifest_path.resolve()), "count": len(reports)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
