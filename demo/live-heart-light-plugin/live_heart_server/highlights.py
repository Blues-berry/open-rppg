from __future__ import annotations

import threading
import time

import numpy as np

from .config import (
    HIGHLIGHT_BASELINE_MAX_AGE_SECONDS,
    HIGHLIGHT_BASELINE_MIN_AGE_SECONDS,
    HIGHLIGHT_EMA_ALPHA,
    HIGHLIGHT_EXIT_PROMINENCE_HR,
    HIGHLIGHT_EXIT_SECONDS,
    HIGHLIGHT_LIMIT,
    HIGHLIGHT_MAX_CLIP_SECONDS,
    HIGHLIGHT_MAX_SAMPLE_GAP_SECONDS,
    HIGHLIGHT_MAX_SMOOTH_JUMP_PER_SECOND,
    HIGHLIGHT_MAX_HR,
    HIGHLIGHT_MERGE_GAP_SECONDS,
    HIGHLIGHT_MIN_CLIP_SECONDS,
    HIGHLIGHT_MIN_COVERAGE,
    HIGHLIGHT_MIN_HR,
    HIGHLIGHT_MIN_SAMPLES,
    HIGHLIGHT_OVERLAP_RATIO,
    HIGHLIGHT_POST_ROLL_SECONDS,
    HIGHLIGHT_PRE_ROLL_SECONDS,
    HIGHLIGHT_STRONG_PROMINENCE_HR,
    HIGHLIGHT_TRIGGER_PROMINENCE_HR,
    HIGHLIGHT_TRIGGER_RISE_10S,
    PREVIEW_SQI_THRESHOLD,
)
from .utils import clamp_float, safe_float

class HighlightTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self.session_id = None
        self.started_at = None
        self.samples = []
        self.items = []
        self.last_metric_seq = None

    def start_session(self, session_id: str, started_at: float | None):
        with self._lock:
            self.session_id = session_id
            self.started_at = started_at
            self.samples = []
            self.items = []
            self.last_metric_seq = None

    def reset_metrics(self):
        with self._lock:
            self.samples = []
            self.items = []
            self.last_metric_seq = None

    def observe(self, capture_status: dict, model_status: dict, output: dict):
        metric_seq = model_status.get("metric_seq")
        try:
            metric_seq = int(metric_seq)
        except (TypeError, ValueError):
            return
        if metric_seq <= 0:
            return
        capture_started = safe_float(capture_status.get("started_at"))
        metric_at = safe_float(model_status.get("metric_captured_at")) or time.time()
        if capture_started is None:
            return
        elapsed = max(0.0, metric_at - capture_started)
        result = model_status.get("result") or {}
        hr = safe_float(result.get("hr"))
        sqi = safe_float(result.get("SQI"))
        bpm = safe_float(output.get("bpm"))
        valid = (
            capture_status.get("state") == "running"
            and bool(model_status.get("has_face"))
            and hr is not None
            and HIGHLIGHT_MIN_HR <= hr <= HIGHLIGHT_MAX_HR
            and sqi is not None
            and sqi >= PREVIEW_SQI_THRESHOLD
        )
        sample = {
            "metric_seq": metric_seq,
            "elapsed": round(elapsed, 3),
            "wall_time": metric_at,
            "hr": safe_float(hr),
            "bpm": safe_float(bpm),
            "SQI": safe_float(sqi),
            "status": output.get("status"),
            "confidence": safe_float(output.get("confidence")),
            "has_face": bool(model_status.get("has_face")),
            "valid": bool(valid),
        }
        with self._lock:
            if metric_seq == self.last_metric_seq:
                return
            self.last_metric_seq = metric_seq
            self.samples.append(sample)
            self._refresh_locked()

    def status(self, recorder: RecordingManager):
        with self._lock:
            items = [dict(item) for item in self.items]
        for item in items:
            export = recorder.export_info(item["id"])
            item["exportable"] = item.get("status") == "confirmed" and recorder.can_export(item)
            item["export_state"] = export.get("state")
            item["export_url"] = export.get("download_url")
            item["export_path"] = export.get("path")
        return {"items": items}

    def find(self, highlight_id: str):
        with self._lock:
            for item in self.items:
                if item.get("id") == highlight_id:
                    return dict(item)
        return None

    def snapshot(self):
        with self._lock:
            return {
                "session_id": self.session_id,
                "samples": [dict(sample) for sample in self.samples],
                "items": [dict(item) for item in self.items],
            }

    def _refresh_locked(self):
        enriched = self._enriched_samples()
        confirmed_events, observing_event = self._detect_events(enriched)
        confirmed_items = [
            item
            for item in (self._build_item(event, enriched, "confirmed") for event in confirmed_events)
            if item is not None
        ]
        selected = self._select_confirmed(confirmed_items)
        observing_items = []
        if observing_event is not None:
            observing = self._build_item(observing_event, enriched, "observing")
            if observing is not None:
                observing_items.append(observing)
        self.items = sorted(selected + observing_items, key=lambda item: (item["start"], item["status"] != "confirmed"))

    def _enriched_samples(self):
        enriched = []
        recent_raw_hr = []
        previous_smooth = None
        previous_elapsed = None
        for sample in self.samples:
            item = dict(sample)
            hr = safe_float(item.get("hr"))
            elapsed = safe_float(item.get("elapsed"))
            smooth_hr = None
            jump_ok = True
            if item.get("valid") and hr is not None and elapsed is not None:
                recent_raw_hr.append(hr)
                median_hr = float(np.median(recent_raw_hr[-5:]))
                if previous_smooth is None or previous_elapsed is None:
                    smooth_hr = median_hr
                else:
                    dt = max(0.001, elapsed - previous_elapsed)
                    delta = median_hr - previous_smooth
                    max_delta = max(4.0, HIGHLIGHT_MAX_SMOOTH_JUMP_PER_SECOND * dt)
                    if abs(delta) > max_delta and abs(delta) > 12.0:
                        jump_ok = False
                        smooth_hr = previous_smooth + float(np.sign(delta)) * max_delta
                    else:
                        smooth_hr = previous_smooth * (1.0 - HIGHLIGHT_EMA_ALPHA) + median_hr * HIGHLIGHT_EMA_ALPHA
                previous_smooth = smooth_hr
                previous_elapsed = elapsed

            item["smooth_hr"] = safe_float(smooth_hr)
            item["jump_ok"] = bool(jump_ok)
            item["valid_for_event"] = bool(item.get("valid") and smooth_hr is not None and jump_ok)
            enriched.append(item)

        for index, item in enumerate(enriched):
            smooth_hr = safe_float(item.get("smooth_hr"))
            elapsed = safe_float(item.get("elapsed"))
            if not item.get("valid_for_event") or smooth_hr is None or elapsed is None:
                item["baseline_hr"] = None
                item["prominence_hr"] = None
                item["rise_10s"] = None
                continue

            prior = [
                sample
                for sample in enriched[:index]
                if sample.get("valid_for_event") and safe_float(sample.get("smooth_hr")) is not None
            ]
            baseline_pool = []
            for sample in prior:
                sample_elapsed = safe_float(sample.get("elapsed"))
                if sample_elapsed is None:
                    continue
                if elapsed - HIGHLIGHT_BASELINE_MAX_AGE_SECONDS <= sample_elapsed <= elapsed - HIGHLIGHT_BASELINE_MIN_AGE_SECONDS:
                    baseline_pool.append(safe_float(sample.get("smooth_hr")))
            if len(baseline_pool) < 6:
                baseline_pool = [
                    safe_float(sample.get("smooth_hr"))
                    for sample in prior
                    if safe_float(sample.get("elapsed")) is not None and safe_float(sample.get("elapsed")) <= elapsed - 8.0
                ]
            baseline_pool = [value for value in baseline_pool if value is not None]
            baseline_hr = float(np.median(baseline_pool)) if baseline_pool else smooth_hr

            rise_candidates = [
                sample
                for sample in prior
                if safe_float(sample.get("elapsed")) is not None and safe_float(sample.get("elapsed")) <= elapsed - 8.0
            ]
            if rise_candidates:
                target = elapsed - 10.0
                reference = min(
                    rise_candidates,
                    key=lambda sample: abs((safe_float(sample.get("elapsed")) or 0.0) - target),
                )
                reference_hr = safe_float(reference.get("smooth_hr"))
                rise_10s = smooth_hr - reference_hr if reference_hr is not None else 0.0
            else:
                rise_10s = 0.0

            item["baseline_hr"] = baseline_hr
            item["prominence_hr"] = smooth_hr - baseline_hr
            item["rise_10s"] = rise_10s
        return enriched

    def _detect_events(self, samples):
        if not samples:
            return [], None
        raw_events = []
        active = None
        latest_elapsed = safe_float(samples[-1].get("elapsed")) or 0.0
        confirm_wait = max(HIGHLIGHT_EXIT_SECONDS, HIGHLIGHT_POST_ROLL_SECONDS)

        for sample in samples:
            elapsed = safe_float(sample.get("elapsed"))
            if elapsed is None:
                continue
            trigger = self._is_trigger_sample(sample)
            above_exit = (
                sample.get("valid_for_event")
                and safe_float(sample.get("prominence_hr")) is not None
                and safe_float(sample.get("prominence_hr")) >= HIGHLIGHT_EXIT_PROMINENCE_HR
            )

            if active is None:
                if trigger:
                    active = self._new_event(sample)
                continue

            if sample.get("valid_for_event"):
                self._update_event_peak(active, sample)
            if trigger or above_exit:
                active["last_active_at"] = elapsed
                if trigger:
                    active["last_trigger_at"] = elapsed
            if elapsed - active["last_active_at"] >= confirm_wait:
                raw_events.append(active)
                active = None

        observing = None
        if active is not None:
            if latest_elapsed - active["last_active_at"] >= confirm_wait:
                raw_events.append(active)
            else:
                observing = active
        return self._merge_events(raw_events), observing

    def _new_event(self, sample):
        elapsed = safe_float(sample.get("elapsed")) or 0.0
        return {
            "trigger_at": elapsed,
            "last_trigger_at": elapsed,
            "last_active_at": elapsed,
            "peak_at": elapsed,
            "peak_hr": safe_float(sample.get("smooth_hr")) or safe_float(sample.get("hr")) or 0.0,
            "baseline_hr": safe_float(sample.get("baseline_hr")),
            "max_prominence": max(0.0, safe_float(sample.get("prominence_hr")) or 0.0),
            "max_rise_10s": max(0.0, safe_float(sample.get("rise_10s")) or 0.0),
            "reason": self._reason_for_sample(sample),
        }

    @staticmethod
    def _update_event_peak(event, sample):
        smooth_hr = safe_float(sample.get("smooth_hr"))
        if smooth_hr is None:
            return
        prominence = max(0.0, safe_float(sample.get("prominence_hr")) or 0.0)
        rise_10s = max(0.0, safe_float(sample.get("rise_10s")) or 0.0)
        if smooth_hr >= event["peak_hr"]:
            event["peak_hr"] = smooth_hr
            event["peak_at"] = safe_float(sample.get("elapsed")) or event["peak_at"]
            event["baseline_hr"] = safe_float(sample.get("baseline_hr"))
            event["reason"] = HighlightTracker._reason_for_sample(sample)
        event["max_prominence"] = max(event["max_prominence"], prominence)
        event["max_rise_10s"] = max(event["max_rise_10s"], rise_10s)

    @staticmethod
    def _is_trigger_sample(sample):
        if not sample.get("valid_for_event"):
            return False
        prominence = safe_float(sample.get("prominence_hr"))
        rise_10s = safe_float(sample.get("rise_10s"))
        if prominence is None or rise_10s is None:
            return False
        return (
            prominence >= HIGHLIGHT_STRONG_PROMINENCE_HR
            or (prominence >= HIGHLIGHT_TRIGGER_PROMINENCE_HR and rise_10s >= HIGHLIGHT_TRIGGER_RISE_10S)
        )

    @staticmethod
    def _reason_for_sample(sample):
        prominence = safe_float(sample.get("prominence_hr")) or 0.0
        rise_10s = safe_float(sample.get("rise_10s")) or 0.0
        if prominence >= HIGHLIGHT_STRONG_PROMINENCE_HR:
            return "峰值突出"
        if rise_10s >= HIGHLIGHT_TRIGGER_RISE_10S:
            return "快速上升"
        return "连续高位"

    @staticmethod
    def _merge_events(events):
        merged = []
        for event in sorted(events, key=lambda item: item["trigger_at"]):
            if not merged or event["trigger_at"] - merged[-1]["last_active_at"] > HIGHLIGHT_MERGE_GAP_SECONDS:
                merged.append(dict(event))
                continue
            previous = merged[-1]
            previous["last_active_at"] = max(previous["last_active_at"], event["last_active_at"])
            previous["last_trigger_at"] = max(previous["last_trigger_at"], event["last_trigger_at"])
            previous["trigger_at"] = min(previous["trigger_at"], event["trigger_at"])
            if event["peak_hr"] >= previous["peak_hr"]:
                previous["peak_hr"] = event["peak_hr"]
                previous["peak_at"] = event["peak_at"]
                previous["baseline_hr"] = event["baseline_hr"]
                previous["reason"] = event["reason"]
            previous["max_prominence"] = max(previous["max_prominence"], event["max_prominence"])
            previous["max_rise_10s"] = max(previous["max_rise_10s"], event["max_rise_10s"])
        return merged

    def _build_item(self, event, samples, status):
        latest_elapsed = safe_float(samples[-1].get("elapsed")) if samples else 0.0
        raw_start = max(0.0, event["trigger_at"] - HIGHLIGHT_PRE_ROLL_SECONDS)
        raw_end = (
            event["last_active_at"] + HIGHLIGHT_POST_ROLL_SECONDS
            if status == "confirmed"
            else max(latest_elapsed or event["last_active_at"], event["peak_at"])
        )
        start, end = self._dynamic_bounds(raw_start, raw_end, event["peak_at"], status)
        segment = [
            sample
            for sample in samples
            if (safe_float(sample.get("elapsed")) is not None and start <= safe_float(sample.get("elapsed")) <= end)
        ]
        valid_segment = [
            sample
            for sample in segment
            if sample.get("valid_for_event") and safe_float(sample.get("smooth_hr")) is not None
        ]
        total_count = len(segment)
        coverage = len(valid_segment) / total_count if total_count else 0.0
        gaps = [
            (safe_float(valid_segment[index].get("elapsed")) or 0.0) - (safe_float(valid_segment[index - 1].get("elapsed")) or 0.0)
            for index in range(1, len(valid_segment))
        ]
        max_gap = max(gaps) if gaps else 0.0
        continuity = 1.0 if max_gap <= HIGHLIGHT_MAX_SAMPLE_GAP_SECONDS else max(0.0, 1.0 - (max_gap - HIGHLIGHT_MAX_SAMPLE_GAP_SECONDS) / HIGHLIGHT_MAX_SAMPLE_GAP_SECONDS)
        sqis = [safe_float(sample.get("SQI")) for sample in valid_segment if safe_float(sample.get("SQI")) is not None]
        hrs = [safe_float(sample.get("smooth_hr")) for sample in valid_segment if safe_float(sample.get("smooth_hr")) is not None]
        if status == "confirmed" and not self._confirmable_event(event, valid_segment, coverage, max_gap):
            return None
        if not hrs:
            return None

        avg_sqi = float(np.mean(sqis)) if sqis else 0.0
        delta_hr = max(hrs) - min(hrs)
        auc = self._prominence_auc(valid_segment)
        quality = clamp_float(avg_sqi * coverage * continuity)
        prominence = max(0.0, event["max_prominence"])
        rise_10s = max(0.0, event["max_rise_10s"])
        score = prominence * 2.0 + rise_10s * 1.3 + min(20.0, auc / 8.0) + quality * 14.0 + coverage * 6.0
        confidence = clamp_float(
            min(1.0, prominence / 12.0) * 0.38
            + min(1.0, rise_10s / 8.0) * 0.2
            + quality * 0.32
            + coverage * 0.1
        )
        level = "high" if confidence >= 0.72 or (prominence >= 10.0 and rise_10s >= 5.0) else "medium"
        center = event["peak_at"]
        return {
            "id": f"hl-{int(round(event['peak_at'] * 10)):06d}",
            "start": round(start, 2),
            "end": round(end, 2),
            "center": round(center, 2),
            "score": round(float(score), 3),
            "min_hr": round(float(min(hrs)), 1),
            "max_hr": round(float(max(hrs)), 1),
            "delta_hr": round(float(delta_hr), 1),
            "avg_sqi": round(avg_sqi, 3),
            "sample_count": len(valid_segment),
            "status": status,
            "level": level,
            "peak_at": round(float(event["peak_at"]), 2),
            "peak_hr": round(float(event["peak_hr"]), 1),
            "baseline_hr": round(float(event["baseline_hr"]), 1) if event.get("baseline_hr") is not None else None,
            "prominence_hr": round(float(prominence), 1),
            "rise_10s": round(float(rise_10s), 1),
            "coverage": round(float(coverage), 3),
            "quality": round(float(quality), 3),
            "confidence": round(float(confidence), 3),
            "reason": event.get("reason") or "连续高位",
        }

    @staticmethod
    def _confirmable_event(event, valid_segment, coverage, max_gap):
        if len(valid_segment) < HIGHLIGHT_MIN_SAMPLES:
            return False
        if coverage < HIGHLIGHT_MIN_COVERAGE:
            return False
        if max_gap > HIGHLIGHT_MAX_SAMPLE_GAP_SECONDS:
            return False
        if event["max_prominence"] < HIGHLIGHT_TRIGGER_PROMINENCE_HR:
            return False
        if event["max_prominence"] < HIGHLIGHT_STRONG_PROMINENCE_HR and event["max_rise_10s"] < HIGHLIGHT_TRIGGER_RISE_10S:
            return False
        active_duration = event["last_active_at"] - event["trigger_at"]
        return active_duration >= 1.5 or event["max_prominence"] >= HIGHLIGHT_STRONG_PROMINENCE_HR

    @staticmethod
    def _dynamic_bounds(raw_start, raw_end, peak_at, status):
        start = max(0.0, float(raw_start))
        end = max(start, float(raw_end))
        duration = end - start
        if status == "confirmed" and duration < HIGHLIGHT_MIN_CLIP_SECONDS:
            missing = HIGHLIGHT_MIN_CLIP_SECONDS - duration
            start = max(0.0, start - missing * 0.45)
            end = start + HIGHLIGHT_MIN_CLIP_SECONDS
        if end - start > HIGHLIGHT_MAX_CLIP_SECONDS:
            start = max(0.0, float(peak_at) - 18.0)
            end = start + HIGHLIGHT_MAX_CLIP_SECONDS
            if peak_at > end - HIGHLIGHT_POST_ROLL_SECONDS:
                end = float(peak_at) + HIGHLIGHT_POST_ROLL_SECONDS
                start = max(0.0, end - HIGHLIGHT_MAX_CLIP_SECONDS)
        return start, max(end, start + (HIGHLIGHT_MIN_CLIP_SECONDS if status == "confirmed" else 0.0))

    @staticmethod
    def _prominence_auc(valid_segment):
        if len(valid_segment) < 2:
            return 0.0
        auc = 0.0
        for index in range(1, len(valid_segment)):
            left = valid_segment[index - 1]
            right = valid_segment[index]
            dt = max(0.0, (safe_float(right.get("elapsed")) or 0.0) - (safe_float(left.get("elapsed")) or 0.0))
            left_prom = max(0.0, (safe_float(left.get("prominence_hr")) or 0.0) - HIGHLIGHT_EXIT_PROMINENCE_HR)
            right_prom = max(0.0, (safe_float(right.get("prominence_hr")) or 0.0) - HIGHLIGHT_EXIT_PROMINENCE_HR)
            auc += (left_prom + right_prom) * 0.5 * dt
        return auc

    @staticmethod
    def _select_confirmed(candidates):
        selected = []
        candidates.sort(key=lambda item: (item["score"], item["confidence"], item["peak_at"]), reverse=True)
        for candidate in candidates:
            if all(HighlightTracker._overlap_ratio(candidate, existing) <= HIGHLIGHT_OVERLAP_RATIO for existing in selected):
                selected.append(candidate)
            if len(selected) >= HIGHLIGHT_LIMIT:
                break
        selected.sort(key=lambda item: item["start"])
        return selected

    @staticmethod
    def _overlap_ratio(a: dict, b: dict):
        overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
        if overlap <= 0:
            return 0.0
        shortest = max(0.001, min(a["end"] - a["start"], b["end"] - b["start"]))
        return overlap / shortest


