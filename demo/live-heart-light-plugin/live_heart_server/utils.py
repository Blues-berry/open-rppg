from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

from .config import (
    AGENT_DEFAULT_API_KEY,
    AGENT_DEFAULT_AUTH_TOKEN,
    AGENT_DEFAULT_BASE_URL,
    AGENT_DEFAULT_MODEL,
    AGENT_DEFAULT_PROTOCOL,
    AGENT_DEFAULT_VERSION,
    AGENT_LOCAL_CONFIG_PATH,
)

def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def clamp_int(value, min_value, max_value, default):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return min(max_value, max(min_value, value))


def clamp_float(value, min_value=0.0, max_value=1.0, default=0.0):
    value = safe_float(value)
    if value is None:
        return default
    return min(max_value, max(min_value, value))


def as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() not in {"0", "false", "no", "off"}


def load_agent_config():
    config = {
        "protocol": AGENT_DEFAULT_PROTOCOL,
        "base_url": AGENT_DEFAULT_BASE_URL,
        "auth_token": AGENT_DEFAULT_AUTH_TOKEN,
        "api_key": AGENT_DEFAULT_API_KEY,
        "model": AGENT_DEFAULT_MODEL,
        "version": AGENT_DEFAULT_VERSION,
    }
    try:
        if AGENT_LOCAL_CONFIG_PATH.exists():
            with AGENT_LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
                local_config = json.load(handle)
            if isinstance(local_config, dict):
                for key in config:
                    if isinstance(local_config.get(key), str):
                        config[key] = local_config[key]
    except Exception:
        pass
    return config


def save_agent_config(settings: dict):
    """Persist user-owned Agent settings without ever returning the secret to UI."""
    allowed = {"protocol", "base_url", "api_key", "auth_token", "model", "version"}
    current = load_agent_config()
    for key, value in settings.items():
        if key in allowed and isinstance(value, str):
            current[key] = value.strip()
    AGENT_LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = AGENT_LOCAL_CONFIG_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(current, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(AGENT_LOCAL_CONFIG_PATH)


def agent_config_value(config: dict, env_name: str, key: str):
    if env_name in os.environ:
        return os.environ.get(env_name, "").strip()
    return str(config.get(key) or "").strip()

def camera_api():
    return cv2.CAP_DSHOW if sys.platform.startswith("win32") else 0
