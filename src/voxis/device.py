from __future__ import annotations

from typing import Any


def mlx_device_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {"backend": "mlx"}
    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        payload["selected"] = str(mx.default_device())
    except Exception as exc:
        payload["selected"] = "unknown"
        payload["error_type"] = type(exc).__name__
    return payload
