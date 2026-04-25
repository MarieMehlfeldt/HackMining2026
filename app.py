"""Flask server to receive and serve validated NumPy matrices."""

import os
from threading import Lock
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from dataclasses import dataclass

app = Flask(__name__)




# Configure which sender IP is accepted (can be overridden via env var).
ALLOWED_SENDER_IP = os.getenv("ALLOWED_SENDER_IP", "127.0.0.1")

EXPECTED_SHAPES = {
    "coords": (16, 720, 3),
    "intensity": (16 * 720, 1),
    "reflectivity": (16, 720, 1),
}

_latest_matrices: dict[str, np.ndarray] | None = None
_old_matrices: dict[str, np.ndarray] | None = None
_store_lock = Lock()


def _get_client_ip() -> str:
    """Return client IP, honoring proxy forwarding headers if present."""
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or ""


def _parse_matrix(value: Any, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert incoming JSON matrix data to numpy and validate/reshape shape."""
    array = np.asarray(value)
    expected_size = int(np.prod(expected_shape))

    if array.size != expected_size:
        raise ValueError(
            f"{name} must contain exactly {expected_size} values; got {array.size}."
        )

    try:
        array = array.reshape(expected_shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} cannot be reshaped to {expected_shape}; got input shape {array.shape}."
        ) from exc

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric.")

    return array.astype(np.float32, copy=False)


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "allowed_sender_ip": ALLOWED_SENDER_IP})


@app.post("/matrices")
def receive_matrices() -> Any:
    """Receive and validate the three required matrices from the allowed sender IP."""
    client_ip = _get_client_ip()
    if client_ip != ALLOWED_SENDER_IP:
        return jsonify({"error": f"Forbidden sender IP: {client_ip}"}), 403

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected JSON object payload."}), 400

    missing = [name for name in EXPECTED_SHAPES if name not in payload]
    if missing:
        return jsonify({"error": f"Missing required matrices: {missing}"}), 400

    try:
        matrices = {
            name: _parse_matrix(payload[name], expected_shape, name)
            for name, expected_shape in EXPECTED_SHAPES.items()
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _store_lock:
        global _latest_matrices
        global _old_matrices
        _old_matrices = _latest_matrices
        _latest_matrices = matrices

    return jsonify(
        {
            "status": "received",
            "sender_ip": client_ip,
            "shapes": {name: list(arr.shape) for name, arr in matrices.items()},
        }
    )


@app.get("/matrices/shapes")
def get_latest_shapes() -> Any:
    """Fetch only matrix shapes from the most recent valid upload."""
    with _store_lock:
        if _latest_matrices is None:
            return jsonify({"error": "No matrices received yet."}), 404
        shapes = {name: list(arr.shape) for name, arr in _latest_matrices.items()}
    return jsonify({"shapes": shapes})


@app.get("/matrices")
def get_latest_matrices() -> Any:
    """Fetch the most recently uploaded matrices as JSON lists."""
    with _store_lock:
        if _latest_matrices is None:
            return jsonify({"error": "No matrices received yet."}), 404
        data = {name: arr.tolist() for name, arr in _latest_matrices.items()}
    return jsonify(data)


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
