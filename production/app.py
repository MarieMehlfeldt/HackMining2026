"""Flask server to receive and serve validated NumPy matrices."""

import os
import time
from threading import Lock, Thread
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, render_template
from dataclasses import dataclass
from .pipeline import process_frame, AppSettings
import requests
from . import cloud_state


from threading import Lock, Thread

# Add this near your other globals at the top of app.py
_processing_lock = Lock()

app = Flask(__name__)

SETTINGS = AppSettings()

# Configure which sender IP is accepted (can be overridden via env var).
ALLOWED_SENDER_IP = os.getenv("ALLOWED_SENDER_IP", "192.168.0.33")

EXPECTED_SHAPES = {
    "coords":       (16, 720, 3),
    "intensity":    (16 * 720, 1),
    "reflectivity": (16, 720, 1),
}

_latest_matrices: dict[str, np.ndarray] | None = None
_old_matrices:    dict[str, np.ndarray] | None = None
_store_lock = Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home() -> Any:
    """Serve the dashboard webpage."""
    return render_template("index.html")


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "allowed_sender_ip": ALLOWED_SENDER_IP})


@app.post("/matrices")
def receive_matrices() -> Any:
    global _latest_matrices, _old_matrices

    client_ip = _get_client_ip()
    if client_ip != ALLOWED_SENDER_IP:
        return jsonify({"error": f"Forbidden sender IP: {client_ip}"}), 403

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected JSON object payload."}), 400

    missing = [name for name in EXPECTED_SHAPES if name not in payload]
    if missing:
        return jsonify({"error": f"Missing required matrices: {missing}"}), 400

    # 1. Parse the incoming JSON into the `matrices` dictionary
    try:
        matrices = {
            name: _parse_matrix(payload[name], expected_shape, name)
            for name, expected_shape in EXPECTED_SHAPES.items()
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # 2. Save it to the global state safely
    with _store_lock:
        _old_matrices = _latest_matrices
        _latest_matrices = matrices

    # 3. Check if the pipeline is busy. If it is, drop the frame.
    if not _processing_lock.acquire(blocking=False):
        print("Pipeline is busy (exceeded processing time). Dropping frame.")
        return jsonify({
            "status": "dropped",
            "reason": "pipeline busy processing previous frame"
        }), 429

    # 4. Define the background task that safely releases the lock when done
    def background_task(curr, old, sett):
        try:
            process_frame(curr, old, sett)
        finally:
            _processing_lock.release()

    # 5. Launch the protected processing thread
    Thread(
        target=background_task,
        args=(matrices, _old_matrices, SETTINGS),
        daemon=True
    ).start()

    return jsonify({
        "status": "received",
        "sender_ip": client_ip,
        "shapes": {name: list(arr.shape) for name, arr in matrices.items()},
    })


@app.get("/matrices")
def get_latest_matrices() -> Any:
    """Fetch the most recently uploaded matrices as JSON lists."""
    with _store_lock:
        if _latest_matrices is None:
            return jsonify({"error": "No matrices received yet."}), 404
        data = {name: arr.tolist() for name, arr in _latest_matrices.items()}
    return jsonify(data)


@app.get("/matrices/shapes")
def get_latest_shapes() -> Any:
    """Fetch only matrix shapes from the most recent valid upload."""
    with _store_lock:
        if _latest_matrices is None:
            return jsonify({"error": "No matrices received yet."}), 404
        shapes = {name: list(arr.shape) for name, arr in _latest_matrices.items()}
    return jsonify({"shapes": shapes})


# ── Webpage API endpoints ───────────────────────────────────────────────────────

@app.get("/api/data")
def get_unified_data() -> Any:
    """Return the combined sectors and pointcloud data in one JSON payload."""
    with cloud_state.data_lock:
        return jsonify(cloud_state.data_state)

# ── Optional upstream polling (disabled by default) ────────────────────────────

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://192.168.0.33:5000/matrices")

def poll_upstream_forever():
    while True:
        try:
            resp = requests.get(UPSTREAM_URL, timeout=2.0)
            resp.raise_for_status()
            payload = resp.json()

            matrices = {
                name: _parse_matrix(payload[name], expected_shape, name)
                for name, expected_shape in EXPECTED_SHAPES.items()
            }

            with _store_lock:
                global _latest_matrices, _old_matrices
                _old_matrices    = _latest_matrices
                _latest_matrices = matrices

            # Thread(
            #     target=process_frame,
            #     args=(matrices, _old_matrices, SETTINGS),
            #     daemon=True
            # ).start()
            process_frame(matrices, _old_matrices, SETTINGS)

        except Exception as exc:
            app.logger.warning(f"Polling failed: {exc}")

        time.sleep(0.05)  # 20 Hz


def main(args=None):
    # Uncomment to enable upstream polling instead of receiving POST requests:
    # Thread(target=poll_upstream_forever, daemon=True).start()

    host  = os.getenv("FLASK_HOST",  "0.0.0.0")
    port  = int(os.getenv("FLASK_PORT",  "5001"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)