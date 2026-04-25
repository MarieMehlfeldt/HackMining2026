from flask import Flask, jsonify, request, render_template
from threading import Lock

app = Flask(__name__)

state = {"value": 0}
lock = Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/dirt")
def get_dirt():
    with lock:
        return jsonify(state)


@app.post("/api/dirt")
def set_dirt():
    data = request.get_json(silent=True) or {}

    try:
        value = float(data.get("value"))
    except (TypeError, ValueError):
        return jsonify({"error": "Send JSON like {'value': 42}"}), 400

    value = max(0, min(100, value))

    with lock:
        state["value"] = value

    return jsonify({"ok": True, "value": value})


# Easy browser test route:
# Example: http://localhost:5000/set/75
@app.get("/set/<float:value>")
def set_dirt_from_browser(value):
    value = max(0, min(100, value))

    with lock:
        state["value"] = value

    return f"Dirt-O-Meter set to {value}%. Go back to http://localhost:5000"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)