from flask import Flask, render_template, request, jsonify
from threading import Lock

app = Flask(__name__)

N_SECTORS = 5

data_lock = Lock()
current_sectors = [0.0, 0.0, 0.0, 0.0, 0.0]
current_value = 0.0


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(maximum, value))


def sanitize_sector_values(values):
    if not isinstance(values, list):
        raise ValueError("sectors must be a list")

    if len(values) != N_SECTORS:
        raise ValueError(f"sectors must contain exactly {N_SECTORS} values")

    clean_values = []

    for value in values:
        clean_values.append(clamp(float(value), 0.0, 100.0))

    return clean_values


def calculate_overall_dirtiness(sectors):
    if not sectors:
        return 0.0

    return sum(sectors) / len(sectors)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/sectors", methods=["GET", "POST"])
def sectors():
    global current_sectors
    global current_value

    if request.method == "POST":
        data = request.get_json(silent=True) or {}

        raw_sectors = data.get("sectors")

        if raw_sectors is None:
            return jsonify({
                "ok": False,
                "error": "Missing JSON key: sectors"
            }), 400

        try:
            clean_sectors = sanitize_sector_values(raw_sectors)
        except ValueError as error:
            return jsonify({
                "ok": False,
                "error": str(error)
            }), 400

        with data_lock:
            current_sectors = clean_sectors
            current_value = calculate_overall_dirtiness(clean_sectors)

        return jsonify({
            "ok": True,
            "sectors": current_sectors,
            "value": current_value
        })

    with data_lock:
        return jsonify({
            "ok": True,
            "sectors": current_sectors,
            "value": current_value
        })


@app.route("/api/dirt", methods=["GET", "POST"])
def dirt():
    global current_sectors
    global current_value

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        value = clamp(float(data.get("value", 0.0)), 0.0, 100.0)

        with data_lock:
            current_value = value
            current_sectors = [value] * N_SECTORS

        return jsonify({
            "ok": True,
            "value": current_value,
            "sectors": current_sectors
        })

    with data_lock:
        return jsonify({
            "ok": True,
            "value": current_value,
            "sectors": current_sectors
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)