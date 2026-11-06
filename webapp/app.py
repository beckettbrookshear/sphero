import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, render_template, request

from webapp.robot_controller import RobotController
from webapp.versioning import ROOT, get_app_version

APP_VERSION = get_app_version(ROOT)

app = Flask(__name__, template_folder="templates")
robot = RobotController(risk_model_path=str(ROOT / ".learned" / "risk_model.json"))


@app.route("/")
def index():
    app_version = get_app_version(ROOT)
    return render_template("index.html", app_version=app_version)


@app.route("/api/connect", methods=["POST"])
def connect_robot():
    payload = request.get_json(silent=True) or {}
    display_text = payload.get("display_text")
    color = payload.get("color", [0, 255, 0])
    speed = int(payload.get("speed", 8))
    repeat = bool(payload.get("repeat", False))

    try:
        result = robot.connect()
        if display_text:
            message = robot.display_message(
                str(display_text).strip() or "Ready",
                color=tuple(color),
                speed=speed,
                repeat=repeat,
            )
            return jsonify({"status": "ok", "message": result, "display_text": message["text"]})
        return jsonify({"status": "ok", "message": result})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/display", methods=["POST"])
def robot_display():
    payload = request.get_json(silent=True) or {}
    display_text = payload.get("display_text", "Ready")
    color = payload.get("color", [0, 255, 0])
    speed = int(payload.get("speed", 8))
    repeat = bool(payload.get("repeat", False))

    try:
        result = robot.display_message(display_text, color=tuple(color), speed=speed, repeat=repeat)
        return jsonify({"status": "ok", "display_text": result["text"]})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/pixel-art", methods=["POST"])
def robot_pixel_art():
    payload = request.get_json(silent=True) or {}
    pixels = payload.get("pixels", [])
    color = payload.get("color", [255, 255, 255])

    try:
        result = robot.display_pixel_art(pixels, color=tuple(color), clear_first=True)
        return jsonify({"status": "ok", "pixels": result["pixels"], "color": result["color"]})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/disconnect", methods=["POST"])
def disconnect_robot():
    try:
        robot.disconnect()
        return jsonify({"status": "ok", "message": "Robot disconnected"})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/command", methods=["POST"])
def robot_command():
    payload = request.get_json(silent=True) or {}
    command = payload.get("command", "stop")
    speed = int(payload.get("speed", 60))
    turn_step = int(payload.get("turn_step", 15))
    turn_rate = payload.get("turn_rate")
    turn_rate = float(turn_rate) if turn_rate is not None else None

    try:
        action = robot.drive(command, speed, turn_step=turn_step, turn_rate=turn_rate)
        return jsonify({"status": "ok", "command": command, "speed": action["speed"], "heading": action["heading"]})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/auto", methods=["POST"])
def robot_auto():
    payload = request.get_json(silent=True) or {}
    steps = max(1, int(payload.get("steps", 4)))
    turn_degrees = int(payload.get("turn_degrees", 90))
    speed = int(payload.get("speed", 60))
    pause = float(payload.get("pause", 0.5))

    try:
        result = robot.auto_patrol(steps=steps, turn_degrees=turn_degrees, forward_speed=speed, pause=pause)
        return jsonify({"status": "ok", "steps": result["steps"], "heading": result["heading"]})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/auto-mode", methods=["POST"])
def robot_auto_mode():
    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled", False))

    try:
        result = robot.set_auto_mode(enabled)
        return jsonify({"status": "ok", "enabled": result["enabled"], "message": result["status"]})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/auto-map")
def robot_auto_map():
    return jsonify(robot.auto_map())


@app.route("/api/emergency-stop", methods=["POST"])
def robot_emergency_stop():
    payload = request.get_json(silent=True) or {}
    delay_seconds = float(payload.get("delay_seconds", 10.0))

    try:
        result = robot.emergency_stop(delay_seconds=delay_seconds)
        return jsonify({"status": "ok", "seconds": result["seconds"], "message": result["status"]})
    except Exception as exc:  # pragma: no cover - runtime only
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/status")
def robot_status():
    connected_name = getattr(robot.toy, "name", None)
    if robot.toy is not None:
        try:
            robot._verify_toy_is_responding()
            detected_robots = []
            help_text = "Connected successfully. If the robot stops responding, disconnect and reconnect."
            connected = True
        except RuntimeError:
            robot._close_toy()
            connected = False
            connected_name = None
            detected_robots = robot.detect_robots()
            if detected_robots:
                help_text = (
                    "A Sphero robot is visible to Bluetooth, but another app may still be holding the connection. "
                    "Close the Sphero app or disconnect the robot there before connecting from this website. "
                    "If it is out of its cradle, it may be asleep or powered down."
                )
            else:
                help_text = (
                    "No Sphero robot detected. The robot may be asleep, off, or disconnected when removed from its cradle. "
                    "Turn on Bluetooth, keep the robot close to the computer, press the power button to wake it up, and "
                    "make sure it is not already connected to the Sphero Edu app."
                )
    else:
        connected = False
        detected_robots = robot.detect_robots()
        if detected_robots:
            help_text = (
                "A Sphero robot is visible to Bluetooth, but another app may still be holding the connection. "
                "Close the Sphero app or disconnect the robot there before connecting from this website. "
                "If it is out of its cradle, it may be asleep or powered down."
            )
        else:
            help_text = (
                "No Sphero robot detected. The robot may be asleep, off, or disconnected when removed from its cradle. "
                "Turn on Bluetooth, keep the robot close to the computer, press the power button to wake it up, and "
                "make sure it is not already connected to the Sphero Edu app."
            )

    version = get_app_version(ROOT)
    return jsonify({
        "connected": connected,
        "connected_robot": connected_name,
        "detected_robots": detected_robots,
        "auto_mode": bool(getattr(robot, "auto_mode", False)),
        "auto_stop_reason": getattr(robot, "auto_stop_reason", None),
        "auto_debug": robot.auto_debug() if getattr(robot, "auto_mode", False) else None,
        "version": version,
        "help": help_text,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=port)
