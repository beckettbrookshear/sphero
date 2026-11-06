import pathlib

import pytest

from webapp.app import app
from webapp import robot_controller
from webapp.robot_controller import RobotController
from webapp.versioning import get_app_version


def test_app_has_no_explore_route():
    route_rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/api/explore" not in route_rules


def test_controller_has_no_exploration_logic():
    controller = RobotController()
    assert not hasattr(controller, "explore_area")


def test_detects_any_sphero_robot_type(monkeypatch):
    calls = []

    class FakeToy:
        def __init__(self, name):
            self.name = name

    def fake_find_toys(timeout=3.0, toy_types=None):
        calls.append(("find_toys", timeout, toy_types))
        return [FakeToy("R2D2"), FakeToy("BB-8")]

    def fake_find_toy(toy_name=None, timeout=3.0, toy_types=None):
        calls.append(("find_toy", toy_name, timeout, toy_types))
        return FakeToy("R2D2")

    monkeypatch.setattr(robot_controller.scanner, "find_toys", fake_find_toys)
    monkeypatch.setattr(robot_controller.scanner, "find_toy", fake_find_toy)

    controller = RobotController()
    names = controller.detect_robots(timeout=2.0)

    assert names == ["R2D2", "BB-8"]
    assert calls[0][0] == "find_toys"
    assert calls[0][2] is None


def test_general_sphero_connection_is_simple(monkeypatch):
    class FakeToy:
        def __init__(self, name):
            self.name = name

    def fake_find_toys(timeout=3.0, toy_types=None):
        return [FakeToy("Bolt"), FakeToy("Indy")]

    monkeypatch.setattr(robot_controller.scanner, "find_toys", fake_find_toys)

    controller = RobotController()
    names = controller.detect_robots(timeout=2.0)

    assert names == ["Bolt", "Indy"]

    html = __import__("pathlib").Path("webapp/templates/index.html").read_text()
    assert "robot-model" not in html
    assert "tight-space-mode" in html


def test_auto_patrol_exists_and_runs(monkeypatch):
    calls = []

    class FakeToy:
        def __init__(self, name):
            self.name = name

        def drive_with_heading(self, speed, heading, flags):
            calls.append(("drive", speed, heading, flags))

        def set_heading(self, heading):
            calls.append(("heading", heading))

    monkeypatch.setattr(robot_controller.time, "sleep", lambda seconds: None)

    controller = RobotController()
    controller.toy = FakeToy("Bolt")

    result = controller.auto_patrol(steps=2, turn_degrees=90, forward_speed=60, pause=0.1)

    assert result["steps"] == 2
    assert len(calls) >= 4
    assert "/api/auto" in {rule.rule for rule in app.url_map.iter_rules()}


def test_auto_pilot_treats_sustained_tilt_as_a_bump_and_keeps_going():
    import time as _time

    controller = RobotController()
    controller.TILT_GRACE_SECONDS = 0.0
    controller.TILT_BLOCKED_SECONDS = 0.1
    controller.IMPACT_GRACE_SECONDS = 0.0
    controller.BACKOFF_SECONDS = 0.05
    controller.BUMP_SETTLE_SECONDS = 0.05
    controller.toy = _SensorFakeToy()

    controller.set_auto_mode(True)
    send = controller.toy.sensor_control.listener
    # Once position arrives the robot is driving; tilt is only meaningful then.
    send({"locator": {"x": 0.0, "y": 0.0}, "attitude": {"pitch": 2.0, "roll": 1.0, "yaw": 0.0}})
    _time.sleep(0.5)
    deadline = _time.monotonic() + 3
    while _time.monotonic() < deadline and controller.bump_count == 0:
        send({"locator": {"x": 0.0, "y": 0.0}, "attitude": {"pitch": 25.0, "roll": 1.0, "yaw": 0.0}})
        _time.sleep(0.05)

    assert controller.bump_count >= 1
    assert controller.auto_mode is True
    controller.set_auto_mode(False)


def test_auto_pilot_ignores_small_tilt_and_tilt_that_was_there_at_start():
    import time as _time

    controller = RobotController()
    controller.TILT_GRACE_SECONDS = 0.0
    controller.TILT_BLOCKED_SECONDS = 0.05
    controller.toy = _SensorFakeToy()

    controller.set_auto_mode(True)
    send = controller.toy.sensor_control.listener
    send({"attitude": {"pitch": 30.0, "roll": 0.0, "yaw": 0.0}})  # started on a slope
    for _ in range(5):
        send({"attitude": {"pitch": 33.0, "roll": 0.0, "yaw": 0.0}})
        _time.sleep(0.05)

    assert controller._pending_bump is None
    assert controller.bump_count == 0
    assert controller.auto_mode is True
    controller.set_auto_mode(False)


def test_auto_pilot_stops_when_boxed_in_by_repeated_bumps():
    controller = RobotController()
    controller.BACKOFF_SECONDS = 0.0
    controller.BUMP_SETTLE_SECONDS = 0.0
    controller.toy = _SensorFakeToy()
    controller.auto_mode = True
    controller._auto_stop_event.clear()
    controller._start_velocity_monitor()

    results = [controller._recover_from_bump("the robot hit an obstacle") for _ in range(controller.BOXED_IN_BUMPS)]

    assert results[-1] is False
    assert all(results[:-1])
    assert controller.auto_mode is False
    assert "boxed in" in controller.auto_stop_reason


def test_html_describes_a_start_and_b_stop_auto_controls():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "A = start auto patrol" in html
    assert "B = stop auto patrol" in html

    response = app.test_client().get('/')
    assert response.status_code == 200
    assert f"v{get_app_version()}" in response.get_data(as_text=True)


def test_html_mentions_y_button_disables_everything():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "Y = disable everything" in html


def test_status_includes_version_number():
    client = app.test_client()
    response = client.get('/api/status')

    assert response.status_code == 200
    assert response.json["version"] == get_app_version()


def test_version_bumps_when_source_files_change(tmp_path):
    source_file = tmp_path / "example.py"
    source_file.write_text("print('hello')\n")

    first = get_app_version(tmp_path)
    assert first == "1.0.0"

    source_file.write_text("print('updated')\n")
    second = get_app_version(tmp_path)

    assert second == "1.0.1"


def test_pixel_art_can_be_rendered_on_matrix(monkeypatch):
    calls = []

    class FakeToy:
        def __init__(self):
            self.name = "Bolt"

        def set_compressed_frame_player_pixel(self, x, y, r, g, b):
            calls.append((x, y, r, g, b))

    controller = RobotController()
    controller.toy = FakeToy()

    result = controller.display_pixel_art([(0, 0), (1, 1)], color=(10, 20, 30))

    assert result["pixels"] == [(0, 0), (1, 1)]
    assert calls == [(0, 0, 10, 20, 30), (1, 1, 10, 20, 30)]
    assert "/api/pixel-art" in {rule.rule for rule in app.url_map.iter_rules()}


def test_pixel_art_supports_a_different_color_per_pixel():
    calls = []

    class FakeToy:
        name = "Bolt"

        def set_compressed_frame_player_pixel(self, x, y, r, g, b):
            if (r, g, b) != (0, 0, 0):
                calls.append((x, y, r, g, b))

    controller = RobotController()
    controller.toy = FakeToy()

    controller.display_pixel_art([(0, 0, 255, 0, 0), (1, 1, 0, 0, 255), (2, 2)], color=(9, 9, 9))

    assert calls == [(0, 0, 255, 0, 0), (1, 1, 0, 0, 255), (2, 2, 9, 9, 9)]


def test_default_pixel_art_uses_florida_flag_pattern():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "Florida" in html or "florida" in html
    assert "defaultFloridaPixels" in html


def test_auto_mode_can_be_reenabled_after_stopping():
    class FakeToy:
        def __init__(self):
            self.events = []
            self.heading = 0

        def add_collision_detected_notify_listener(self, callback):
            self.events.append(("listen", callback))

        def remove_collision_detected_notify_listener(self, callback):
            self.events.append(("unlisten", callback))

        def configure_collision_detection(self, *args, **kwargs):
            self.events.append(("configure", args, kwargs))

        def enable_collision_detected_notify(self, enabled):
            self.events.append(("notify", enabled))

        def drive_with_heading(self, speed, heading, flags):
            self.events.append(("drive", speed, heading, flags))

        def set_heading(self, heading):
            self.heading = heading

    controller = RobotController()
    controller.toy = FakeToy()

    controller.set_auto_mode(True)
    controller.set_auto_mode(False)
    controller.set_auto_mode(True)

    assert controller.auto_mode is True
    assert controller._auto_stop_event.is_set() is False


def test_collision_is_recorded_as_a_bump_and_repeats_are_not_queued():
    class FakeToy:
        def __init__(self):
            self.events = []

        def drive_with_heading(self, speed, heading, flags):
            self.events.append(("drive", speed, heading, flags))

    controller = RobotController()
    controller.toy = FakeToy()
    controller.auto_mode = True

    controller._handle_collision({"x_axis": True, "y_axis": False})
    controller._handle_collision({"x_axis": True, "y_axis": False})

    assert controller._pending_bump == "the robot hit an obstacle"
    assert controller._take_pending_bump() == "the robot hit an obstacle"
    assert controller._take_pending_bump() is None


def test_collision_while_auto_pilot_is_off_is_ignored():
    controller = RobotController()
    controller.toy = object()

    controller._handle_collision({"x_axis": True})

    assert controller._pending_bump is None


def test_custom_message_can_be_sent_to_matrix(monkeypatch):
    calls = []

    class FakeToy:
        def __init__(self, name):
            self.name = name

        def set_compressed_frame_player_text_scrolling(self, text, r, g, b, speed, repeat):
            calls.append((text, r, g, b, speed, repeat))

    controller = RobotController()
    controller.toy = FakeToy("Bolt")

    result = controller.display_message("Hello!", color=(0, 255, 0), speed=8, repeat=False)

    assert result["text"] == "Hello!"
    assert calls == [("Hello!", 0, 255, 0, 8, False)]
    assert "/api/display" in {rule.rule for rule in app.url_map.iter_rules()}


def test_connect_accepts_custom_matrix_color(monkeypatch):
    calls = []

    class FakeController:
        def connect(self):
            calls.append("connect")
            return "Connected to Bolt"

        def display_message(self, text, color=(0, 255, 0), speed=8, repeat=False):
            calls.append((text, tuple(color), speed, repeat))
            return {"text": text, "color": list(color), "speed": speed, "repeat": repeat}

    monkeypatch.setattr("webapp.app.robot", FakeController())

    client = app.test_client()
    response = client.post('/api/connect', json={"display_text": "Hello", "color": [255, 0, 128], "speed": 10, "repeat": True})

    assert response.status_code == 200
    assert calls[0] == "connect"
    assert calls[1] == ("Hello", (255, 0, 128), 10, True)


def test_auto_mode_registers_collision_sensor_and_stop_still_works(monkeypatch):
    class FakeToy:
        def __init__(self):
            self.events = []
            self.heading = 0

        def add_collision_detected_notify_listener(self, callback):
            self.events.append(("listen", callback))

        def remove_collision_detected_notify_listener(self, callback):
            self.events.append(("unlisten", callback))

        def configure_collision_detection(self, *args, **kwargs):
            self.events.append(("configure", args, kwargs))

        def enable_collision_detected_notify(self, enabled):
            self.events.append(("notify", enabled))

        def drive_with_heading(self, speed, heading, flags):
            self.events.append(("drive", speed, heading, flags))

        def set_heading(self, heading):
            self.heading = heading

    toy = FakeToy()
    controller = RobotController()
    controller.toy = toy

    result = controller.set_auto_mode(True)
    assert result["enabled"] is True
    assert any(evt[0] == "configure" for evt in toy.events)

    controller._handle_collision({"x_axis": True, "y_axis": False})
    assert controller.auto_mode is True

    result_off = controller.set_auto_mode(False)
    assert result_off["enabled"] is False
    assert controller.auto_mode is False
    assert controller._auto_stop_event.is_set() is True
    assert "/api/auto-mode" in {rule.rule for rule in app.url_map.iter_rules()}


def test_auto_mode_can_be_disabled_even_if_robot_is_not_responding(monkeypatch):
    controller = RobotController()
    controller.toy = object()
    monkeypatch.setattr(controller, "_verify_toy_is_responding", lambda: (_ for _ in ()).throw(RuntimeError("not responding")))

    result = controller.set_auto_mode(False)

    assert result["enabled"] is False
    assert controller.auto_mode is False
    assert controller._auto_stop_event.is_set() is True


def test_emergency_stop_restores_default_ready_state(monkeypatch):
    class FakeToy:
        def __init__(self):
            self.events = []
            self.heading = 0

        def drive_with_heading(self, speed, heading, flags):
            self.events.append(("drive", speed, heading, flags))

        def set_heading(self, heading):
            self.heading = heading

        def set_compressed_frame_player_text_scrolling(self, text, r, g, b, speed, repeat):
            self.events.append(("display", text, r, g, b, speed, repeat))

    controller = RobotController()
    controller.toy = FakeToy()
    controller.auto_mode = True

    class FakeTimer:
        def __init__(self, delay, func):
            self.delay = delay
            self.func = func
            self.daemon = True

        def start(self):
            self.func()

    monkeypatch.setattr("webapp.robot_controller.time.sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr("webapp.robot_controller.threading.Timer", FakeTimer)
    controller.emergency_stop(delay_seconds=0.01)

    assert controller.auto_mode is False
    assert controller.heading == 0
    assert controller.toy.events[-1][0] == "display"
    assert controller.toy.events[-1][1] == "Ready"


def test_stop_command_turns_off_auto_mode():
    class FakeToy:
        def __init__(self):
            self.events = []
            self.heading = 0

        def drive_with_heading(self, speed, heading, flags):
            self.events.append(("drive", speed, heading, flags))

        def set_heading(self, heading):
            self.heading = heading

    controller = RobotController()
    controller.toy = FakeToy()
    controller.auto_mode = True
    controller._auto_stop_event.clear()

    controller.drive("stop")

    assert controller.auto_mode is False
    assert controller._auto_stop_event.is_set() is True
    assert controller.toy.events[-1][1] == 0


def test_manual_drive_halts_when_commands_stop_arriving():
    import time as _time

    class FakeToy:
        def __init__(self):
            self.events = []

        def drive_with_heading(self, speed, heading, flags):
            self.events.append(speed)

    controller = RobotController()
    controller.DRIVE_WATCHDOG_SECONDS = 0.05
    controller.toy = FakeToy()

    controller.drive("forward", 60)
    assert controller.toy.events[-1] == 60

    _time.sleep(0.25)

    assert controller.toy.events[-1] == 0


def test_manual_stop_does_not_touch_collision_settings_when_auto_is_off():
    class FakeToy:
        def __init__(self):
            self.events = []

        def drive_with_heading(self, speed, heading, flags):
            self.events.append(("drive", speed))

        def enable_collision_detected_notify(self, enabled):
            self.events.append(("notify", enabled))

    controller = RobotController()
    controller.toy = FakeToy()

    controller.drive("stop")

    assert controller.toy.events == [("drive", 0)]


def test_gamepad_stops_immediately_when_stick_is_released():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "commandInFlight" in html
    assert "lastGamepadCommand && now - lastGamepadTimestamp > 150" not in html


def test_auto_pilot_treats_robot_that_is_driven_but_not_moving_as_a_bump():
    import time as _time

    class FakeSensorControl:
        def __init__(self):
            self.listener = None

        def add_sensor_data_listener(self, listener):
            self.listener = listener

        def remove_sensor_data_listener(self, listener):
            self.listener = None

        def set_count(self, count):
            pass

        def set_interval(self, interval):
            pass

        def enable(self, *sensors):
            pass

        def disable(self, *sensors):
            pass

    class FakeToy:
        def __init__(self):
            self.sensor_control = FakeSensorControl()
            self.drives = []

        def drive_with_heading(self, speed, heading, flags):
            self.drives.append(speed)

        def configure_collision_detection(self, *args):
            pass

        def add_collision_detected_notify_listener(self, callback):
            pass

        def remove_collision_detected_notify_listener(self, callback):
            pass

    controller = RobotController()
    controller.STALL_GRACE_SECONDS = 0.0
    controller.STALL_SECONDS = 0.3
    controller.BACKOFF_SECONDS = 0.05
    controller.BUMP_SETTLE_SECONDS = 0.05
    toy = FakeToy()
    controller.toy = toy

    controller.set_auto_mode(True)
    deadline = _time.monotonic() + 3
    while _time.monotonic() < deadline and controller.bump_count == 0:
        # Robot reports it is not moving even though it is being driven.
        toy.sensor_control.listener({"velocity": {"x": 0.0, "y": 0.0}, "locator": {"x": 5.0, "y": 5.0}})
        _time.sleep(0.05)

    assert controller.bump_count >= 1
    assert controller._planner.obstacles
    assert controller.auto_mode is True
    controller.set_auto_mode(False)
    assert toy.drives[-1] == 0


def test_auto_pilot_keeps_going_while_robot_is_moving():
    import time as _time

    class FakeSensorControl:
        listener = None

        def add_sensor_data_listener(self, listener):
            self.listener = listener

        def remove_sensor_data_listener(self, listener):
            pass

        def set_count(self, count):
            pass

        def set_interval(self, interval):
            pass

        def enable(self, *sensors):
            pass

        def disable(self, *sensors):
            pass

    class FakeToy:
        def __init__(self):
            self.sensor_control = FakeSensorControl()

        def drive_with_heading(self, speed, heading, flags):
            pass

    controller = RobotController()
    controller.STALL_GRACE_SECONDS = 0.0
    controller.STALL_SECONDS = 0.3
    toy = FakeToy()
    controller.toy = toy

    controller.set_auto_mode(True)
    end = _time.monotonic() + 0.8
    while _time.monotonic() < end:
        toy.sensor_control.listener({"velocity": {"x": 30.0, "y": 0.0}})
        _time.sleep(0.05)

    assert controller.auto_mode is True
    controller.set_auto_mode(False)


class _SensorFakeToy:
    class _Sensors:
        def __init__(self):
            self.listener = None
            self.fail_enable = False

        def add_sensor_data_listener(self, listener):
            self.listener = listener

        def remove_sensor_data_listener(self, listener):
            self.listener = None

        def set_count(self, count):
            pass

        def set_interval(self, interval):
            pass

        def enable(self, *sensors):
            if self.fail_enable:
                raise RuntimeError("no sensors")

        def disable(self, *sensors):
            pass

    def __init__(self):
        self.sensor_control = self._Sensors()
        self.drives = []

    def drive_with_heading(self, speed, heading, flags):
        self.drives.append(speed)


def test_auto_pilot_treats_sudden_acceleration_spike_as_a_bump_without_collision_event():
    import time as _time

    controller = RobotController()
    controller.IMPACT_GRACE_SECONDS = 0.0
    controller.BACKOFF_SECONDS = 0.05
    controller.BUMP_SETTLE_SECONDS = 0.05
    controller.toy = _SensorFakeToy()

    controller.set_auto_mode(True)
    listener = controller.toy.sensor_control.listener
    deadline = _time.monotonic() + 3
    while _time.monotonic() < deadline and controller.bump_count == 0:
        # Position arrives first so the robot is actually driving, then the spike.
        listener({"locator": {"x": 0.0, "y": 0.0}})
        _time.sleep(0.4)
        listener({"accelerometer": {"x": 0.0, "y": 0.0, "z": 1.0}, "locator": {"x": 0.0, "y": 0.0}})
        listener({"accelerometer": {"x": 0.1, "y": 2.5, "z": 1.0}, "locator": {"x": 0.0, "y": 0.0}})
        _time.sleep(0.3)

    assert controller.bump_count >= 1
    assert controller.auto_mode is True
    controller.set_auto_mode(False)
    assert controller.toy.drives[-1] == 0


def test_auto_pilot_stops_when_sensor_data_goes_silent():
    import time as _time

    controller = RobotController()
    controller.SENSOR_SILENCE_SECONDS = 0.3
    controller.toy = _SensorFakeToy()

    controller.set_auto_mode(True)
    deadline = _time.monotonic() + 3
    while _time.monotonic() < deadline and controller.auto_mode:
        _time.sleep(0.05)

    assert controller.auto_mode is False
    assert controller.toy.drives[-1] == 0
    assert "sensor data" in controller.auto_stop_reason


def test_failed_auto_pilot_start_does_not_leave_auto_mode_enabled():
    controller = RobotController()
    controller.toy = _SensorFakeToy()
    controller.toy.sensor_control.fail_enable = True

    with pytest.raises(RuntimeError, match="motion sensors"):
        controller.set_auto_mode(True)

    assert controller.auto_mode is False
    assert controller._auto_thread is None


def test_stop_during_auto_pilot_start_wins():
    controller = RobotController()
    toy = _SensorFakeToy()
    controller.toy = toy
    original_enable = toy.sensor_control.enable

    def enable_then_stop_arrives(*sensors):
        original_enable(*sensors)
        controller.drive("stop")

    toy.sensor_control.enable = enable_then_stop_arrives

    result = controller.set_auto_mode(True)

    assert result["enabled"] is False
    assert controller.auto_mode is False
    assert controller._auto_thread is None


def test_starting_auto_pilot_twice_does_not_restart_it():
    controller = RobotController()
    controller.toy = _SensorFakeToy()

    controller.set_auto_mode(True)
    first_thread = controller._auto_thread
    result = controller.set_auto_mode(True)

    assert result["enabled"] is True
    assert controller._auto_thread is first_thread
    controller.set_auto_mode(False)


def test_html_ignores_status_polls_while_auto_request_is_in_flight():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "autoRequestInFlight" in html


def test_status_reports_why_auto_pilot_stopped():
    client = app.test_client()
    response = client.get('/api/status')
    assert "auto_stop_reason" in response.json


def test_html_polls_status_so_button_resets_when_auto_pilot_ends():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "pollAutoPilotStatus" in html


def test_html_dpad_drives_robot():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "D-pad = drive" in html
    assert "dpadPressed(12)" in html
    assert "FINE_CONTROL_BUTTONS" not in html


def test_html_controller_requires_a_b_b_unlock():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "UNLOCK_SEQUENCE = ['A', 'B', 'B']" in html
    assert "controllerUnlocked" in html
    assert "press A, B, B to unlock" in html
    # Stop actions must not be gated by the lock.
    assert "// Stop is never gated by the lock." in html


def test_start_auto_pilot_button_is_visible():
    html = __import__("pathlib").Path("webapp/templates/index.html").read_text()
    assert "Start Auto Pilot" in html
    assert "Stop Auto Pilot" in html


def test_status_skips_scanning_while_connected(monkeypatch):
    connected = RobotController()
    connected.toy = type("FakeToy", (), {"name": "Bolt"})()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Bluetooth scan should not be triggered while connected")

    monkeypatch.setattr("webapp.app.robot", connected)
    monkeypatch.setattr(connected, "detect_robots", fail_if_called)

    client = app.test_client()
    response = client.get('/api/status')

    assert response.status_code == 200
    assert response.json["connected"] is True
    assert response.json["connected_robot"] == "Bolt"


def test_stale_connected_robot_is_rejected_before_commands_are_sent():
    class FakeToy:
        name = "Bolt"

        def ping(self):
            raise RuntimeError("device is not responding")

    controller = RobotController()
    controller.toy = FakeToy()

    with pytest.raises(RuntimeError, match="not responding"):
        controller.display_message("Ready")


def test_connect_retries_when_current_toy_is_stale(monkeypatch):
    class StaleToy:
        name = "Bolt"

        def ping(self):
            raise RuntimeError("device is not responding")

    class FreshToy:
        name = "Bolt"

        def __enter__(self):
            return self

        def get_api_protocol_version(self):
            return True

    def fake_find_toy(*args, **kwargs):
        return FreshToy()

    def fake_find_toys(*args, **kwargs):
        return [FreshToy()]

    controller = RobotController()
    controller.toy = StaleToy()

    monkeypatch.setattr(robot_controller.scanner, "find_toy", fake_find_toy)
    monkeypatch.setattr(robot_controller.scanner, "find_toys", fake_find_toys)

    result = controller.connect()

    assert result == "Connected to Bolt"
    assert controller.toy is not None
    assert controller.toy.name == "Bolt"


def test_held_turn_rotates_at_the_requested_degrees_per_second(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(robot_controller.time, "monotonic", lambda: clock["now"])

    class FakeToy:
        def __init__(self):
            self.calls = []

        def set_heading(self, heading):
            pass

        def drive_with_heading(self, speed, heading, flags):
            self.calls.append((speed, heading))

    controller = RobotController()
    controller.toy = FakeToy()
    monkeypatch.setattr(controller, "_verify_toy_is_responding", lambda: None)

    # Hold "left" at 45 deg/s, one request every 0.2 s, for 2 seconds.
    for _ in range(11):
        controller.drive("left", 40, turn_rate=45)
        clock["now"] += 0.2
    # First request counts 0.1 s, the next ten 0.2 s each: 2.1 s * 45 = 94.5 deg.
    assert controller.heading in (94, 95)

    # Releasing and holding the other way resets timing and turns the other way.
    controller.drive("stop")
    start = controller.heading
    clock["now"] += 5
    for _ in range(10):
        controller.drive("right", 40, turn_rate=10)
        clock["now"] += 0.25
    # 0.1 s + nine 0.25 s steps = 2.35 s * 10 = 23.5 deg to the right.
    assert (start - controller.heading) % 360 in (23, 24)


def test_dpad_is_slow_and_stick_turns_at_45_degrees_per_second():
    html = pathlib.Path("webapp/templates/index.html").read_text()
    assert "STICK_TURN_RATE = 45" in html
    assert "DPAD_TURN_RATE" in html and "DPAD_DRIVE_SPEED" in html
    assert "turn_rate" in html


def _controller_with_pending_bump(source, velocity_age, last_moving_age, pending_age=1.0):
    import time as _time

    controller = RobotController()
    controller.toy = object()
    now = _time.monotonic()
    controller._sensor_feedback = True
    controller._last_velocity_at = None if velocity_age is None else now - velocity_age
    controller._last_moving_at = now - last_moving_age
    controller._pending_bump = "the robot hit an obstacle"
    controller._pending_bump_source = source
    controller._pending_bump_at = now - pending_age
    return controller


def test_bump_is_rejected_when_the_robot_kept_moving_at_speed():
    controller = _controller_with_pending_bump("jerk", velocity_age=0.05, last_moving_age=0.05)

    assert controller._take_confirmed_bump() is None
    assert controller.rejected_bumps == 1
    assert controller._pending_bump is None


def test_bump_is_accepted_when_the_robot_really_stopped():
    controller = _controller_with_pending_bump("tilt", velocity_age=0.05, last_moving_age=0.6)

    assert controller._take_confirmed_bump() == ("the robot hit an obstacle", "tilt")
    assert controller.rejected_bumps == 0


def test_bump_waits_for_the_confirmation_window_before_deciding():
    controller = _controller_with_pending_bump("jerk", 0.05, 0.05, pending_age=0.05)

    assert controller._take_confirmed_bump() is None
    assert controller._pending_bump is not None
    assert controller.rejected_bumps == 0


def test_bump_is_accepted_at_once_when_speed_cannot_be_verified():
    controller = _controller_with_pending_bump("collision", velocity_age=None, last_moving_age=0.0, pending_age=0.0)

    assert controller._take_confirmed_bump() == ("the robot hit an obstacle", "collision")


def test_bump_sources_are_counted_and_reported():
    controller = RobotController()

    class FakeToy:
        def drive_with_heading(self, *args):
            pass

    controller.toy = FakeToy()
    controller.BACKOFF_SECONDS = 0.01
    controller.BUMP_SETTLE_SECONDS = 0.01
    controller.auto_mode = True
    controller._recover_from_bump("the robot hit an obstacle", "jerk")
    controller._recover_from_bump("the robot was tilted", "tilt")

    report = controller.auto_map()
    assert report["bump_sources"] == {"jerk": 1, "tilt": 1}
    assert report["rejected_bumps"] == 0


def test_position_is_pulled_back_after_a_bump_so_wheel_slip_is_not_counted_as_travel():
    controller = RobotController()
    controller._position_origin = (100.0, 100.0)
    controller._last_moving_position = (40.0, 0.0)
    controller._position = (55.0, 0.0)  # wheels kept turning against the object

    controller._undo_slip_after_bump()

    assert controller._position == (40.0, 0.0)
    assert controller.position_corrections == 1
    # The next raw reading is interpreted relative to the corrected origin.
    controller._on_sensor_data({"locator": {"x": 155.0, "y": 100.0}})
    assert controller._position == (40.0, 0.0)


def test_implausible_position_jumps_are_not_used_to_correct_after_a_bump():
    controller = RobotController()
    controller._position_origin = (0.0, 0.0)
    controller._last_moving_position = (0.0, 0.0)
    controller._position = (200.0, 0.0)

    controller._undo_slip_after_bump()

    assert controller._position == (200.0, 0.0)
    assert controller.position_corrections == 0
