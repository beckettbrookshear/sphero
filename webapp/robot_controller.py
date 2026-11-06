import inspect
import math
import threading
import time
from collections import deque
from typing import Dict, Optional

from spherov2 import scanner
from spherov2.commands.drive import DriveFlags
from spherov2.commands.sensor import CollisionDetectionMethods

from webapp.patrol_planner import PatrolPlanner
from webapp.risk_model import RiskModel


_DIRECTION_HEADINGS = {
    "forward": {"speed": 60, "heading": 0},
    "backward": {"speed": 60, "heading": 180},
    "left": {"speed": 60, "heading": 90},
    "right": {"speed": 60, "heading": 270},
    "stop": {"speed": 0, "heading": 0},
}


def command_to_action(command: str, speed: int = 60) -> Dict[str, int]:
    normalized = command.lower().strip()

    if normalized not in _DIRECTION_HEADINGS:
        raise ValueError(f"Unsupported command: {command}")

    action = _DIRECTION_HEADINGS[normalized].copy()
    if normalized != "stop":
        action["speed"] = max(0, min(255, speed))
    return action


class RobotController:
    DRIVE_WATCHDOG_SECONDS = 0.4
    # Auto pilot stall detection: if the robot is being driven forward but its
    # measured speed stays below STALL_SPEED_CM_S for STALL_SECONDS, it is blocked.
    STALL_SPEED_CM_S = 6.0
    STALL_GRACE_SECONDS = 1.0
    STALL_SECONDS = 1.0
    VELOCITY_STALE_SECONDS = 1.0
    # Independent impact detector: a sudden jump in measured acceleration between
    # two consecutive samples (in g) is treated as a collision, even if the robot
    # never sends its own collision notification. Not yet tuned on real hardware.
    IMPACT_JERK_G = 1.5
    IMPACT_GRACE_SECONDS = 0.5
    # Fail-safe: if the sensor stream goes quiet the robot cannot see obstacles,
    # so auto pilot stops instead of driving blind.
    SENSOR_SILENCE_SECONDS = 2.0
    # Gentle enough that an impact is soft, fast enough to cross the floor and
    # clear the stall detector's speed threshold with a good margin.
    AUTO_SPEED = 50
    AUTO_TICK_SECONDS = 0.1
    # Blocked-by-tilt detection: when the shell is stopped by an obstacle, the
    # drive carriage inside it keeps pushing and climbs up the inside of the shell,
    # so the robot's pitch/roll tilts away from where it was when auto pilot began.
    # Not yet tuned on real hardware.
    TILT_BLOCKED_DEGREES = 20.0
    TILT_BLOCKED_SECONDS = 0.4
    TILT_GRACE_SECONDS = 0.5
    # Patrol area: a square grid centred on the start position, measured with the
    # robot's own position sensor (wheel encoders + gyro, so it drifts a little).
    ARENA_HALF_CELLS = 5
    CELL_CM = 30.0
    # Beyond the arena edge by this much the robot is lost: stop.
    FENCE_MARGIN_CM = 40.0
    POSITION_WAIT_SECONDS = 3.0
    POSITION_STALE_SECONDS = 1.5
    # Position axes may not line up with the heading the robot is given, so the
    # offset is measured from how it actually moves.
    CALIBRATION_LEG_CM = 20.0
    # Before patrolling, drive one short leg at heading 0 and one at heading 90 and
    # watch where the robot really goes. That tells which way the position axes run
    # relative to the headings (rotated and/or mirrored).
    CALIBRATION_MOVE_CM = 25.0
    CALIBRATION_TIMEOUT_SECONDS = 6.0
    TURN_IN_PLACE_DEGREES = 60.0
    TURN_PAUSE_SECONDS = 0.5
    # Any kind of bump (collision message, acceleration spike, tilt, stall) makes
    # the robot back off, remember the blocked cell and plan around it.
    BUMP_AHEAD_CM = 12.0
    # Extra route cost for a cell touching a known object (a normal cell costs 1).
    NEAR_OBJECT_COST = 4.0
    # Speed within two cells of a known object (still above the stall threshold).
    NEAR_OBJECT_SPEED = 32
    # Most the position estimate is pulled back after a bump (wheel slip against the object).
    MAX_SLIP_CORRECTION_CM = 40.0
    # A bump signal (collision message, acceleration spike, tilt) only counts if the
    # robot really stopped: after BUMP_CONFIRM_SECONDS its measured speed must have
    # been below the stall threshold for at least BUMP_CONFIRM_STILL_SECONDS.
    # Otherwise it was vibration or noise and is ignored.
    BUMP_CONFIRM_SECONDS = 0.35
    BUMP_CONFIRM_STILL_SECONDS = 0.2
    BACKOFF_SECONDS = 0.5
    BACKOFF_SPEED = 30
    BUMP_SETTLE_SECONDS = 0.5
    # This many bumps within the window means it is boxed in: stop.
    BOXED_IN_BUMPS = 5
    BOXED_IN_WINDOW_SECONDS = 10.0
    # Map recording: the path is kept as points at least this far apart.
    MAP_TRAIL_SPACING_CM = 2.0
    MAP_TRAIL_MAX_POINTS = 8000
    MAP_MAX_POINTS_SENT = 3000
    # Manual turning is time based: a held turn command rotates at a steady rate
    # (degrees per second) no matter how often the browser sends it.
    MAX_TURN_RATE_DEG_PER_SEC = 180.0
    TURN_NOMINAL_TICK_SECONDS = 0.1
    TURN_MAX_TICK_SECONDS = 0.3

    def __init__(self, risk_model_path: Optional[str] = None) -> None:
        self._risk_model = RiskModel(risk_model_path)
        self._risk_seen: set = set()
        self.toy: Optional[object] = None
        self.heading = 0
        self.auto_mode = False
        self.auto_stop_reason: Optional[str] = None
        self._velocity_listener = None
        self._last_velocity_at: Optional[float] = None
        self._last_sensor_at: Optional[float] = None
        self._last_accel: Optional[tuple] = None
        self._sensor_feedback = False
        self._tilt_baseline: Optional[tuple] = None
        self._tilt_since: Optional[float] = None
        self._settle_until = 0.0
        self._turn_lock = threading.Lock()
        self._last_turn_at: Optional[float] = None
        self._turn_remainder = 0.0
        self._bump_lock = threading.Lock()
        self._emergency_timer: Optional[threading.Timer] = None
        self._trail: deque = deque(maxlen=self.MAP_TRAIL_MAX_POINTS)
        self._bump_points: list = []
        self._pending_bump: Optional[str] = None
        self._pending_bump_source = "collision"
        self._pending_bump_at = 0.0
        self.bump_sources: Dict[str, int] = {}
        self.rejected_bumps = 0
        self.position_corrections = 0
        self.repeat_bumps = 0
        self._recent_bumps: list = []
        self.bump_count = 0
        self.avoided_count = 0
        self._near_obstacles: set = set()
        self._planner = PatrolPlanner(self.ARENA_HALF_CELLS, self.CELL_CM)
        self._prev_cell: Optional[tuple] = None
        self._last_moving_position: Optional[tuple] = None
        self._travel_step: Optional[tuple] = None
        self._position: Optional[tuple] = None
        self._position_origin: Optional[tuple] = None
        self._last_position_at: Optional[float] = None
        self._goal: Optional[tuple] = None
        self._heading_offset = 0.0
        self._heading_sign = 1
        self._cal_stage: Optional[int] = None
        self._cal_start: Optional[tuple] = None
        self._cal_started_at = 0.0
        self._cal_aligned = False
        self._cal_first_bearing = 0.0
        self._leg_start: Optional[tuple] = None
        self._leg_heading = 0.0
        self._last_command_heading: Optional[float] = None
        self._last_moving_at = 0.0
        self._auto_started_at = 0.0
        self._drive_watchdog: Optional[threading.Timer] = None
        self._watchdog_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._collision_lock = threading.Lock()
        self._auto_thread: Optional[threading.Thread] = None
        self._auto_state_lock = threading.Lock()
        self._auto_generation = 0
        self._auto_stop_event = threading.Event()
        self._collision_listener = None
        self._last_collision_at = 0.0

    def _cancel_pending_auto_start(self) -> None:
        # Any stop request invalidates an auto-pilot start that is still being set up.
        with self._auto_state_lock:
            self._auto_generation += 1

    def _close_toy(self) -> None:
        self._cancel_pending_auto_start()
        toy = self.toy
        self.toy = None
        self.heading = 0
        self.auto_mode = False
        self._auto_stop_event.set()
        if toy is None:
            return
        try:
            close_method = getattr(toy, "close", None)
            if callable(close_method):
                close_method()
            else:
                disconnect_method = getattr(toy, "disconnect", None)
                if callable(disconnect_method):
                    disconnect_method()
                else:
                    toy.__exit__(None, None, None)
        except Exception:
            try:
                toy.__exit__(None, None, None)
            except Exception:
                pass

    def detect_robots(self, timeout: float = 3.0) -> list[str]:
        try:
            toys = scanner.find_toys(timeout=timeout)
        except Exception:
            return []

        names: list[str] = []
        for toy in toys or []:
            name = getattr(toy, "name", None) or getattr(toy, "device_name", None) or getattr(toy, "toy_name", None)
            if name and name not in names:
                names.append(name)
        return names

    def _find_toy(self, preferred_name: Optional[str] = None, timeout: float = 3.0):
        candidates = []
        if preferred_name:
            try:
                candidates.append(scanner.find_toy(toy_name=preferred_name, timeout=timeout))
            except Exception:
                pass
        try:
            candidates.extend(scanner.find_toys(timeout=timeout))
        except Exception:
            pass

        if not candidates:
            return None

        seen = set()
        for toy in candidates:
            name = getattr(toy, 'name', None)
            if name and name not in seen:
                seen.add(name)
                return toy
        return candidates[0]

    def connect(self, preferred_name: Optional[str] = None) -> str:
        with self._connect_lock:
            if self.toy is not None:
                try:
                    self._verify_toy_is_responding()
                    return "already connected"
                except RuntimeError:
                    self._close_toy()

            last_error: Optional[Exception] = None
            for attempt in range(3):
                try:
                    toy = self._find_toy(preferred_name=preferred_name, timeout=3.0)
                    if toy is None:
                        raise RuntimeError('No Sphero robot discovered')

                    self.toy = None
                    enter_method = getattr(toy, "__enter__", None)
                    if callable(enter_method):
                        enter_method()
                    self.toy = toy
                    self.heading = 0
                    self._verify_toy_is_responding()
                    try:
                        self.display_message("Ready", color=(0, 255, 0), speed=8, repeat=False)
                    except Exception:
                        pass
                    return f"Connected to {self.toy.name}"
                except Exception as exc:
                    last_error = exc
                    self.toy = None
                    if attempt < 2:
                        continue

            raise RuntimeError(
                "No Sphero robot found. The robot may be asleep, powered off, or disconnected when removed from its cradle. "
                "Turn on Bluetooth, keep the robot close to the computer, press the power button to wake it up, and close the Sphero app if it is already connected there. "
                f"({last_error})"
            ) from last_error

    def disconnect(self) -> None:
        with self._connect_lock:
            if self.auto_mode:
                self.set_auto_mode(False)
            if self.toy is None:
                return
            try:
                self._close_toy()
            finally:
                self.toy = None
                self.heading = 0

    def _verify_toy_is_responding(self) -> None:
        if self.toy is None:
            raise RuntimeError("Robot is not connected yet. Click Connect Robot first.")

        health_checks = [
            ("ping", lambda method: method(b"\x00")),
            ("get_api_protocol_version", lambda method: method()),
            ("get_bluetooth_name", lambda method: method()),
        ]

        for method_name, probe in health_checks:
            method = getattr(self.toy, method_name, None)
            if not callable(method):
                continue

            try:
                if method_name == "ping":
                    probe(method)
                else:
                    signature = inspect.signature(method)
                    required_params = [
                        parameter
                        for parameter in signature.parameters.values()
                        if parameter.default is inspect.Parameter.empty
                        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    ]
                    if required_params:
                        continue
                    probe(method)
                return
            except Exception as exc:
                self._close_toy()
                raise RuntimeError(
                    "The robot is connected, but it is not responding. Disconnect and reconnect it, then try again."
                ) from exc

    def turn_by_degrees(self, degrees: int) -> int:
        self._verify_toy_is_responding()

        self.heading = (self.heading + int(degrees)) % 360
        if hasattr(self.toy, "set_heading"):
            self.toy.set_heading(self.heading)
        self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
        return self.heading

    def _cancel_drive_watchdog(self) -> None:
        with self._watchdog_lock:
            timer = self._drive_watchdog
            self._drive_watchdog = None
        if timer is not None:
            timer.cancel()

    def _arm_drive_watchdog(self) -> None:
        # Manual driving must be refreshed continuously; if commands stop
        # arriving (stick released, request lost, browser closed) the robot halts.
        self._cancel_drive_watchdog()
        timer = threading.Timer(self.DRIVE_WATCHDOG_SECONDS, self._watchdog_stop)
        timer.daemon = True
        with self._watchdog_lock:
            self._drive_watchdog = timer
        timer.start()

    def _watchdog_stop(self) -> None:
        with self._watchdog_lock:
            self._drive_watchdog = None
        if self.auto_mode or self.toy is None:
            return
        try:
            self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
        except Exception:
            pass

    def _turn_delta_for_rate(self, rate: float) -> int:
        """Whole degrees to turn now for a turn held at `rate` degrees per second."""
        now = time.monotonic()
        last = self._last_turn_at
        self._last_turn_at = now
        if last is None or now - last > self.TURN_MAX_TICK_SECONDS:
            elapsed = self.TURN_NOMINAL_TICK_SECONDS
        else:
            elapsed = now - last
        rate = max(0.0, min(self.MAX_TURN_RATE_DEG_PER_SEC, float(rate)))
        self._turn_remainder += rate * elapsed
        whole = int(self._turn_remainder)
        self._turn_remainder -= whole
        return whole

    def _reset_turn_rate_state(self) -> None:
        self._last_turn_at = None
        self._turn_remainder = 0.0

    def drive(
        self,
        command: str,
        speed: int = 60,
        turn_step: int = 15,
        turn_rate: Optional[float] = None,
    ) -> Dict[str, int]:
        if command == "stop":
            self._reset_turn_rate_state()
            self._cancel_drive_watchdog()
            self._cancel_pending_auto_start()
            if self.toy is not None:
                try:
                    self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
                except Exception:
                    pass
            if self.auto_mode:
                self.set_auto_mode(False)
            self._auto_stop_event.set()
            return {"speed": 0, "heading": self.heading}

        self._verify_toy_is_responding()

        action = command_to_action(command, speed)
        step = max(5, min(90, int(turn_step)))

        if command not in {"left", "right", "turn_left", "turn_right"}:
            self._reset_turn_rate_state()

        try:
            if command == "forward":
                if hasattr(self.toy, "set_heading"):
                    self.toy.set_heading(self.heading)
                self.toy.drive_with_heading(action["speed"], self.heading, DriveFlags.FORWARD)
                self._arm_drive_watchdog()
                return {"speed": action["speed"], "heading": self.heading}

            if command == "backward":
                heading = (self.heading + 180) % 360
                if hasattr(self.toy, "set_heading"):
                    self.toy.set_heading(heading)
                self.toy.drive_with_heading(action["speed"], heading, DriveFlags.FORWARD)
                self._arm_drive_watchdog()
                return {"speed": action["speed"], "heading": heading}

            if command in {"left", "right", "turn_left", "turn_right"}:
                self._cancel_drive_watchdog()
                if turn_rate is not None:
                    amount = self._turn_delta_for_rate(turn_rate)
                    if amount == 0:
                        return {"speed": 0, "heading": self.heading}
                else:
                    amount = step
                delta = amount if command in {"left", "turn_left"} else -amount
                heading = self.turn_by_degrees(delta)
                return {"speed": 0, "heading": heading}

            raise ValueError(f"Unsupported command: {command}")
        except Exception:
            self._close_toy()
            raise RuntimeError("Robot connection was lost. Please reconnect and try again.")

    def auto_patrol(self, steps: int = 4, turn_degrees: int = 90, forward_speed: int = 60, pause: float = 0.5) -> Dict[str, object]:
        self._verify_toy_is_responding()

        repeats = max(1, int(steps))
        turn_delta = int(turn_degrees)
        speed = max(0, min(255, int(forward_speed)))
        wait_time = max(0.0, float(pause))

        for _ in range(repeats):
            self.toy.drive_with_heading(speed, self.heading, DriveFlags.FORWARD)
            time.sleep(wait_time)
            self.turn_by_degrees(turn_delta)
            time.sleep(wait_time / 2)

        self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
        return {"steps": repeats, "heading": self.heading}

    def _handle_collision(self, event):
        self._register_bump("the robot hit an obstacle")

    def _register_bump(self, reason: str, source: str = "collision") -> None:
        # Called from sensor threads. Only records the bump; the auto-pilot thread
        # does the backing off and re-planning.
        if not self.auto_mode or self.toy is None:
            return
        now = time.monotonic()
        if now - self._auto_started_at < self.IMPACT_GRACE_SECONDS or now < self._settle_until:
            return
        with self._bump_lock:
            if self._pending_bump is None:
                self._pending_bump = reason
                self._pending_bump_source = source
                self._pending_bump_at = now

    def _take_confirmed_bump(self) -> Optional[tuple]:
        """A pending bump as (reason, source) once the robot's speed confirms it; None otherwise."""
        with self._bump_lock:
            reason = self._pending_bump
            source = self._pending_bump_source
            since = self._pending_bump_at
        if reason is None:
            return None
        now = time.monotonic()
        can_verify = (
            self._sensor_feedback
            and self._last_velocity_at is not None
            and now - self._last_velocity_at <= self.VELOCITY_STALE_SECONDS
        )
        if can_verify and now - since < self.BUMP_CONFIRM_SECONDS:
            return None  # keep driving and look again
        with self._bump_lock:
            self._pending_bump = None
        if can_verify and now - self._last_moving_at < self.BUMP_CONFIRM_STILL_SECONDS:
            self.rejected_bumps += 1  # it kept rolling at normal speed, so nothing was in the way
            return None
        return (reason, source)

    def _take_pending_bump(self) -> Optional[str]:
        with self._bump_lock:
            reason = self._pending_bump
            self._pending_bump = None
        return reason

    def _stop_robot_motion(self) -> None:
        self.auto_mode = False
        self._auto_stop_event.set()
        if self.toy is not None:
            try:
                self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
            except Exception:
                pass

    def _halt_auto(self, reason: str) -> None:
        # Single place that ends an auto run because of a detected problem.
        if not self.auto_mode:
            return
        self.auto_stop_reason = reason
        self.auto_mode = False
        self._auto_stop_event.set()
        if self.toy is not None:
            try:
                self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
            except Exception:
                pass

    def _auto_loop(self):
        try:
            while not self._auto_stop_event.is_set() and self.auto_mode and self.toy is not None:
                if self._sensors_silent():
                    self._halt_auto("the robot stopped reporting sensor data, so obstacles can't be detected")
                    break
                bump = self._take_confirmed_bump()
                if bump is None and self._is_stalled():
                    bump = ("the robot was blocked and stopped moving", "stall")
                if bump is not None:
                    if not self._recover_from_bump(bump[0], bump[1]):
                        break
                    continue
                if not self._patrol_step():
                    break
                if self._auto_stop_event.wait(timeout=self.AUTO_TICK_SECONDS):
                    break
        finally:
            is_current = self._auto_thread is threading.current_thread()
            if is_current:
                self._auto_thread = None
            # Never let a finished old loop stop a newer auto-pilot session.
            if is_current or not self.auto_mode:
                self._stop_robot_motion()
            if is_current:
                self._teardown_auto_sensors()
                self._risk_model.save()

    def _drive_auto(self, speed: int, heading: float, flags=DriveFlags.FORWARD) -> bool:
        try:
            self.toy.drive_with_heading(speed, int(heading) % 360, flags)
            return True
        except Exception:
            self._stop_robot_motion()
            return False

    def _patrol_step(self) -> bool:
        """Steer toward the next cell of the patrol grid. False means auto pilot ended."""
        now = time.monotonic()
        position = self._position
        if position is None:
            if now - self._auto_started_at > self.POSITION_WAIT_SECONDS:
                self._halt_auto("the robot isn't reporting its position, so it can't patrol safely")
                return False
            # Stay still until the first position arrives; that isn't a stall.
            self._settle_until = now + 0.3
            self._last_moving_at = now + 0.3
            return True
        if self._last_position_at is not None and now - self._last_position_at > self.POSITION_STALE_SECONDS:
            self._halt_auto("the robot stopped reporting its position")
            return False

        planner = self._planner
        x, y = position
        limit = planner.half_extent_cm + self.FENCE_MARGIN_CM
        if abs(x) > limit or abs(y) > limit:
            self._halt_auto("the robot left the patrol area")
            return False

        cell = planner.cell_of(x, y)
        if self._cal_stage is not None:
            return self._calibrate_step(position, now)
        planner.note_position(x, y)
        self._track_avoided_objects(cell)
        if cell not in self._risk_seen and cell not in planner.obstacles:
            # Learn from a safe pass: this kind of cell did not cause a bump.
            self._risk_seen.add(cell)
            self._risk_model.record_safe_pass(cell)
        if self._prev_cell is not None and cell != self._prev_cell:
            dx, dy = cell[0] - self._prev_cell[0], cell[1] - self._prev_cell[1]
            if abs(dx) + abs(dy) == 1:
                self._travel_step = (dx, dy)
        self._prev_cell = cell
        if self._goal is None or self._goal in planner.obstacles or self._goal in planner.visited:
            self._goal = planner.next_goal(cell, self._travel_step)
            if self._goal is None:
                self._halt_auto("there is nowhere left to patrol that the robot can reach")
                return False
        steps = planner.path(cell, self._goal, cost=self._route_cost)
        if steps is None:
            self._goal = None
            return True
        target_cell = steps[0] if steps else self._goal
        # Aim at the end of the straight run, so it sweeps in lines instead of
        # stopping to turn at every cell.
        if steps:
            run_step = (steps[0][0] - cell[0], steps[0][1] - cell[1])
            for index in range(1, len(steps)):
                if (steps[index][0] - steps[index - 1][0], steps[index][1] - steps[index - 1][1]) != run_step:
                    break
                target_cell = steps[index]
        target_x, target_y = planner.center(target_cell)

        bearing = math.degrees(math.atan2(target_x - x, target_y - y)) % 360.0
        desired = self._heading_for_bearing(bearing)
        self._update_heading_offset(position, desired)
        desired = self._heading_for_bearing(bearing)

        last = self._last_command_heading
        self._last_command_heading = desired
        self.heading = int(desired)
        if last is not None and abs(self._angle_difference(desired, last)) > self.TURN_IN_PLACE_DEGREES:
            # Face the new direction first instead of driving in an arc.
            if not self._drive_auto(0, desired):
                return False
            self._settle_until = now + self.TURN_PAUSE_SECONDS
            self._last_moving_at = now + self.TURN_PAUSE_SECONDS
            self._leg_start = None
            self._auto_stop_event.wait(timeout=self.TURN_PAUSE_SECONDS)
            return True
        return self._drive_auto(self._approach_speed(cell, target_cell), desired)

    def _approach_speed(self, cell: tuple, target_cell: tuple) -> int:
        # Roll gently within two cells of a known object so a misjudged position is a soft tap.
        planner = self._planner
        if planner.near_obstacle(cell, 2) or planner.near_obstacle(target_cell, 2):
            return min(self.AUTO_SPEED, self.NEAR_OBJECT_SPEED)
        return self.AUTO_SPEED

    def _route_cost(self, cell: tuple) -> float:
        # Learned risk, plus a detour penalty for squeezing past a known object.
        cost = self._risk_model.travel_cost(cell)
        if self._planner.near_obstacle(cell):
            cost += self.NEAR_OBJECT_COST
        return cost

    def _heading_for_bearing(self, bearing: float) -> float:
        # world bearing = sign * heading + offset, so heading = sign * (bearing - offset)
        return (self._heading_sign * (bearing - self._heading_offset)) % 360.0

    def _bearing_for_heading(self, heading: float) -> float:
        return (self._heading_sign * heading + self._heading_offset) % 360.0

    def _calibrate_step(self, position: tuple, now: float) -> bool:
        """Learn how headings map onto the position axes. False means auto pilot ended."""
        stage = self._cal_stage
        heading = 0.0 if stage == 0 else 90.0
        self.heading = int(heading)
        self._last_command_heading = heading
        if not self._cal_aligned:
            # Face the leg's direction before measuring so the turn isn't counted as travel.
            if not self._drive_auto(0, heading):
                return False
            self._cal_aligned = True
            self._cal_start = None
            self._settle_until = now + self.TURN_PAUSE_SECONDS
            self._last_moving_at = now + self.TURN_PAUSE_SECONDS
            self._auto_stop_event.wait(timeout=self.TURN_PAUSE_SECONDS)
            return True
        if self._cal_start is None:
            self._cal_start = position
            self._cal_started_at = now
        dx = position[0] - self._cal_start[0]
        dy = position[1] - self._cal_start[1]
        if math.hypot(dx, dy) >= self.CALIBRATION_MOVE_CM:
            measured = math.degrees(math.atan2(dx, dy)) % 360.0
            self._cal_start = None
            self._cal_aligned = False
            if stage == 0:
                self._cal_first_bearing = measured
                self._cal_stage = 1
                return self._drive_auto(0, 0)
            diff = self._angle_difference(measured, self._cal_first_bearing)
            if abs(abs(diff) - 90.0) > 45.0:
                self._halt_auto("the robot's movements didn't match its position readings, so it can't patrol safely")
                return False
            self._heading_sign = 1 if diff > 0 else -1
            self._heading_offset = self._cal_first_bearing
            self._cal_stage = None
            self._leg_start = None
            self._last_command_heading = None
            return self._drive_auto(0, heading)
        if now - self._cal_started_at > self.CALIBRATION_TIMEOUT_SECONDS:
            self._halt_auto("the robot didn't move during start-up calibration")
            return False
        return self._drive_auto(self.AUTO_SPEED, heading)

    def _track_avoided_objects(self, cell: tuple) -> None:
        """Count a known object as avoided each time the robot comes up next to it and gets by without bumping it."""
        near = {
            obstacle
            for obstacle in list(self._planner.obstacles)
            if max(abs(obstacle[0] - cell[0]), abs(obstacle[1] - cell[1])) <= 1
        }
        self.avoided_count += len(near - self._near_obstacles)
        self._near_obstacles = near

    def auto_debug(self) -> Dict[str, object]:
        planner = self._planner
        position = self._position
        return {
            "position_cm": None if position is None else [round(position[0], 1), round(position[1], 1)],
            "cell": None if position is None else list(planner.cell_of(*position)),
            "goal": None if self._goal is None else list(self._goal),
            "calibrating": self._cal_stage is not None,
            "heading_sign": self._heading_sign,
            "heading_offset": round(self._heading_offset, 1),
            "commanded_heading": self.heading,
            "bumps": self.bump_count,
            "avoided": self.avoided_count,
            "blocked_cells": sorted(list(c) for c in list(planner.obstacles)),
            "visited_cells": len(planner.visited),
        }

    def auto_map(self) -> Dict[str, object]:
        """Everything the page needs to draw the map of the current/last auto pilot run."""
        planner = self._planner
        position = self._position
        trail = list(self._trail)
        if len(trail) > self.MAP_MAX_POINTS_SENT:
            step = len(trail) // self.MAP_MAX_POINTS_SENT + 1
            trail = trail[::step] + [trail[-1]]
        if self.auto_mode:
            state = "calibrating" if self._cal_stage is not None else "patrolling"
        else:
            state = "idle"
        robot = None
        if position is not None:
            robot = {
                "x": round(position[0], 1),
                "y": round(position[1], 1),
                "bearing": round(self._bearing_for_heading(self.heading), 1),
            }
        return {
            "state": state,
            "running": bool(self.auto_mode),
            "stop_reason": self.auto_stop_reason,
            "cell_cm": planner.cell_cm,
            "half_cells": planner.half_cells,
            "robot": robot,
            "trail": [list(p) for p in trail],
            "bump_points": [list(p) for p in list(self._bump_points)],
            "blocked_cells": sorted(list(c) for c in list(planner.obstacles)),
            "walls": planner.walls(),
            "visited_cells": sorted(list(c) for c in list(planner.visited)),
            "goal": None if self._goal is None else list(self._goal),
            "laps": planner.laps,
            "bumps": self.bump_count,
            "bump_sources": dict(self.bump_sources),
            "rejected_bumps": self.rejected_bumps,
            "repeat_bumps": self.repeat_bumps,
            "position_corrections": self.position_corrections,
            "avoided": self.avoided_count,
            "risk": {
                "cells": self._risk_model.risk_map(planner.half_cells),
                "samples": self._risk_model.samples,
                "hits": self._risk_model.hits,
            },
        }

    def _update_heading_offset(self, position: tuple, heading: float) -> None:
        # The position axes may be rotated relative to the heading the robot is
        # given. Measure the real direction of travel over a leg and correct for it.
        start = self._leg_start
        if start is None or abs(self._angle_difference(heading, self._leg_heading)) > 25.0:
            self._leg_start = position
            self._leg_heading = heading
            return
        dx = position[0] - start[0]
        dy = position[1] - start[1]
        if math.hypot(dx, dy) < self.CALIBRATION_LEG_CM:
            return
        measured = math.degrees(math.atan2(dx, dy)) % 360.0
        refined = (measured - self._heading_sign * self._leg_heading) % 360.0
        # Small corrections only; a big jump means this leg was disturbed (e.g. a bump).
        if abs(self._angle_difference(refined, self._heading_offset)) < 45.0:
            self._heading_offset = refined
        self._leg_start = position
        self._leg_heading = heading

    def _undo_slip_after_bump(self) -> None:
        """Wheels keep counting while the shell is stuck against something, which makes the
        robot think it moved on through the object. Put the position estimate back to
        where it last rolled normally."""
        before = self._last_moving_position
        position = self._position
        origin = self._position_origin
        if before is None or position is None or origin is None:
            return
        shift = (before[0] - position[0], before[1] - position[1])
        slipped = math.hypot(*shift)
        if slipped < 2.0 or slipped > self.MAX_SLIP_CORRECTION_CM:
            return
        self._position_origin = (origin[0] - shift[0], origin[1] - shift[1])
        self._position = before
        self.position_corrections += 1

    def _recover_from_bump(self, reason: str, source: str = "collision") -> bool:
        """Back off, remember what is in the way. False means auto pilot ended."""
        now = time.monotonic()
        self._recent_bumps = [t for t in self._recent_bumps if now - t < self.BOXED_IN_WINDOW_SECONDS]
        self._recent_bumps.append(now)
        if len(self._recent_bumps) >= self.BOXED_IN_BUMPS:
            self._halt_auto("the robot is boxed in and keeps hitting obstacles")
            return False
        self.bump_count += 1
        self.bump_sources[source] = self.bump_sources.get(source, 0) + 1
        self._undo_slip_after_bump()

        position = self._position
        if position is not None:
            travel = math.radians(self._bearing_for_heading(self.heading))
            own_cell = self._planner.cell_of(*position)
            # If the point just ahead is still inside the robot's own cell, the object may
            # be in the next cell on; reach into that one so the real cell gets marked.
            reach = float(self.BUMP_AHEAD_CM)
            while reach < self.BUMP_AHEAD_CM + self.CELL_CM and self._planner.cell_of(
                position[0] + reach * math.sin(travel), position[1] + reach * math.cos(travel)
            ) == own_cell:
                reach += 3.0
            ahead_x = position[0] + reach * math.sin(travel)
            ahead_y = position[1] + reach * math.cos(travel)
            known_before = self._planner.near_obstacle(self._planner.cell_of(ahead_x, ahead_y), 1) or (
                self._planner.cell_of(ahead_x, ahead_y) in self._planner.obstacles
            )
            # The object it just hit is not one it avoided; it only counts after the robot leaves and returns.
            hit_cell = self._planner.mark_obstacle(ahead_x, ahead_y)
            if hit_cell is not None:
                if known_before:
                    self.repeat_bumps += 1
                self._near_obstacles.add(hit_cell)
                self._risk_model.record_bump(hit_cell)
                self._risk_model.save()
            if len(self._bump_points) < 300:
                self._bump_points.append([round(ahead_x, 1), round(ahead_y, 1)])
        self._goal = None
        self._leg_start = None
        self._cal_start = None
        self._cal_aligned = False
        self._cal_started_at = now
        self._last_command_heading = float(self.heading)

        quiet_until = now + self.BACKOFF_SECONDS + self.BUMP_SETTLE_SECONDS
        self._settle_until = quiet_until
        self._last_moving_at = quiet_until
        if not self._drive_auto(self.BACKOFF_SPEED, self.heading, DriveFlags.BACKWARD):
            return False
        if self._auto_stop_event.wait(timeout=self.BACKOFF_SECONDS):
            return False
        if not self._drive_auto(0, self.heading):
            return False
        if self._auto_stop_event.wait(timeout=self.BUMP_SETTLE_SECONDS):
            return False
        # Sensors lag a little behind the motion; discard anything caused by backing off.
        self._take_pending_bump()
        self._settle_until = time.monotonic() + 0.3
        self._last_moving_at = self._settle_until
        return True

    def _on_sensor_data(self, data) -> None:
        data = data or {}
        now = time.monotonic()
        self._last_sensor_at = now

        accel = data.get("accelerometer")
        if accel:
            current = (float(accel.get("x", 0.0)), float(accel.get("y", 0.0)), float(accel.get("z", 0.0)))
            previous = self._last_accel
            self._last_accel = current
            if (
                previous is not None
                and self.auto_mode
                and now - self._auto_started_at >= self.IMPACT_GRACE_SECONDS
                and now >= self._settle_until
                and math.sqrt(sum((c - p) ** 2 for c, p in zip(current, previous))) >= self.IMPACT_JERK_G
            ):
                self._register_bump("the robot hit an obstacle", "jerk")

        attitude = data.get("attitude")
        if attitude:
            self._check_tilt(attitude, now)

        locator = data.get("locator")
        if locator:
            raw = (float(locator.get("x", 0.0)), float(locator.get("y", 0.0)))
            if self._position_origin is None:
                self._position_origin = raw
            self._position = (raw[0] - self._position_origin[0], raw[1] - self._position_origin[1])
            self._last_position_at = now
            if self.auto_mode:
                trail = self._trail
                if not trail or math.hypot(self._position[0] - trail[-1][0], self._position[1] - trail[-1][1]) >= self.MAP_TRAIL_SPACING_CM:
                    trail.append((round(self._position[0], 1), round(self._position[1], 1)))
        velocity = data.get("velocity")
        if not velocity:
            return
        speed = math.hypot(float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0)))
        self._last_velocity_at = now
        if speed >= self.STALL_SPEED_CM_S:
            self._last_moving_at = now
            self._last_moving_position = self._position

    @staticmethod
    def _angle_difference(a: float, b: float) -> float:
        return (a - b + 180.0) % 360.0 - 180.0

    def _check_tilt(self, attitude, now: float) -> None:
        pitch = float(attitude.get("pitch", 0.0))
        roll = float(attitude.get("roll", 0.0))
        with self._turn_lock:
            if not self.auto_mode:
                return
            if self._tilt_baseline is None:
                # Whatever tilt the robot had when auto pilot began is "level".
                self._tilt_baseline = (pitch, roll)
                return
            if now - self._auto_started_at < self.TILT_GRACE_SECONDS or now < self._settle_until:
                self._tilt_since = None
                return

            base_pitch, base_roll = self._tilt_baseline
            tilt = max(
                abs(self._angle_difference(pitch, base_pitch)),
                abs(self._angle_difference(roll, base_roll)),
            )
            if tilt < self.TILT_BLOCKED_DEGREES:
                self._tilt_since = None
                return
            if self._tilt_since is None:
                self._tilt_since = now
                return
            if now - self._tilt_since >= self.TILT_BLOCKED_SECONDS:
                self._tilt_since = None
                self._register_bump("the robot tilted inside its shell, so it is blocked", "tilt")

    def _start_velocity_monitor(self) -> None:
        now = time.monotonic()
        self._auto_started_at = now
        self._last_moving_at = now
        self._last_velocity_at = None
        self._last_sensor_at = None
        self._last_accel = None
        self._tilt_baseline = None
        self._tilt_since = None
        self._settle_until = 0.0
        self._recent_bumps = []
        self._pending_bump = None
        self.bump_count = 0
        self.bump_sources = {}
        self.rejected_bumps = 0
        self.position_corrections = 0
        self.repeat_bumps = 0
        self.avoided_count = 0
        self._near_obstacles = set()
        self._planner = PatrolPlanner(self.ARENA_HALF_CELLS, self.CELL_CM)
        self._prev_cell = None
        self._last_moving_position = None
        self._travel_step = None
        self._risk_model.start_run(self.ARENA_HALF_CELLS)
        self._risk_seen = set()
        self._position = None
        self._position_origin = None
        self._last_position_at = None
        self._trail.clear()
        self._bump_points = []
        self._goal = None
        self._heading_offset = 0.0
        self._heading_sign = 1
        self._cal_stage = 0
        self._cal_start = None
        self._cal_started_at = now
        self._cal_aligned = False
        self._leg_start = None
        self._last_command_heading = None
        self._sensor_feedback = False
        sensor_control = getattr(self.toy, "sensor_control", None)
        if sensor_control is None:
            return
        try:
            sensor_control.add_sensor_data_listener(self._on_sensor_data)
            self._velocity_listener = self._on_sensor_data
            sensor_control.set_count(0)
            sensor_control.set_interval(50)
            sensor_control.enable("velocity", "accelerometer", "attitude", "locator")
        except Exception as exc:
            raise RuntimeError(
                "Could not start the robot's motion sensors, so auto pilot can't detect obstacles. "
                "Disconnect and reconnect the robot, then try again."
            ) from exc
        self._sensor_feedback = True

    def _sensors_silent(self) -> bool:
        if not self._sensor_feedback:
            return False
        reference = self._last_sensor_at if self._last_sensor_at is not None else self._auto_started_at
        return time.monotonic() - reference > self.SENSOR_SILENCE_SECONDS

    def _is_stalled(self) -> bool:
        if self._last_velocity_at is None:
            return False
        now = time.monotonic()
        if now - self._auto_started_at < self.STALL_GRACE_SECONDS or now < self._settle_until:
            return False
        if now - self._last_velocity_at > self.VELOCITY_STALE_SECONDS:
            return False
        return now - self._last_moving_at >= self.STALL_SECONDS

    def _teardown_auto_sensors(self) -> None:
        toy = self.toy
        collision_listener = self._collision_listener
        velocity_listener = self._velocity_listener
        self._collision_listener = None
        self._velocity_listener = None
        if toy is None:
            return

        if hasattr(toy, "enable_collision_detected_notify"):
            try:
                toy.enable_collision_detected_notify(False)
            except Exception:
                pass

        if collision_listener is not None and hasattr(toy, "remove_collision_detected_notify_listener"):
            try:
                toy.remove_collision_detected_notify_listener(collision_listener)
            except Exception:
                pass

        if velocity_listener is not None:
            try:
                sensor_control = getattr(toy, "sensor_control", None)
                if sensor_control is not None:
                    sensor_control.remove_sensor_data_listener(velocity_listener)
                    sensor_control.disable("velocity", "accelerometer", "attitude", "locator")
            except Exception:
                pass

    def set_auto_mode(self, enabled: bool = True) -> Dict[str, object]:
        if enabled:
            with self._auto_state_lock:
                generation = self._auto_generation
            self._verify_toy_is_responding()
            if self.auto_mode and self._auto_thread is not None and self._auto_thread.is_alive():
                return {"enabled": True, "status": "auto mode already active"}

            self._cancel_drive_watchdog()
            self._cancel_emergency_reset()
            previous = self._auto_thread
            if previous is not None and previous is not threading.current_thread() and previous.is_alive():
                # An old session that is ending on its own must finish shutting down first,
                # or it would stop the new session when it finishes.
                previous.join(timeout=1.0)
                if previous.is_alive():
                    raise RuntimeError("The previous auto pilot run is still shutting down. Try again in a moment.")
            self.auto_stop_reason = None
            self._auto_stop_event.clear()
            self._last_collision_at = 0.0

            try:
                if hasattr(self.toy, "configure_collision_detection"):
                    # Same detection mode the Sphero library itself uses; thresholds are
                    # a little more sensitive than its defaults because patrol is slow.
                    self.toy.configure_collision_detection(
                        CollisionDetectionMethods.ACCELEROMETER_BASED_DETECTION,
                        60,
                        60,
                        60,
                        60,
                        10,
                    )

                if hasattr(self.toy, "enable_collision_detected_notify"):
                    self.toy.enable_collision_detected_notify(True)

                if hasattr(self.toy, "add_collision_detected_notify_listener"):
                    self._collision_listener = self._handle_collision
                    self.toy.add_collision_detected_notify_listener(self._collision_listener)

                self._start_velocity_monitor()
            except Exception:
                # A half-started auto pilot must never leave auto_mode set: that would
                # disable the manual-drive watchdog with nothing driving the robot.
                self.auto_mode = False
                self._auto_stop_event.set()
                self._teardown_auto_sensors()
                raise

            # Only now is the robot actually under auto-pilot control, unless a stop
            # request arrived while sensors were being set up (stop always wins).
            with self._auto_state_lock:
                cancelled = generation != self._auto_generation
                if not cancelled:
                    self.auto_mode = True
            if cancelled:
                self._auto_stop_event.set()
                self._teardown_auto_sensors()
                return {"enabled": False, "status": "auto mode start cancelled"}

            if self._auto_thread is None or not self._auto_thread.is_alive():
                self._auto_thread = threading.Thread(target=self._auto_loop, daemon=True)
                self._auto_thread.start()

            return {"enabled": True, "status": "auto mode activated"}

        self._cancel_pending_auto_start()
        self.auto_mode = False
        self._auto_stop_event.set()
        self._last_collision_at = 0.0

        if self.toy is not None:
            try:
                self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
            except Exception:
                pass

        self._teardown_auto_sensors()

        if self._auto_thread is not None and self._auto_thread.is_alive() \
                and self._auto_thread is not threading.current_thread():
            self._auto_thread.join(timeout=0.5)
        self._auto_thread = None

        return {"enabled": False, "status": "auto mode deactivated"}

    def reset_to_default(self) -> Dict[str, object]:
        self.auto_mode = False
        self._auto_stop_event.set()
        self._last_collision_at = 0.0
        self.heading = 0

        if self.toy is not None:
            try:
                if hasattr(self.toy, "set_heading"):
                    self.toy.set_heading(self.heading)
                self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
            except Exception:
                pass

        if self.toy is not None:
            try:
                self.display_message("Ready", color=(0, 255, 0), speed=8, repeat=False)
            except Exception:
                pass

        if self._auto_thread is not None and self._auto_thread.is_alive():
            self._auto_thread.join(timeout=0.25)
        self._auto_thread = None
        self._collision_listener = None
        return {"status": "default restored", "text": "Ready"}

    def emergency_stop(self, delay_seconds: float = 10.0) -> Dict[str, object]:
        self.set_auto_mode(False)
        self._auto_stop_event.set()
        if self.toy is not None:
            try:
                self.toy.drive_with_heading(0, self.heading, DriveFlags.FORWARD)
            except Exception:
                pass

        self._cancel_emergency_reset()
        timer = threading.Timer(float(delay_seconds), self._finish_emergency_stop)
        timer.daemon = True
        self._emergency_timer = timer
        timer.start()
        return {"status": "emergency stop active", "seconds": float(delay_seconds), "default": "Ready"}

    def _cancel_emergency_reset(self) -> None:
        timer = self._emergency_timer
        self._emergency_timer = None
        if timer is not None:
            timer.cancel()

    def _finish_emergency_stop(self) -> None:
        self._emergency_timer = None
        # If auto pilot was started again in the meantime, don't kill it.
        if self.auto_mode:
            return
        self.reset_to_default()

    def display_message(self, text: str, color=(0, 255, 0), speed: int = 8, repeat: bool = False) -> Dict[str, object]:
        self._verify_toy_is_responding()

        message = str(text or "Ready").strip() or "Ready"
        message = message[:20]
        r, g, b = [max(0, min(255, int(channel))) for channel in color]
        speed_value = max(1, min(20, int(speed)))
        repeat_value = bool(repeat)

        if hasattr(self.toy, "reset_compressed_frame_player_animation"):
            try:
                self.toy.reset_compressed_frame_player_animation()
            except Exception:
                pass

        if hasattr(self.toy, "set_compressed_frame_player_text_scrolling"):
            self.toy.set_compressed_frame_player_text_scrolling(message, r, g, b, speed_value, repeat_value)
            if hasattr(self.toy, "release_led_requests"):
                try:
                    self.toy.release_led_requests()
                except Exception:
                    pass
            return {"text": message, "color": [r, g, b], "speed": speed_value, "repeat": repeat_value}

        if hasattr(self.toy, "set_compressed_frame_player_single_character"):
            self.toy.set_compressed_frame_player_single_character(r, g, b, message[0])
            if hasattr(self.toy, "release_led_requests"):
                try:
                    self.toy.release_led_requests()
                except Exception:
                    pass
            return {"text": message, "color": [r, g, b], "speed": speed_value, "repeat": repeat_value}

        if hasattr(self.toy, "set_main_led"):
            self.toy.set_main_led(r, g, b)
            return {"text": message, "color": [r, g, b], "speed": speed_value, "repeat": repeat_value}

        raise RuntimeError("This robot does not support a custom display message.")

    def display_pixel_art(self, pixels, color=(255, 255, 255), clear_first: bool = False) -> Dict[str, object]:
        self._verify_toy_is_responding()

        normalized_pixels = []
        pixel_colors = {}
        for item in pixels or []:
            item_color = None
            if isinstance(item, dict):
                x = item.get("x")
                y = item.get("y")
                if x is None or y is None:
                    continue
                item_color = item.get("color")
            else:
                try:
                    if len(item) == 5:
                        x, y = item[0], item[1]
                        item_color = item[2:5]
                    else:
                        x, y = item
                except (TypeError, ValueError):
                    continue

            try:
                position = (int(x), int(y))
                if item_color is not None:
                    item_color = tuple(max(0, min(255, int(c))) for c in item_color)
                    if len(item_color) != 3:
                        item_color = None
            except (TypeError, ValueError):
                continue

            normalized_pixels.append(position)
            if item_color is not None:
                pixel_colors[position] = item_color

        if not normalized_pixels:
            raise ValueError("Pick at least one pixel to display.")

        r, g, b = [max(0, min(255, int(channel))) for channel in color]
        if hasattr(self.toy, "reset_compressed_frame_player_animation"):
            try:
                self.toy.reset_compressed_frame_player_animation()
            except Exception:
                pass

        if clear_first and hasattr(self.toy, "set_compressed_frame_player_pixel"):
            for x in range(8):
                for y in range(8):
                    self.toy.set_compressed_frame_player_pixel(x, y, 0, 0, 0)

        if hasattr(self.toy, "set_compressed_frame_player_pixel"):
            for x, y in normalized_pixels:
                if 0 <= x < 8 and 0 <= y < 8:
                    pr, pg, pb = pixel_colors.get((x, y), (r, g, b))
                    self.toy.set_compressed_frame_player_pixel(x, y, pr, pg, pb)
            if hasattr(self.toy, "release_led_requests"):
                try:
                    self.toy.release_led_requests()
                except Exception:
                    pass
            return {"pixels": normalized_pixels, "color": [r, g, b]}

        raise RuntimeError("This robot does not support custom pixel art on the display.")
