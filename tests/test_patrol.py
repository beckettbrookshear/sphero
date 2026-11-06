import math
import threading
import time

from webapp.patrol_planner import PatrolPlanner
from webapp.robot_controller import RobotController
from spherov2.commands.drive import DriveFlags


def test_planner_cells_and_centres():
    planner = PatrolPlanner(half_cells=3, cell_cm=30.0)
    assert planner.cell_of(0, 0) == (0, 0)
    assert planner.cell_of(44, -44) == (1, -1)
    assert planner.cell_of(1000, -1000) == (3, -3)
    assert planner.center((2, -1)) == (60.0, -30.0)


def test_planner_visits_nearest_unvisited_cell_and_starts_new_lap():
    planner = PatrolPlanner(half_cells=1, cell_cm=30.0)
    planner.note_position(0, 0)
    assert planner.visited == {(0, 0)}
    first = planner.next_goal((0, 0))
    assert first in {(0, 1), (0, -1), (1, 0), (-1, 0)}

    for i in range(-1, 2):
        for j in range(-1, 2):
            planner.visited.add((i, j))
    assert planner.next_goal((0, 0)) is not None
    assert planner.laps == 1


def test_planner_routes_around_blocked_cells():
    planner = PatrolPlanner(half_cells=2, cell_cm=30.0)
    planner.mark_obstacle(30, 0)  # cell (1, 0)

    path = planner.path((0, 0), (2, 0))

    assert path is not None
    assert (1, 0) not in path
    assert path[-1] == (2, 0)


def test_planner_reports_unreachable_goal_and_never_picks_blocked_cells():
    planner = PatrolPlanner(half_cells=1, cell_cm=30.0)
    for cell in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        planner.mark_obstacle(*planner.center(cell))

    assert planner.path((0, 0), (1, 1)) is None
    planner.visited.add((0, 0))
    assert planner.next_goal((0, 0)) is None or planner.next_goal((0, 0)) == (0, 0)
    assert planner.mark_obstacle(5000, 0) is None


class _SimToy:
    """A robot on a floor with one box. Its position axes are rotated 90 degrees
    from the headings it is given, like a real robot may be."""

    OFFSET = 90.0
    SIGN = 1
    CM_PER_SECOND_PER_SPEED = 4.0

    class _Sensors:
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

    def __init__(self, box, sign=1, offset=90.0):
        self.SIGN = sign
        self.OFFSET = offset
        self.sensor_control = self._Sensors()
        self.box = box  # (xmin, xmax, ymin, ymax) in world cm
        self.x = 500.0  # arbitrary world origin; the controller must not care
        self.y = -200.0
        self.start = (self.x, self.y)
        self.speed = 0
        self.heading = 0
        self.backward = False
        self.collision_listener = None
        self.last_collision = 0.0
        self.max_inside_box = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # robot api used by the controller
    def drive_with_heading(self, speed, heading, flags):
        self.speed = speed
        self.heading = heading
        self.backward = flags == DriveFlags.BACKWARD

    def configure_collision_detection(self, *args):
        pass

    def enable_collision_detected_notify(self, enabled):
        pass

    def add_collision_detected_notify_listener(self, callback):
        self.collision_listener = callback

    def remove_collision_detected_notify_listener(self, callback):
        self.collision_listener = None

    def rel(self):
        return (self.x - self.start[0], self.y - self.start[1])

    def in_box(self, x, y):
        rx, ry = x - self.start[0], y - self.start[1]
        xmin, xmax, ymin, ymax = self.box
        return xmin <= rx <= xmax and ymin <= ry <= ymax

    def begin(self):
        self._thread.start()

    def finish(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        dt = 0.02
        while not self._stop.is_set():
            world = math.radians(self.SIGN * self.heading - self.OFFSET)
            direction = -1.0 if self.backward else 1.0
            distance = direction * self.speed * self.CM_PER_SECOND_PER_SPEED * dt
            nx = self.x + distance * math.sin(world)
            ny = self.y + distance * math.cos(world)
            blocked = self.in_box(nx, ny)
            if blocked:
                if self.collision_listener is not None and time.monotonic() - self.last_collision > 0.3:
                    self.last_collision = time.monotonic()
                    self.collision_listener({"x_axis": True})
            else:
                self.x, self.y = nx, ny
            listener = self.sensor_control.listener
            if listener is not None:
                moving = 0.0 if blocked else self.speed * self.CM_PER_SECOND_PER_SPEED
                listener({"locator": {"x": self.x, "y": self.y}, "velocity": {"x": moving, "y": 0.0}})
            time.sleep(dt)


def test_patrol_covers_the_area_learns_a_box_and_keeps_going():
    _run_patrol_with_box(sign=1, offset=90.0)


def test_patrol_works_when_position_axes_are_mirrored_relative_to_headings():
    _run_patrol_with_box(sign=-1, offset=-30.0)


def _run_patrol_with_box(sign, offset):
    controller = RobotController()
    controller.ARENA_HALF_CELLS = 2
    controller.AUTO_SPEED = 30
    controller.IMPACT_GRACE_SECONDS = 0.1
    controller.TILT_GRACE_SECONDS = 0.1
    controller.BACKOFF_SECONDS = 0.15
    controller.BUMP_SETTLE_SECONDS = 0.1
    controller.TURN_PAUSE_SECONDS = 0.1
    controller.BOXED_IN_BUMPS = 20

    toy = _SimToy(box=(40.0, 80.0, -20.0, 20.0), sign=sign, offset=offset)  # sits at the (1, 0)/(2, 0) cell edge
    controller.toy = toy
    toy.begin()
    try:
        controller.set_auto_mode(True)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            planner = controller._planner
            if controller.bump_count >= 1 and len(planner.visited) >= 8:
                break
            time.sleep(0.1)

        assert controller.auto_mode is True, controller.auto_stop_reason
        assert controller.bump_count >= 1
        assert controller._heading_sign == sign
        assert controller._planner.obstacles, "a bump should be remembered as a blocked cell"
        assert len(controller._planner.visited) >= 8
        # The robot never ended up inside the box (sanity check of the simulation).
        assert not toy.in_box(toy.x, toy.y)
    finally:
        controller.set_auto_mode(False)
        toy.finish()

    assert toy.speed == 0


def test_patrol_stops_if_robot_leaves_the_patrol_area():
    controller = RobotController()
    controller.toy = _SimToy(box=(10000, 10001, 10000, 10001))
    controller.set_auto_mode(True)
    try:
        controller._position_origin = (0.0, 0.0)
        controller._position = (5000.0, 0.0)
        controller._last_position_at = time.monotonic()

        assert controller._patrol_step() is False
        assert controller.auto_mode is False
        assert "left the patrol area" in controller.auto_stop_reason
    finally:
        controller.set_auto_mode(False)


def test_patrol_stops_if_position_never_arrives():
    controller = RobotController()
    controller.POSITION_WAIT_SECONDS = 0.2
    controller.toy = _SimToy(box=(10000, 10001, 10000, 10001))  # never started: no data

    controller.set_auto_mode(True)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and controller.auto_mode:
        time.sleep(0.05)

    assert controller.auto_mode is False
    assert "position" in controller.auto_stop_reason or "sensor data" in controller.auto_stop_reason


def test_delayed_emergency_reset_does_not_kill_a_newly_started_auto_pilot():
    controller = RobotController()
    controller.toy = _SimToy(box=(10000, 10001, 10000, 10001))

    controller.emergency_stop(delay_seconds=60)
    controller.set_auto_mode(True)
    try:
        assert controller._emergency_timer is None  # starting auto pilot cancels the pending reset
        controller._finish_emergency_stop()  # even if it fired anyway
        assert controller.auto_mode is True
    finally:
        controller.set_auto_mode(False)


def test_auto_pilot_can_be_started_again_right_after_it_ended_by_itself():
    controller = RobotController()
    controller.POSITION_WAIT_SECONDS = 0.1
    controller.toy = _SimToy(box=(10000, 10001, 10000, 10001))  # never started: no data

    for _ in range(3):
        controller.POSITION_WAIT_SECONDS = 0.1
        controller.set_auto_mode(True)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and controller.auto_mode:
            time.sleep(0.01)
        assert controller.auto_mode is False
        # Start again immediately, while the old thread may still be finishing.
    controller.set_auto_mode(False)


def test_auto_map_records_path_and_bumps_keeps_them_after_stop_and_resets_on_next_run():
    controller = RobotController()
    controller.ARENA_HALF_CELLS = 2
    controller.IMPACT_GRACE_SECONDS = 0.1
    controller.TILT_GRACE_SECONDS = 0.1
    controller.BACKOFF_SECONDS = 0.15
    controller.BUMP_SETTLE_SECONDS = 0.1
    controller.TURN_PAUSE_SECONDS = 0.1
    controller.BOXED_IN_BUMPS = 20

    toy = _SimToy(box=(40.0, 80.0, -20.0, 20.0), sign=1, offset=90.0)
    controller.toy = toy
    toy.begin()
    try:
        assert controller.auto_map()["state"] == "idle"
        assert controller.auto_map()["trail"] == []

        controller.set_auto_mode(True)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if controller.bump_count >= 1 and len(controller._trail) >= 30:
                break
            time.sleep(0.1)
        running = controller.auto_map()
        assert running["running"] is True
        assert running["state"] in ("calibrating", "patrolling")
        assert running["robot"] is not None
        assert len(running["trail"]) >= 30
        assert running["bump_points"], "a bump should be shown on the map"
        assert running["blocked_cells"]
        assert {"cell_cm", "half_cells", "visited_cells", "goal", "laps", "bumps", "avoided", "stop_reason"} <= set(running)

        controller.set_auto_mode(False)
        stopped = controller.auto_map()
        assert stopped["running"] is False and stopped["state"] == "idle"
        assert len(stopped["trail"]) >= len(running["trail"]), "the map stays visible after stopping"

        controller.set_auto_mode(True)
        fresh = controller.auto_map()
        assert len(fresh["trail"]) < len(stopped["trail"])
        assert fresh["bump_points"] == []
    finally:
        controller.set_auto_mode(False)
        toy.finish()


def test_auto_map_endpoint_and_page_canvas():
    from webapp.app import app

    client = app.test_client()
    response = client.get("/api/auto-map")
    assert response.status_code == 200
    data = response.get_json()
    assert {"state", "trail", "bump_points", "blocked_cells", "visited_cells", "robot"} <= set(data)

    page = client.get("/").get_data(as_text=True)
    assert 'id="map-canvas"' in page
    assert "/api/auto-map" in page


def test_objects_avoided_counts_each_pass_but_not_the_object_just_hit():
    controller = RobotController()
    planner = controller._planner
    planner.obstacles.add((2, 0))
    assert controller.avoided_count == 0

    controller._track_avoided_objects((0, 0))  # far away
    assert controller.avoided_count == 0

    controller._track_avoided_objects((1, 1))  # comes up next to it
    controller._track_avoided_objects((1, 0))  # still beside the same object
    controller._track_avoided_objects((1, -1))
    assert controller.avoided_count == 1

    controller._track_avoided_objects((-1, 0))  # leaves, then comes back for a second pass
    controller._track_avoided_objects((1, 0))
    assert controller.avoided_count == 2

    # An object it has just bumped is already "near", so the bump itself is not an avoidance.
    controller._near_obstacles.add((0, 2))
    planner.obstacles.add((0, 2))
    controller._track_avoided_objects((0, 1))
    assert controller.avoided_count == 2


def test_planner_prefers_to_keep_going_straight_when_choosing_the_next_cell():
    from webapp.patrol_planner import PatrolPlanner

    planner = PatrolPlanner(half_cells=3)
    planner.visited = {(0, 0)}

    assert planner.next_goal((0, 0), prefer=(1, 0)) == (1, 0)
    assert planner.next_goal((0, 0), prefer=(0, -1)) == (0, -1)
    assert planner.next_goal((0, 0), prefer=(-1, 0)) == (-1, 0)


def test_a_straight_line_of_blocked_cells_is_reported_as_a_wall_but_lone_objects_are_not():
    planner = PatrolPlanner(half_cells=5, cell_cm=30.0)
    for j in range(-2, 2):
        planner.obstacles.add((3, j))  # a vertical line of 4 cells
    planner.obstacles.add((-3, 0))  # a lone object
    planner.obstacles.update({(0, -4), (1, -4)})  # too short to be a wall

    assert planner.walls() == [{"x1": 90.0, "y1": -60.0, "x2": 90.0, "y2": 30.0}]
    assert planner.walls(min_cells=2)[0:1] != []


def test_planner_keeps_goals_away_from_known_objects_until_nothing_else_is_left():
    planner = PatrolPlanner(half_cells=1, cell_cm=30.0)
    planner.obstacles.add((1, 1))
    # Everything except the cells touching the object and (-1, -1) is visited.
    planner.visited = {(0, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)}

    assert planner.near_obstacle((0, 1)) and planner.near_obstacle((0, 0))
    assert planner.next_goal((0, 0)) == (-1, -1)
    planner.visited.add((-1, -1))
    assert planner.next_goal((0, 0)) in {(0, 1), (1, 0)}


def test_route_prefers_cells_that_do_not_touch_a_known_object():
    controller = RobotController()
    controller._planner.obstacles.add((1, 0))

    assert controller._route_cost((1, 1)) > controller._route_cost((-1, -1))


def test_robot_slows_down_within_two_cells_of_a_known_object():
    controller = RobotController()
    controller._planner.obstacles.add((3, 0))

    assert controller._approach_speed((0, 0), (1, 0)) == controller.NEAR_OBJECT_SPEED
    assert controller._approach_speed((0, 0), (0, 1)) == controller.AUTO_SPEED
    assert controller.NEAR_OBJECT_SPEED < controller.AUTO_SPEED
