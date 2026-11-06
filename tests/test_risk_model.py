import json

from webapp.patrol_planner import PatrolPlanner
from webapp.risk_model import PRIOR_WEIGHTS, RiskModel
from webapp.robot_controller import RobotController


def test_cells_next_to_a_bump_look_riskier_than_far_away_cells():
    model = RiskModel()
    model.start_run(3)
    far = model.predict((-2, -2))
    model.record_bump((1, 1))
    assert model.predict((1, 1)) > far
    assert model.predict((1, 2)) > far  # beside the bump
    assert model.predict((2, 2)) > far  # diagonal to the bump
    assert model.predict((1, 2)) > model.predict((-2, -2))


def test_model_learns_from_experience():
    model = RiskModel()
    model.start_run(3)
    # Object hits keep happening on the edge ring, never in the middle.
    for _ in range(60):
        model.record_bump((3, 0))
        model.bump_counts.clear()
        model.record_safe_pass((0, 0))
        model.record_safe_pass((1, 0))
    assert model.weights != PRIOR_WEIGHTS
    assert model.predict((-3, 1)) > 3 * model.predict((0, 1)), "edge cells should now look riskier"
    assert model.samples == 180
    assert model.hits == 60


def test_each_run_starts_with_no_bump_counts_but_keeps_the_learning():
    model = RiskModel()
    model.start_run(3)
    model.record_bump((1, 0))
    learned = list(model.weights)
    model.start_run(3)
    assert model.bump_counts == {}
    assert model.weights == learned


def test_model_is_saved_and_loaded_and_survives_a_damaged_file(tmp_path):
    path = tmp_path / "sub" / "risk.json"
    model = RiskModel(str(path))
    model.start_run(3)
    for _ in range(10):
        model.record_bump((3, 3))
        model.record_safe_pass((0, 0))
    model.save()

    reloaded = RiskModel(str(path))
    assert reloaded.weights == model.weights
    assert reloaded.samples == 20 and reloaded.hits == 10

    path.write_text("{not json")
    broken = RiskModel(str(path))
    assert broken.weights == PRIOR_WEIGHTS and broken.samples == 0

    path.write_text(json.dumps({"weights": [1, 2], "features": ["x"]}))
    assert RiskModel(str(path)).weights == PRIOR_WEIGHTS

    assert RiskModel().path is None
    RiskModel().save()  # no path: nothing happens


def test_planner_routes_around_a_risky_cell_but_still_reaches_everything():
    planner = PatrolPlanner(half_cells=1)
    assert planner.path((-1, 0), (1, 0)) == [(0, 0), (1, 0)]

    def cost(cell):
        return 50.0 if cell == (0, 0) else 1.0

    detour = planner.path((-1, 0), (1, 0), cost=cost)
    assert (0, 0) not in detour
    assert detour[-1] == (1, 0)
    assert len(detour) == 4

    # Blocked cells are still never used, and unreachable goals still report None.
    planner.obstacles.update({(0, 1), (0, -1), (0, 0)})
    assert planner.path((-1, 0), (1, 0), cost=cost) is None


def test_controller_learns_from_a_bump_saves_and_reports_it_on_the_map(tmp_path):
    path = tmp_path / "risk.json"
    controller = RobotController(risk_model_path=str(path))
    controller._risk_model.start_run(3)
    before = controller._risk_model.samples

    controller._risk_model.record_bump((1, 0))
    controller._risk_model.save()
    assert controller._risk_model.samples == before + 1
    assert path.exists()

    risk = controller.auto_map()["risk"]
    assert risk["samples"] == before + 1 and risk["hits"] == 1
    size = 2 * controller._planner.half_cells + 1
    assert len(risk["cells"]) == size * size
    assert all(0.0 <= c[2] <= 1.0 for c in risk["cells"])

    # A new controller picks the learned weights back up.
    assert RobotController(risk_model_path=str(path))._risk_model.weights == controller._risk_model.weights
