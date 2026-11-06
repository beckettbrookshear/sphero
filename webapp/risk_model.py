"""Small online-learning model of how likely a patrol cell is to cause a bump.

It is a logistic regression trained one sample at a time (stochastic gradient
descent), with no external libraries. Every time the robot drives into a cell
without hitting anything that is a "safe" sample; every bump is a "hit" sample.
The features describe a cell's surroundings (bumps already seen in the cell and
next to it, and whether it is on the edge of the arena), so what is learned
carries over between runs even though each run has its own coordinate origin.
Only the weights are saved; the per-cell bump counts belong to one run.

The patrol planner uses the predicted risk as an extra travel cost, so routes
bend away from cells that look like they hide an object.
"""

import json
import math
import os
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

Cell = Tuple[int, int]

FEATURE_NAMES = ("bias", "own_bumps", "side_neighbour_bumps", "corner_neighbour_bumps", "edge")
# Starting weights: a cell with bumps in or beside it starts out looking riskier.
PRIOR_WEIGHTS = [-2.0, 1.0, 0.8, 0.4, 0.3]
_SIDE_STEPS = ((0, 1), (0, -1), (1, 0), (-1, 0))
_CORNER_STEPS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class RiskModel:
    LEARNING_RATE = 0.15
    HIT_SAMPLE_WEIGHT = 4.0  # bumps are rare, so each one counts for more than a safe pass
    L2 = 0.001
    WEIGHT_LIMIT = 8.0
    SAVE_VERSION = 1

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = None if path is None else str(path)
        self.half_cells = 3
        self.weights: List[float] = list(PRIOR_WEIGHTS)
        self.samples = 0
        self.hits = 0
        self.bump_counts: Dict[Cell, int] = {}
        self._lock = threading.Lock()
        self._load()

    # ---- per-run state -------------------------------------------------------
    def start_run(self, half_cells: int) -> None:
        """Forget this run's bump counts (cells are relative to the start point) but keep what was learned."""
        with self._lock:
            self.half_cells = int(half_cells)
            self.bump_counts = {}

    # ---- model ---------------------------------------------------------------
    def features(self, cell: Cell) -> List[float]:
        counts = self.bump_counts
        side = sum(counts.get((cell[0] + dx, cell[1] + dy), 0) for dx, dy in _SIDE_STEPS)
        corner = sum(counts.get((cell[0] + dx, cell[1] + dy), 0) for dx, dy in _CORNER_STEPS)
        on_edge = 1.0 if max(abs(cell[0]), abs(cell[1])) >= self.half_cells else 0.0
        return [
            1.0,
            min(counts.get(cell, 0), 3) / 3.0,
            min(side, 4) / 4.0,
            min(corner, 4) / 4.0,
            on_edge,
        ]

    def predict(self, cell: Cell) -> float:
        """Estimated chance (0..1) of a bump when driving through this cell."""
        with self._lock:
            return self._predict(cell)

    def _predict(self, cell: Cell) -> float:
        x = self.features(cell)
        return _sigmoid(sum(w * v for w, v in zip(self.weights, x)))

    def _learn(self, cell: Cell, hit: bool) -> None:
        x = self.features(cell)
        p = _sigmoid(sum(w * v for w, v in zip(self.weights, x)))
        error = (1.0 if hit else 0.0) - p
        sample_weight = self.HIT_SAMPLE_WEIGHT if hit else 1.0
        limit = self.WEIGHT_LIMIT
        for k, value in enumerate(x):
            step = self.LEARNING_RATE * sample_weight * (error * value - self.L2 * self.weights[k])
            self.weights[k] = max(-limit, min(limit, self.weights[k] + step))
        self.samples += 1

    def record_safe_pass(self, cell: Cell) -> None:
        with self._lock:
            self._learn(cell, hit=False)

    def record_bump(self, cell: Cell) -> None:
        with self._lock:
            # Learn from the situation as it was before this bump, then remember the bump.
            self._learn(cell, hit=True)
            self.hits += 1
            self.bump_counts[cell] = self.bump_counts.get(cell, 0) + 1

    def travel_cost(self, cell: Cell, risk_weight: float = 10.0) -> float:
        """Cost of stepping into a cell: 1 plus a penalty that grows with the predicted risk."""
        return 1.0 + risk_weight * self.predict(cell)

    def risk_map(self, half_cells: int) -> List[List[float]]:
        """[column, row, risk] for every cell of the arena, for drawing."""
        with self._lock:
            return [
                [i, j, round(self._predict((i, j)), 4)]
                for i in range(-half_cells, half_cells + 1)
                for j in range(-half_cells, half_cells + 1)
            ]

    # ---- persistence ---------------------------------------------------------
    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {
                "version": self.SAVE_VERSION,
                "features": list(FEATURE_NAMES),
                "weights": list(self.weights),
                "samples": self.samples,
                "hits": self.hits,
            }
        directory = os.path.dirname(self.path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            handle, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(data, stream)
            os.replace(temp_path, self.path)
        except OSError:
            # Learning still works for this session even if the file can't be written.
            pass

    def _load(self) -> None:
        if self.path is None or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            weights = [float(w) for w in data["weights"]]
            if data.get("features") != list(FEATURE_NAMES) or len(weights) != len(PRIOR_WEIGHTS):
                return
            if not all(math.isfinite(w) for w in weights):
                return
            self.weights = weights
            self.samples = int(data.get("samples", 0))
            self.hits = int(data.get("hits", 0))
        except (OSError, ValueError, KeyError, TypeError):
            # A missing or damaged file just means starting from the built-in prior.
            self.weights = list(PRIOR_WEIGHTS)
            self.samples = 0
            self.hits = 0
