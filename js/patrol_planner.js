export class PatrolPlanner {
  constructor(halfCells = 3, cellCm = 30.0, visitRadiusCm = 12.0) {
    this.halfCells = Number(halfCells);
    this.cellCm = Number(cellCm);
    this.visitRadiusCm = Number(visitRadiusCm);
    this.obstacles = new Set();
    this.visited = new Set();
    this.laps = 0;
  }

  inBounds(cell) {
    return Math.abs(cell[0]) <= this.halfCells && Math.abs(cell[1]) <= this.halfCells;
  }

  cellOf(x, y) {
    const i = Math.floor(x / this.cellCm + 0.5);
    const j = Math.floor(y / this.cellCm + 0.5);
    const limit = this.halfCells;
    return [Math.max(-limit, Math.min(limit, i)), Math.max(-limit, Math.min(limit, j))];
  }

  center(cell) {
    return [cell[0] * this.cellCm, cell[1] * this.cellCm];
  }

  get halfExtentCm() {
    return (this.halfCells + 0.5) * this.cellCm;
  }

  markObstacle(x, y) {
    if (Math.abs(x) > this.halfExtentCm || Math.abs(y) > this.halfExtentCm) {
      return null;
    }
    const cell = this.cellOf(x, y);
    this.obstacles.add(JSON.stringify(cell));
    this.visited.delete(JSON.stringify(cell));
    return cell;
  }

  walls(minCells = 3) {
    const found = [];
    for (const alongX of [true, false]) {
      const lines = new Map();
      for (const obstacle of this.obstacles) {
        const cell = JSON.parse(obstacle);
        const key = alongX ? cell[1] : cell[0];
        const value = alongX ? cell[0] : cell[1];
        if (!lines.has(key)) lines.set(key, []);
        lines.get(key).push(value);
      }
      for (const [key, values] of lines) {
        const sorted = [...values].sort((a, b) => a - b);
        let run = [sorted[0]];
        for (let idx = 1; idx <= sorted.length; idx += 1) {
          const value = sorted[idx];
          if (value !== undefined && value === run[run.length - 1] + 1) {
            run.push(value);
            continue;
          }
          if (run.length >= minCells) {
            const a = run[0] * this.cellCm;
            const b = run[run.length - 1] * this.cellCm;
            const fixed = key * this.cellCm;
            if (alongX) {
              found.push({ x1: a, y1: fixed, x2: b, y2: fixed });
            } else {
              found.push({ x1: fixed, y1: a, x2: fixed, y2: b });
            }
          }
          if (value !== undefined) run = [value];
        }
      }
    }
    return found;
  }

  notePosition(x, y) {
    const cell = this.cellOf(x, y);
    const [cx, cy] = this.center(cell);
    if (Math.hypot(x - cx, y - cy) <= this.visitRadiusCm && !this.obstacles.has(JSON.stringify(cell))) {
      this.visited.add(JSON.stringify(cell));
    }
  }

  _search(start, prefer = null) {
    const steps = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    const preferred = prefer ? JSON.stringify(prefer) : null;
    if (prefer !== null) {
      const withPrefer = [prefer, ...steps.filter(([dx, dy]) => !(dx === prefer[0] && dy === prefer[1]))];
      steps.splice(0, steps.length, ...withPrefer);
    }

    const parents = new Map();
    const queue = [start];
    parents.set(JSON.stringify(start), null);
    while (queue.length > 0) {
      const cell = queue.shift();
      const key = JSON.stringify(cell);
      for (const [dx, dy] of steps) {
        const neighbour = [cell[0] + dx, cell[1] + dy];
        const neighbourKey = JSON.stringify(neighbour);
        if (parents.has(neighbourKey) || !this.inBounds(neighbour) || this.obstacles.has(neighbourKey)) {
          continue;
        }
        parents.set(neighbourKey, cell);
        queue.push(neighbour);
      }
    }
    return parents;
  }

  nearObstacle(cell, radius = 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      for (let dy = -radius; dy <= radius; dy += 1) {
        if (dx === 0 && dy === 0) continue;
        if (this.obstacles.has(JSON.stringify([cell[0] + dx, cell[1] + dy]))) {
          return true;
        }
      }
    }
    return false;
  }

  nextGoal(current, prefer = null) {
    let goal = this._nearestUnvisited(current, prefer);
    if (goal === null && this.visited.size > 0) {
      this.visited = new Set();
      this.laps += 1;
      goal = this._nearestUnvisited(current, prefer);
    }
    return goal;
  }

  _nearestUnvisited(current, prefer = null) {
    let fallback = null;
    for (const [key, parent] of this._search(current, prefer).entries()) {
      const cell = JSON.parse(key);
      if (this.visited.has(key) || this.obstacles.has(key)) continue;
      if (!this.nearObstacle(cell)) return cell;
      if (fallback === null) fallback = cell;
    }
    return fallback;
  }

  _cheapestSearch(start, cost) {
    const parents = new Map([[JSON.stringify(start), null]]);
    const best = new Map([[JSON.stringify(start), 0]]);
    const heap = [{ total: 0, order: 0, cell: start }];
    let counter = 0;
    const done = new Set();

    while (heap.length > 0) {
      heap.sort((a, b) => a.total - b.total || a.order - b.order);
      const current = heap.shift();
      const key = JSON.stringify(current.cell);
      if (done.has(key)) continue;
      done.add(key);

      for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
        const neighbour = [current.cell[0] + dx, current.cell[1] + dy];
        const neighbourKey = JSON.stringify(neighbour);
        if (done.has(neighbourKey) || !this.inBounds(neighbour) || this.obstacles.has(neighbourKey)) continue;
        const newTotal = current.total + Math.max(0.01, cost(neighbour));
        if (newTotal < (best.get(neighbourKey) ?? Number.POSITIVE_INFINITY)) {
          best.set(neighbourKey, newTotal);
          parents.set(neighbourKey, current.cell);
          counter += 1;
          heap.push({ total: newTotal, order: counter, cell: neighbour });
        }
      }
    }

    return parents;
  }

  path(start, goal, cost = null) {
    const parents = cost === null ? this._search(start) : this._cheapestSearch(start, cost);
    const goalKey = JSON.stringify(goal);
    if (!parents.has(goalKey)) return null;

    const steps = [];
    let cell = goal;
    while (cell !== null && JSON.stringify(cell) !== JSON.stringify(start)) {
      steps.push(cell);
      cell = parents.get(JSON.stringify(cell));
    }
    steps.reverse();
    return steps;
  }
}
