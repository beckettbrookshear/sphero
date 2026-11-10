import test from 'node:test';
import assert from 'node:assert/strict';

import { PRIOR_WEIGHTS, RiskModel } from './risk_model.js';
import { PatrolPlanner } from './patrol_planner.js';

test('cells next to a bump look riskier than far away cells', () => {
  const model = new RiskModel();
  model.startRun(3);
  const far = model.predict([-2, -2]);
  model.recordBump([1, 1]);
  assert.ok(model.predict([1, 1]) > far);
  assert.ok(model.predict([1, 2]) > far);
  assert.ok(model.predict([2, 2]) > far);
  assert.ok(model.predict([1, 2]) > model.predict([-2, -2]));
});

test('model learns from experience', () => {
  const model = new RiskModel();
  model.startRun(3);
  for (let i = 0; i < 60; i += 1) {
    model.recordBump([3, 0]);
    model.bumpCounts = {};
    model.recordSafePass([0, 0]);
    model.recordSafePass([1, 0]);
  }
  assert.notDeepEqual(model.weights, PRIOR_WEIGHTS);
  assert.ok(model.predict([-3, 1]) > 3 * model.predict([0, 1]));
  assert.equal(model.samples, 180);
  assert.equal(model.hits, 60);
});

test('each run starts with no bump counts but keeps the learning', () => {
  const model = new RiskModel();
  model.startRun(3);
  model.recordBump([1, 0]);
  const learned = [...model.weights];
  model.startRun(3);
  assert.equal(Object.keys(model.bumpCounts).length, 0);
  assert.deepEqual(model.weights, learned);
});

test('planner routes around a risky cell but still reaches everything', () => {
  const planner = new PatrolPlanner(1);
  assert.deepEqual(planner.path([-1, 0], [1, 0]), [[0, 0], [1, 0]]);

  const cost = (cell) => (cell[0] === 0 && cell[1] === 0 ? 50 : 1);
  const detour = planner.path([-1, 0], [1, 0], cost);
  assert.ok(!detour.some(([x, y]) => x === 0 && y === 0));
  assert.deepEqual(detour[detour.length - 1], [1, 0]);
  assert.equal(detour.length, 4);

  planner.obstacles.add(JSON.stringify([0, 1]));
  planner.obstacles.add(JSON.stringify([0, -1]));
  planner.obstacles.add(JSON.stringify([0, 0]));
  assert.equal(planner.path([-1, 0], [1, 0], cost), null);
});
