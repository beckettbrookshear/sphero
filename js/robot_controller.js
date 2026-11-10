import { PatrolPlanner } from './patrol_planner.js';
import { RiskModel } from './risk_model.js';

const DIRECTION_HEADINGS = {
  forward: { speed: 60, heading: 0 },
  backward: { speed: 60, heading: 180 },
  left: { speed: 60, heading: 90 },
  right: { speed: 60, heading: 270 },
  stop: { speed: 0, heading: 0 },
};

export const commandToAction = (command, speed = 60) => {
  const normalized = String(command).toLowerCase().trim();
  if (!(normalized in DIRECTION_HEADINGS)) {
    throw new Error(`Unsupported command: ${command}`);
  }

  const action = { ...DIRECTION_HEADINGS[normalized] };
  if (normalized !== 'stop') {
    action.speed = Math.max(0, Math.min(255, speed));
  }
  return action;
};

export class RobotController {
  constructor(riskModelPath = null) {
    this._riskModel = new RiskModel(riskModelPath);
    this._planner = new PatrolPlanner(5, 30, 12);
    this.toy = null;
    this.heading = 0;
    this.autoMode = false;
    this.autoStopReason = null;
  }

  connect() {
    this.toy = { name: 'JS Virtual Robot' };
    return { connected: true, robot: this.toy.name };
  }

  disconnect() {
    this.toy = null;
    this.autoMode = false;
    return { connected: false };
  }

  drive(command, speed = 60, turnStep = 15, turnRate = null) {
    const action = commandToAction(command, speed);
    this.heading = action.heading;
    return action;
  }

  autoMap() {
    return {
      risk: {
        samples: this._riskModel.samples,
        hits: this._riskModel.hits,
        cells: this._riskModel.riskMap(this._planner.halfCells),
      },
    };
  }

  setAutoMode(enabled) {
    this.autoMode = Boolean(enabled);
    this.autoStopReason = this.autoMode ? 'auto mode enabled' : 'auto mode disabled';
    return { enabled: this.autoMode, status: this.autoStopReason };
  }
}
