# Sphero Bolt Robot Controller

This project turns the current workspace into a small Flask app that connects to a Sphero Bolt and lets you drive it from a browser.

Features:
- Connect to the nearest Sphero Bolt over Bluetooth
- Drive from the page or an Xbox controller (sticks for normal driving, D-pad for very fine control)
- Auto pilot: calibrates, patrols a grid, learns obstacles from bumps, and stops on its own if something is wrong
- Live map of the auto pilot run (swept floor, path, bumps, blocked cells)
- A small learning model (online logistic regression, no extra libraries) that predicts which cells are likely to cause bumps. The patrol planner routes around risky cells, and the learned weights are saved in `.learned/risk_model.json` so it keeps improving between runs. Delete that file to reset it.
- Emergency stop that always works, including from the controller (Y)
- LED text and pixel art

Quick start:

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Start the app:

```bash
python webapp/app.py
```

3. Open http://127.0.0.1:5001 in your browser.

4. Click "Connect Robot", then use the movement buttons.

Notes:
- Turn on the Sphero Bolt and keep it close to your computer.
- Bluetooth must be enabled on the machine running the app.
- If the device is not found, check that the robot is awake and in pairing range.

Controller:
- X: stop and lock. Y: emergency stop and lock. A, B, B: unlock. A (unlocked): start auto pilot. B: stop auto pilot.
- Left stick: forward/backward. Right stick: turn (45 degrees per second at full deflection).
- D-pad: slow creep and 10 degrees per second turns.

Tests:

```bash
python -m pytest -q
```

The auto pilot thresholds (bump detection, calibration, patrol speed) are untuned guesses and have only been verified against simulations.
