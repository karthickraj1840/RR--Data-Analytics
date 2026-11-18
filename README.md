# SupeRR Pressure Lab

A local cricket-analysis dashboard for the Rajasthan Royals SupeRR Selector Season 2 Round 1 challenge. It processes the supplied match data, calculates pressure-aware player metrics, and provides a shortlist workflow. Rankings are decision support, not an automatically selected final squad.

## Setup

Requirements:

- Python 3.13 (or a compatible Python version supported by the installed dependencies)
- Node.js 18 or newer, with npm
- The supplied data pack in `6af458274c1e8_data_pack\Data Pack` or the matching zip at the project root

From `D:\RR`, install dependencies:

```bat
py -3.13 -m venv backend\.venv
call backend\.venv\Scripts\activate
python -m pip install -r backend\requirements.txt
cd frontend
npm install
cd ..
```

The frontend launcher also runs `npm install` when `frontend\node_modules` is absent. The backend launcher expects its virtual environment and Python dependencies to have been installed first.

## How to run

For the first run, or after changing the raw data or analysis configuration, rebuild the processed outputs:

```bat
cd /d D:\RR\backend
call .venv\Scripts\activate
python -m pipeline.run_all
```

Then either:

- Double-click `start_all.bat` to open the backend and frontend in separate command windows and open `http://localhost:5173` after six seconds; or
- Double-click `start_backend.bat` and `start_frontend.bat` separately, then open `http://localhost:5173`.

The backend launcher frees port 8000 if another process is listening there. Both launchers remain open so their output and errors are visible; press a key when prompted to close each window.

Run the checks manually from their project folders:

```bat
cd /d D:\RR\backend
pytest -q
cd /d D:\RR\frontend
npm run build
```

## Architecture

- `config.yaml` holds pressure thresholds, competition and index weights, eligibility rules, and the role split.
- `6af458274c1e8_data_pack\Data Pack` contains the input CSVs. The pipeline cleans and validates the data, builds match state and out-of-fold win probabilities, derives leverage and pressure flags, then writes player metrics, selection, sensitivity, backtest, and chart outputs under `backend\data\processed`.
- `backend\pipeline` contains the data-processing and analysis stages. `backend\app` serves the processed data through a FastAPI API on port 8000; API routes are in `backend\app\api\routes.py`.
- `frontend\src` contains the React and Vite dashboard. It reads the local API and provides player comparisons, leaderboards, explanations, squad-fit templates, and shortlist export.
- `backend\data\external` contains templates for user-supplied player attributes and squad data. Shortlist edits are stored locally by the API.

## Metric definitions

### Match state and win probability

Each delivery is represented by the state immediately before it, including runs and wickets so far, legal balls elapsed, runs needed, balls left, current run rate, and required run rate. The win-probability model is a separate histogram gradient-boosting classifier for each innings. It uses five-fold `GroupKFold` cross-validation grouped by match, so each delivery's stored probability is out-of-fold relative to that match. The model uses only pre-delivery state features.

Win probability added (WPA) is the change from the before-delivery probability to the after-delivery probability. For batting metrics, positive WPA favors the batting side; for bowling metrics, the sign is reversed so a positive value favors the bowler. The pipeline checks that match-level signed WPA sums reconcile.

### Pressure deliveries

A pressure delivery is a legal ball matching at least one of these configured flags:

- **High leverage:** the delivery's leverage index is at or above the configured 85th percentile. Leverage is the delivery's absolute WPA relative to the global mean absolute WPA for its match/innings/legal-ball/wickets-before state bucket.
- **Hard chase:** second innings, required run rate at least 10, and no more than 60 balls left.
- **Death overs:** overs 16 through 20.
- **Collapse:** at least three wickets have fallen by over 10.
- **Tight finish:** second innings, no more than 30 balls left, and pre-delivery win probability from 0.25 through 0.75.

Knockout status is tracked as an additional flag but is not itself included in the composite pressure-delivery definition used for player metrics and backtests.

### Player metrics and selection score

- **Runs saved per ball** compares a batter's runs or a bowler's runs conceded with the average for the same innings, phase (overs 1-6, 7-15, or 16-20), and wickets-before bucket. Positive values are better for the player.
- **WPA per ball** is role-signed WPA divided by legal balls. Pressure values are shrunk toward a pooled prior to reduce small-sample extremes; overall values are also shrunk.
- **Consistency** is the Beta-prior-smoothed fraction of player event units with positive WPA. Batter units are match-innings samples with at least six pressure balls; bowler units are match-innings-over samples.
- **Pressure reliability index (PRI)** is the configured weighted sum of standardized pressure WPA per ball, runs saved per ball, and consistency (default weights 0.4, 0.4, and 0.2 respectively). Each role/flag index is computed against qualifying players for that role.
- **Eligibility** for a batter or bowler requires at least 80 pressure balls in that role and an IPL share of at least 30%. An all-rounder classification requires at least 100 pressure balls in both roles. Top-10 selection uses a configured six-batter/four-bowler role split; a player can occupy only one slot.
- **Selection score** combines role-specific standardized selected WPA per ball, pressure runs saved per ball, and pressure consistency using the weights reported by the pipeline. The selected WPA metric blends shrunk pressure and overall WPA per ball; role-specific pressure blend weights are configured in `config.yaml`.

All thresholds and weights are configurable; the values above describe the current project configuration.

## Backtest method

The backtest predicts future composite-pressure WPA per ball (shrunk), with bowling WPA sign-reversed. It uses season-based, match-separated temporal folds:

1. Train on 2022-2023 and validate on 2024.
2. Train on 2022-2024 and validate on 2025.
3. Train on 2022-2025 and evaluate the final 2026 holdout.

Situation baselines are fit on each training period only. Rolling validation reports Spearman rank correlation, precision at 10 against the highest-performing 20% of the test population, and mean realized WPA-per-ball uplift over the test population. It also reports a random precision baseline. The final 2026 holdout is not used to select weights. The configured weights remain the selection weights; grid-search candidates are retained for comparison.

## Limitations

- Historical outcomes and event-level data do not capture every factor relevant to future performance, including fitness, availability, role changes, tactical fit, or current player form.
- Pressure definitions, situation baselines, competition weights, shrinkage, and role eligibility affect the rankings; different choices can change the shortlist.
- Small samples remain uncertain despite shrinkage and confidence/sensitivity reports. Backtest results describe historical periods and are not guarantees of future performance.
- The pipeline excludes incomplete, revised, or otherwise unsupported match records as documented in its data-quality output.
- Player attribute and squad templates are placeholders until populated with trusted, appropriately licensed data.
- Review the metric definitions and evidence for each player; do not treat a score or shortlist as a final selection decision.

## Confidentiality and local-only use

The supplied data pack is treated as confidential project input. Keep it and generated outputs on this machine; do not upload, publish, or send them to external services. The app is intended for localhost-only use at `http://localhost:5173` with its API at `http://localhost:8000`. Do not expose these development servers to a network or deploy them publicly.
