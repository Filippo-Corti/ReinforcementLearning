# DIARY

## 2026-07-28 — Init
**Task**: Finalization of the project's fundamentals.

**Result**: The `AGENTS.md`, `MDP.md`, `TRACK.md` and `DIARY.md` files.

**Commit**: aab80ff

## 2026-07-28 — Specification audit follow-up
**Task**: Resolve the project-readiness findings selected after a repository-wide audit.

**Result**: Reconciled the reward arithmetic and long-horizon discount; clarified the Markov-like Frenet observation and dynamic curvature preview; documented version 0 physics limitations; specified deterministic track generation, storage, projection, LiDAR, signed progress and lap completion; prohibited implicit reuse of lab code; clarified incremental dependency management; and expanded repository ignore rules.

**Provisional choices**: $\gamma=0.9995$, $w=12m$, $\Delta s_{\text{gen}}=0.5m$, 16 inclusive LiDAR rays over $200°$, and $r_{\max}=100m$. These values are now explicit and may be revised before implementation.

**Files**: `.gitignore`, `AGENTS.md`, `README.md`, `docs/MDP.md`, `docs/TRACK.md`, `docs/DIARY.md`.

**Commit**: 6c239f1

## 2026-07-29 — Environment MVP execution roadmap

**Task**: Expand the initial `PLAN.md` into a practical sequence for building and validating the first racing-environment version.

**Result**: Defined the Phase-1 scope, acceptance criteria, execution rules and 12 sequential implementation milestones. Clarified that manual A/D input controls steering angle, added reward and Gymnasium lifecycle work to the environment scope, deferred grip-limited dynamics and LiDAR, and aligned the suggested flat source layout under `src/`.

**Files**: `PLAN.md`, `AGENTS.md`, `README.md`, `docs/DIARY.md`.

**Commit**: 909a83c

## 2026-07-29 — Flat source layout correction

**Task**: Remove the unwanted intermediate package directory from the proposed source structure.

**Result**: Restored `src/envs`, `src/models`, `src/agents`, `src/utils` and `src/configs` as the direct source directories, and updated every roadmap path and descriptive reference accordingly.

**Files**: `PLAN.md`, `AGENTS.md`, `docs/DIARY.md`.

**Commit**: bedfa4b

## 2026-07-29 — Step 0: Python project scaffold

**Task**: Establish the flat-source Python package structure, dependency manifest, packaging configuration and initial test command.

**Result**: Added importable `src/envs` and `src/configs` packages, configured setuptools and pytest in `pyproject.toml`, pinned the approved Phase-1 dependencies in `requirements.txt`, installed the project in editable mode, and added import smoke tests.

**Dependencies**: Gymnasium 1.3.0, NumPy 2.4.4, Pygame 2.6.1, pytest 9.1.1 and Setuptools 80.10.2. Setuptools remains below version 81 because Pygame 2.6.1 still imports the deprecated `pkg_resources` API.

**Validation**: `pip check` reported no broken requirements; all 5 pytest cases passed; and `envs` and `configs` imported successfully from outside the repository.

**Files**: `PLAN.md`, `pyproject.toml`, `requirements.txt`, `src/envs/__init__.py`, `src/configs/__init__.py`, `tests/test_package.py`, `docs/DIARY.md`.

**Commit**: 4d89e3f

*New diary notes go here...*

