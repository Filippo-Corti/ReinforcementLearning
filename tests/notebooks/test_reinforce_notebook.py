from __future__ import annotations

import json
from pathlib import Path


def test_reinforce_notebook_uses_the_readable_engine_and_exposes_circuit_mode() -> None:
    notebook_path = Path(__file__).parents[2] / "notebooks" / "reinforce.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "from agents import ReinforceAgent" in code
    assert "from training.engines.reinforce import ReinforceTrainingEngine" in code
    assert "OnPolicyTrainingEngine" not in code
    assert "shared_engine" not in code
    assert "TRAIN_ON_MULTIPLE_CIRCUITS" in code
    assert "SINGLE_CIRCUIT_SEED" in code
    assert "MULTIPLE_CIRCUIT_SEEDS" in code
    assert "initial_rollout = show_rendered_rollout" in code
    assert "final_rollout = show_rendered_rollout" in code


def test_reinforce_notebook_code_cells_compile() -> None:
    notebook_path = Path(__file__).parents[2] / "notebooks" / "reinforce.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_reinforce_notebook_retains_records_and_rendered_outputs() -> None:
    notebook_path = Path(__file__).parents[2] / "notebooks" / "reinforce.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells_by_id = {cell["id"]: cell for cell in notebook["cells"]}

    assert cells_by_id["render-initial-policy"]["outputs"]
    assert cells_by_id["train-agent"]["outputs"]
    assert cells_by_id["show-records"]["outputs"]
    assert cells_by_id["plot-records"]["outputs"]
    assert cells_by_id["render-final-policy"]["outputs"]
