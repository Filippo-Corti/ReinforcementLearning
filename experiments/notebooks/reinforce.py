"""Run the REINFORCE notebook as a script.

Mirrors `notebooks/reinforce.ipynb`: same circuit, same seeds, same
configuration, same figures. Only the surrounding prose and the live viewer are
left out, because neither survives a headless run.
"""

from __future__ import annotations

import sys
from dataclasses import replace

from _shared import ENVIRONMENT_CONFIG, RunContext, run_notebook

from agents import ReinforceAgent
from configs import MEDIUM_ACTOR_CONFIG, ReinforceConfig
from training import ReinforceTrainingEngine
from utils.random import SeedStream

# Rates selected by the pre-experiment calibration in docs/EXPERIMENT.md.
ACTOR_CONFIG = replace(MEDIUM_ACTOR_CONFIG, learning_rate=1e-3)
REINFORCE_CONFIG = ReinforceConfig(
    completed_episodes_per_update=8,
)


def build(context: RunContext) -> tuple[ReinforceAgent, ReinforceTrainingEngine]:
    """
    Construct the actor-only agent and the engine that collects episodes for it.
    """
    agent = ReinforceAgent(
        observation_dimensions=context.observation_dimensions,
        actor_config=ACTOR_CONFIG,
        config=REINFORCE_CONFIG,
        initialization_generator=context.streams.get_torch_generator(
            SeedStream.ACTOR_INITIALIZATION,
        ),
        sampling_generator=context.torch_generators(SeedStream.POLICY_ACTION_SAMPLING),
    )
    engine = ReinforceTrainingEngine(
        agent,
        context.reference_track,
        ENVIRONMENT_CONFIG,
        context.normalizer,
        **context.engine_arguments(),
    )
    return agent, engine


if __name__ == "__main__":
    sys.exit(run_notebook("REINFORCE", build, include_critic=False))
