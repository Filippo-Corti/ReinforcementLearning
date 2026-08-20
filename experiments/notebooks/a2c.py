"""Run the A2C notebook as a script.

Mirrors `notebooks/a2c.ipynb`: same circuit, same seeds, same configuration,
same figures. Only the surrounding prose and the live viewer are left out,
because neither survives a headless run.
"""

from __future__ import annotations

import sys
from dataclasses import replace

from _shared import ENVIRONMENT_CONFIG, RunContext, run_notebook

from agents import A2CAgent
from configs import FIXED_CRITIC_CONFIG, MEDIUM_ACTOR_CONFIG, A2CConfig
from training import A2CTrainingEngine
from utils.random import SeedStream

# Rates selected by the pre-experiment calibration in docs/EXPERIMENT.md.
ACTOR_CONFIG = replace(MEDIUM_ACTOR_CONFIG, learning_rate=3e-4)
CRITIC_CONFIG = replace(FIXED_CRITIC_CONFIG, learning_rate=1e-2)
A2C_CONFIG = A2CConfig(
    discount=0.9995,
    gae_lambda=0.95,
    transitions_per_rollout=2_048,
)


def build(context: RunContext) -> tuple[A2CAgent, A2CTrainingEngine]:
    """
    Construct the actor-critic agent and the engine that fills its rollout.
    """
    agent = A2CAgent(
        observation_dimensions=context.observation_dimensions,
        actor_config=ACTOR_CONFIG,
        critic_config=CRITIC_CONFIG,
        config=A2C_CONFIG,
        actor_initialization_generator=context.streams.get_torch_generator(
            SeedStream.ACTOR_INITIALIZATION,
        ),
        critic_initialization_generator=context.streams.get_torch_generator(
            SeedStream.CRITIC_INITIALIZATION,
        ),
        sampling_generator=context.torch_generators(SeedStream.POLICY_ACTION_SAMPLING),
    )
    engine = A2CTrainingEngine(
        agent,
        context.reference_track,
        ENVIRONMENT_CONFIG,
        context.normalizer,
        **context.engine_arguments(),
    )
    return agent, engine


if __name__ == "__main__":
    sys.exit(run_notebook("A2C", build, include_critic=True))
