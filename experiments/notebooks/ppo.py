"""Run the PPO notebook as a script.

Mirrors `notebooks/ppo.ipynb`: same circuit, same seeds, same configuration,
same figures. Only the surrounding prose and the live viewer are left out,
because neither survives a headless run.
"""

from __future__ import annotations

import sys
from dataclasses import replace

from _shared import ENVIRONMENT_CONFIG, RunContext, run_notebook

from agents import PPOAgent
from configs import FIXED_CRITIC_CONFIG, MEDIUM_ACTOR_CONFIG, PPOConfig
from training import PPOTrainingEngine
from utils.random import SeedStream

# Rates selected by the pre-experiment calibration in docs/EXPERIMENT.md.
ACTOR_CONFIG = replace(MEDIUM_ACTOR_CONFIG, learning_rate=3e-4)
CRITIC_CONFIG = replace(FIXED_CRITIC_CONFIG, learning_rate=1e-2)
PPO_CONFIG = PPOConfig(
    gae_lambda=0.95,
    transitions_per_rollout=2_048,
    optimization_epochs=4,
    minibatch_size=64,
    clip_epsilon=0.2,
)


def build(context: RunContext) -> tuple[PPOAgent, PPOTrainingEngine]:
    """
    Construct the clipped agent and the engine that fills the rollout it reuses.
    """
    agent = PPOAgent(
        observation_dimensions=context.observation_dimensions,
        actor_config=ACTOR_CONFIG,
        critic_config=CRITIC_CONFIG,
        config=PPO_CONFIG,
        actor_initialization_generator=context.streams.get_torch_generator(
            SeedStream.ACTOR_INITIALIZATION,
        ),
        critic_initialization_generator=context.streams.get_torch_generator(
            SeedStream.CRITIC_INITIALIZATION,
        ),
        sampling_generator=context.torch_generators(SeedStream.POLICY_ACTION_SAMPLING),
        optimization_generator=context.streams.get_torch_generator(
            SeedStream.OPTIMIZATION_BATCH_ORDER,
        ),
    )
    engine = PPOTrainingEngine(
        agent,
        context.reference_track,
        ENVIRONMENT_CONFIG,
        context.normalizer,
        **context.engine_arguments(),
    )
    return agent, engine


if __name__ == "__main__":
    sys.exit(run_notebook("PPO", build, include_critic=True))
