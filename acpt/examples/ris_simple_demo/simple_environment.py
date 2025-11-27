"""Utility helpers for running a small stand-alone RIS environment demo."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from acpt.core.interfaces.environment_interface import Transition
from acpt.environments.ris_environment import RISEnvironment

DEFAULT_STEPS = 5


def build_simple_ris_environment() -> RISEnvironment:
    """Return a lightly configured :class:`RISEnvironment` for quick demos."""

    return RISEnvironment(
        tile_count=32,
        user_count=2,
        corridor_length=40.0,
        corridor_width=12.0,
        tx_power_dbm=28.0,
        noise_floor_dbm=-92.0,
    )


def _snapshot(transition: Transition, step: int) -> Dict[str, Any]:
    """Convert a :class:`Transition` into a serialisable record."""

    state = transition.state.get("agent.ris", {})
    reward = transition.reward.get("agent.ris", 0.0)
    return {
        "step": step,
        "state": copy.deepcopy(state),
        "reward": reward,
    }


def run_simple_ris_episode(steps: int = DEFAULT_STEPS) -> List[Dict[str, Any]]:
    """Simulate a short RIS-only episode and return state history."""

    env = build_simple_ris_environment()
    transition = env.reset(seed=123)
    history: List[Dict[str, Any]] = [_snapshot(transition, step=0)]

    phase_delta = 0.0
    for step in range(1, steps + 1):
        phase_delta = (phase_delta + 0.35) % 3.14
        action = {"agent.ris": {"ris_phase_update": {"phase": phase_delta}}}
        transition = env.step(action)
        history.append(_snapshot(transition, step=step))

    return history
