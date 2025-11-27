"""Composite environment fusing RIS, NOMA, and V2I simulators under one contract."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from acpt.core.interfaces.environment_interface import (
    EnvironmentInterface,
    NakagamiFadingModel,
    RandomWalkMobility,
    RayleighFadingModel,
    RicianFadingModel,
    Transition,
)


class MultiDomainEnvironment(EnvironmentInterface):
    """Aggregate multiple ACP environments into a single Transition-emitting wrapper."""

    _DEFAULT_DELEGATES: Tuple[Mapping[str, Any], ...] = (
        {
            "name": "ris",
            "module": "acpt.environments.ris_environment",
            "class": "RISEnvironment",
            "agents": ["agent.ris"],
        },
        {
            "name": "noma",
            "module": "acpt.environments.noma_environment",
            "class": "NOMAEnvironment",
            "agents": ["agent.noma"],
        },
        {
            "name": "v2i",
            "module": "acpt.environments.v2i_environment",
            "class": "V2IEnvironment",
            "agents": ["agent.v2i"],
        },
    )

    def __init__(self, *, delegates: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        super().__init__()
        specs = list(delegates or self._DEFAULT_DELEGATES)
        if not specs:
            specs = list(self._DEFAULT_DELEGATES)

        self._delegates: List[Dict[str, Any]] = []
        for spec in specs:
            module = self._import(spec.get("module"))
            cls = getattr(module, spec.get("class"))
            args: Sequence[Any] = tuple(spec.get("args", []))
            kwargs: Dict[str, Any] = dict(spec.get("kwargs", {}))
            instance = cls(*args, **kwargs)
            self._configure_delegate(instance, spec)
            self._delegates.append(
                {
                    "name": spec.get("name", cls.__name__),
                    "agents": list(spec.get("agents", [])),
                    "seed": spec.get("seed"),
                    "env": instance,
                }
            )

        self._last_state: Dict[str, Any] = {}
        self._last_reward: Dict[str, float] = {}
        self._last_done = False
        self._last_info: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None) -> Transition:  # noqa: D401
        self._reset_time(seed)
        return self._run_transition("reset", {}, seed)

    def step(self, action_dict: Mapping[str, Mapping[str, Any]]) -> Transition:  # noqa: D401
        self._increment_time()
        return self._run_transition("step", action_dict, None)

    def observe(self) -> Dict[str, Any]:  # noqa: D401
        return dict(self._last_state)

    def reward(self) -> Dict[str, float]:  # noqa: D401
        return dict(self._last_reward)

    # ------------------------------------------------------------------

    def _run_transition(
        self,
        method: str,
        action_dict: Mapping[str, Mapping[str, Any]],
        seed: Optional[int],
    ) -> Transition:
        combined_state: Dict[str, Any] = {}
        combined_reward: Dict[str, float] = {}
        combined_info: Dict[str, Any] = {}
        done_flags: List[bool] = []

        for delegate in self._delegates:
            env = delegate["env"]
            if method == "reset":
                delegate_seed = delegate.get("seed")
                resolved_seed = delegate_seed if delegate_seed is not None else seed
                transition: Transition = env.reset(seed=resolved_seed)
            else:
                actions = self._subset_actions(delegate.get("agents", []), action_dict)
                transition = env.step(actions)

            self._update_agent_mapping(delegate, transition.state.keys())
            combined_state.update({key: dict(value) for key, value in transition.state.items()})
            combined_reward.update({key: float(value) for key, value in transition.reward.items()})
            combined_info[delegate["name"]] = dict(transition.info)
            done_flags.append(bool(transition.done))

        done = any(done_flags)
        self._last_state = combined_state
        self._last_reward = combined_reward
        self._last_done = done
        self._last_info = combined_info

        return Transition(
            dict(combined_state),
            dict(combined_reward),
            done,
            self._info(),
        )

    def _subset_actions(
        self,
        agent_ids: Sequence[str],
        action_dict: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Mapping[str, Any]]:
        if not agent_ids:
            return {}
        return {agent_id: action_dict.get(agent_id, {}) for agent_id in agent_ids}

    @staticmethod
    def _update_agent_mapping(delegate: MutableMapping[str, Any], keys: Iterable[str]) -> None:
        agent_ids = list(keys or [])
        if agent_ids and not delegate.get("agents"):
            delegate["agents"] = agent_ids

    def _info(self) -> Dict[str, Any]:
        return {
            "time_index": self._time_index,
            "delegates": {name: dict(info) for name, info in self._last_info.items()},
        }

    def _configure_delegate(self, env: Any, spec: Mapping[str, Any]) -> None:
        fading_specs = spec.get("fading_models") or []
        if fading_specs and hasattr(env, "register_fading_model"):
            for entry in fading_specs:
                channel_id = entry.get("channel_id")
                if not channel_id:
                    raise ValueError("fading_models entries require 'channel_id'")
                model = self._build_component(entry, self._fading_registry())
                env.register_fading_model(channel_id, model)

        mobility_specs = spec.get("mobility_models") or []
        if mobility_specs and hasattr(env, "register_mobility_model"):
            for entry in mobility_specs:
                agent_id = entry.get("agent_id")
                if not agent_id:
                    raise ValueError("mobility_models entries require 'agent_id'")
                model = self._build_component(entry, self._mobility_registry())
                env.register_mobility_model(agent_id, model)

    @staticmethod
    def _build_component(entry: Mapping[str, Any], registry: Mapping[str, Any]) -> Any:
        if "instance" in entry:
            return entry["instance"]

        type_key = entry.get("type")
        kwargs = dict(entry.get("kwargs", {}))
        if type_key:
            lookup = registry.get(str(type_key).lower())
            if lookup is None:
                raise ValueError(f"Unknown component type '{type_key}'")
            return lookup(**kwargs)

        module_name = entry.get("module")
        class_name = entry.get("class")
        if not module_name or not class_name:
            raise ValueError("Component entries require either 'type' or 'module'/'class'")
        module = import_module(module_name)
        comp_cls = getattr(module, class_name)
        args = tuple(entry.get("args", []))
        return comp_cls(*args, **kwargs)

    @staticmethod
    def _fading_registry() -> Mapping[str, Any]:
        return {
            "rician": RicianFadingModel,
            "rayleigh": RayleighFadingModel,
            "nakagami": NakagamiFadingModel,
        }

    @staticmethod
    def _mobility_registry() -> Mapping[str, Any]:
        return {
            "random_walk": RandomWalkMobility,
        }

    @staticmethod
    def _import(module_name: Optional[str]):
        if not module_name:
            raise ValueError("Delegate configuration must specify a module.")
        return import_module(module_name)
