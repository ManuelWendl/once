import functools
from typing import Any, Callable, Mapping

from ss2r.benchmark_suites.mujoco_cpu.cartpole import CartpoleSwingupCPU, RealEnv
from ss2r.benchmark_suites.mujoco_cpu.go_to_goal import GoToGoalCPU
from ss2r.benchmark_suites.mujoco_cpu.humanoid import HumanoidWalkCPU
from ss2r.benchmark_suites.mujoco_cpu.walker import WalkerWalkCPU
from ss2r.benchmark_suites.mujoco_playground.cartpole import (
    recovery as cartpole_recovery,
)
from ss2r.benchmark_suites.mujoco_playground.humanoid import (
    recovery as humanoid_recovery,
)
from ss2r.benchmark_suites.mujoco_playground.walker import (
    recovery as walker_recovery,
)
from ss2r.benchmark_suites.safety_gym import recovery as go_to_goal_recovery

_envs = {
    "CartpoleSwingupCPU": CartpoleSwingupCPU,
    "GoToGoalCPU": GoToGoalCPU,
    "HumanoidWalkCPU": HumanoidWalkCPU,
    "WalkerWalkCPU": WalkerWalkCPU,
}


def make(environment_cfg: Mapping[str, Any], action_repeat: int, seed: int) -> RealEnv:
    """Creates the CPU environment described by an `environment` config."""
    task_name = environment_cfg["task_name"]
    if task_name not in _envs:
        raise ValueError(f"Unknown CPU environment {task_name}")
    return _envs[task_name](
        **environment_cfg["task_params"],
        real_params=environment_cfg.get("real_params"),
        action_repeat=action_repeat,
        seed=seed,
    )


def get_recovery_fns(environment_cfg: Mapping[str, Any]) -> tuple[Callable, Callable]:
    """The recovery reward and the constraint violation of an observation."""
    task_name = environment_cfg["task_name"]
    task_params = environment_cfg["task_params"]
    if task_name == "CartpoleSwingupCPU":
        recovery_reward_fn = functools.partial(
            cartpole_recovery.recovery_reward_fn,
            position_bound=task_params["recovery_position_bound"],
            velocity_bound=task_params["recovery_velocity_bound"],
            hard_position_bound=task_params["hard_position_bound"],
        )
        violates_fn = functools.partial(
            cartpole_recovery.violates,
            hard_position_bound=task_params["hard_position_bound"],
        )
    elif task_name == "GoToGoalCPU":
        recovery_reward_fn = functools.partial(
            go_to_goal_recovery.recovery_reward_fn,
            recovery_distance=task_params["recovery_distance"],
            recovery_speed=task_params["recovery_speed"],
            hard_distance=task_params["hard_distance"],
            max_distance=task_params.get("max_distance"),
            recovery_max_distance=task_params.get("recovery_max_distance"),
        )
        violates_fn = functools.partial(
            go_to_goal_recovery.violates,
            hard_distance=task_params["hard_distance"],
            max_distance=task_params.get("max_distance"),
        )
    elif task_name == "WalkerWalkCPU":
        recovery_reward_fn = functools.partial(
            walker_recovery.recovery_reward_fn,
            recovery_upright=task_params["recovery_upright"],
            recovery_pitch_rate=task_params["recovery_pitch_rate"],
            hard_upright=task_params["hard_upright"],
            recovery_leg_upright=task_params["recovery_leg_upright"],
        )
        violates_fn = functools.partial(
            walker_recovery.violates,
            hard_upright=task_params["hard_upright"],
        )
    elif task_name == "HumanoidWalkCPU":
        hard_bounds = dict(
            hard_upright=task_params["hard_upright"],
            hard_head_height=task_params["hard_head_height"],
        )
        recovery_reward_fn = functools.partial(
            humanoid_recovery.recovery_reward_fn,
            recovery_upright=task_params["recovery_upright"],
            recovery_head_height=task_params["recovery_head_height"],
            recovery_tilt_rate=task_params["recovery_tilt_rate"],
            **hard_bounds,
        )
        violates_fn = functools.partial(humanoid_recovery.violates, **hard_bounds)
    else:
        raise ValueError(f"Unknown CPU environment {task_name}")
    return recovery_reward_fn, violates_fn
