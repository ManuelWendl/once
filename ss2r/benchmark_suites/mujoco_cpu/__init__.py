from typing import Any, Mapping

from ss2r.benchmark_suites.mujoco_cpu.cartpole import CartpoleSwingupCPU, RealEnv

_envs = {
    "CartpoleSwingupCPU": CartpoleSwingupCPU,
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
