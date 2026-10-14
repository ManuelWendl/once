"""Evaluation of trained CyberRunner policies on fixed simulated conditions, and ONNX export."""

import json
import pathlib
from typing import Any, Callable, Dict, List, Mapping, Tuple

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from brax.training.networks import normalizer_select
from omegaconf import OmegaConf

from ss2r.algorithms.ppo import networks as ppo_networks
from ss2r.benchmark_suites.mujoco_playground.cyberrunner import cyberrunner, randomize
from ss2r.benchmark_suites.wrappers import BraxDomainRandomizationVmapWrapper

# Randomization groups by config key; "delay" is the command delay (delay_ms).
GROUPS: Dict[str, Tuple[str, ...]] = {
    "motor": ("motor_gain", "motor_time_constant", "motor_speed_limit"),
    "joints": ("joint_damping", "joint_friction"),
    "ball_friction": ("ball_friction",),
    "ball_mass": ("ball_mass",),
    "ball_contact": ("ball_contact_damping",),
    "ball_stiction": ("ball_static_deg", "ball_rolling_deg"),
    "tilt": ("gravity_tilt_deg",),
    "delay": (),
}
TRAINING_CONFIGS = ("nominal", "full", *(f"no_{group}" for group in GROUPS))
# Task options that define the simulated system; a run is evaluated with its own.
MODEL_KEYS = ("layout", "ball_stiction", "ctrl_limit", "motor_dc")


def run_model(run_dir) -> Dict[str, Any]:
    """The run's MODEL_KEYS options (defaults for keys older runs lack)."""
    cfg = OmegaConf.load(pathlib.Path(run_dir) / ".hydra" / "config.yaml")
    task = cfg.environment.task_params
    if task.get("motor_model", "torque") != "torque":
        raise ValueError(f"{run_dir}: the servo motor models were removed")
    if task.get("motor_backemf", False):
        raise ValueError(f"{run_dir}: the back-EMF motor option was removed")
    defaults = cyberrunner.default_config()
    plain = OmegaConf.to_container(task)
    return {key: plain.get(key, defaults[key]) for key in MODEL_KEYS}


def _off(key: str) -> List[float]:
    additive = ("gravity_tilt_deg", "ball_static_deg", "ball_rolling_deg")
    return [0.0, 0.0] if key in additive else [1.0, 1.0]


def eval_conditions(
    full_params: Mapping[str, Any], delay_ms: List[float]
) -> Dict[str, Tuple[Dict[str, Any], List[float]]]:
    """Nominal, full randomization, and each group alone: (ranges, delay range)."""
    nominal = {key: _off(key) for key in full_params}
    no_delay = [0.0, 0.0]
    conditions = {
        "nominal": (nominal, no_delay),
        "full": (dict(full_params), list(delay_ms)),
    }
    for group, keys in GROUPS.items():
        params = {**nominal, **{key: full_params[key] for key in keys}}
        conditions[group] = (params, list(delay_ms) if group == "delay" else no_delay)
    return conditions


def training_overrides(name: str) -> List[str]:
    """Hydra overrides of cyberrunner_ppo for a training config."""
    if name == "full":
        return []
    no_delay = "environment.task_params.delay_ms=[0.0,0.0]"
    if name == "nominal":
        return ["training.train_domain_randomization=false", no_delay]
    group = name.removeprefix("no_")
    if group == "delay":
        return [no_delay]
    return [
        f"environment.train_params.{key}=[{_off(key)[0]},{_off(key)[1]}]"
        for key in GROUPS[group]
    ]


def last_checkpoint(run_dir) -> pathlib.Path:
    return sorted(p for p in (pathlib.Path(run_dir) / "ckpt").iterdir() if p.is_dir())[
        -1
    ]


def best_checkpoint(run_dir) -> pathlib.Path:
    """Checkpoint with the highest in-training eval success (then path progress)."""
    run_dir = pathlib.Path(run_dir)
    rows = [
        json.loads(line)
        for line in (run_dir / "summary.jsonl").read_text().splitlines()
    ]
    candidates = [r for r in rows if (run_dir / "ckpt" / f"{r['step']:012d}").is_dir()]
    best = max(
        candidates,
        key=lambda r: (r["eval/episode_success"], r["eval/episode_progress"]),
    )
    return run_dir / "ckpt" / f"{best['step']:012d}"


def load_policy_params(run_dir, ckpt=None) -> Tuple[Callable, Any]:
    """(make_policy, params) of a PPO run's checkpoint (default: the last)."""
    run_dir = pathlib.Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    agent, task = cfg.agent, cfg.environment.task_params
    normalizer, network_params, *_ = checkpoint.load(ckpt or last_checkpoint(run_dir))
    if isinstance(normalizer.mean, Mapping):  # privileged critic: the actor sees state
        normalizer = normalizer_select(normalizer, "state")
    networks = ppo_networks.make_ppo_networks(
        task.history_length * cyberrunner.FRAME_SIZE + 2 * task.num_path_points,
        2,
        preprocess_observations_fn=running_statistics.normalize
        if agent.normalize_observations
        else lambda x, y: x,
        policy_hidden_layer_sizes=tuple(agent.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(agent.value_hidden_layer_sizes),
        activation=getattr(jax.nn, agent.activation),
    )
    inference = ppo_networks.make_inference_fn(networks)
    make_policy = lambda params: inference(params, deterministic=True)
    return make_policy, (normalizer, network_params["policy"])


def load_policy(run_dir) -> Callable:
    make_policy, params = load_policy_params(run_dir)
    return make_policy(params)


class Evaluator:
    """Deterministic episodes from the start of the maze under one condition."""

    def __init__(
        self,
        make_policy: Callable,
        params: Mapping[str, Any],
        delay_ms: List[float],
        num_episodes: int = 1024,
        episode_length: int = 3000,
        impl: str = "warp",
        seed: int = 0,
        task: Mapping[str, Any] = {},
    ):
        cfg = cyberrunner.default_config()
        for key, value in task.items():  # e.g. run_model(run_dir)
            cfg[key] = value
        cfg.impl = impl
        cfg.num_envs_hint = num_episodes
        cfg.random_start = False
        cfg.episode_length = episode_length
        cfg.delay_ms = list(delay_ms)
        env: Any = cyberrunner.CyberRunner(cfg)
        key_physics, key_reset, key_policy = jax.random.split(
            jax.random.PRNGKey(seed), 3
        )
        ranges = OmegaConf.create(dict(params))
        physics_keys = jax.random.split(key_physics, num_episodes)
        env = BraxDomainRandomizationVmapWrapper(
            env,
            lambda sys: randomize.domain_randomization(sys, physics_keys, ranges),
            augment_state=False,
        )
        reset_keys = jax.random.split(key_reset, num_episodes)
        policy_keys = jax.random.split(key_policy, episode_length)

        def rollout(policy_params):
            policy = make_policy(policy_params)
            state = env.reset(reset_keys)
            zeros = jp.zeros(num_episodes)

            def step(carry, key):
                state, active, success, progress, length = carry
                action, _ = policy(state.obs, key)
                state = env.step(state, action)
                success = success + active * state.metrics["success"]
                progress = progress + active * state.metrics["progress"]
                length = length + active
                active = active * (1.0 - state.done)
                return (state, active, success, progress, length), None

            carry = (state, jp.ones(num_episodes), zeros, zeros, zeros)
            carry, _ = jax.lax.scan(step, carry, policy_keys)
            return carry[1:]

        self._rollout = jax.jit(rollout)

    def __call__(self, policy_params) -> Dict[str, np.ndarray]:
        active, success, progress, length = (
            np.asarray(x) for x in self._rollout(policy_params)
        )
        outcome = np.where(
            success > 0, "success", np.where(active > 0, "timeout", "fell")
        )
        return {"outcome": outcome, "progress": progress, "length": length.astype(int)}


def evaluate(
    policy: Callable, params: Mapping[str, Any], delay_ms: List[float], **kwargs
) -> Dict[str, np.ndarray]:
    """Evaluate one fixed policy function under one condition."""
    return Evaluator(lambda _: policy, params, delay_ms, **kwargs)(0)


# ONNX export for the robot's ONNX runner (needs the onnx dependency group).

ACTION_SIZE = 2
_ACTIVATIONS = {"relu": "Relu", "tanh": "Tanh", "elu": "Elu", "sigmoid": "Sigmoid"}


def _activation(name: str, x: str, out: str) -> List[Any]:
    from onnx import helper

    if name in ("swish", "silu"):
        return [
            helper.make_node("Sigmoid", [x], [f"{out}_sigmoid"]),
            helper.make_node("Mul", [x, f"{out}_sigmoid"], [out]),
        ]
    return [helper.make_node(_ACTIVATIONS[name], [x], [out])]


def _tensor(value: Any, name: str) -> Any:
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)


def policy_graph(
    normalizer: Any, policy_params: Any, activation: str, normalize: bool
) -> Any:
    """observation [obs] -> normalizer -> MLP -> tanh(mean) = action [2]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    layers = policy_params["params"]
    dense = [layers[f"hidden_{i}"] for i in range(len(layers))]
    obs_size = dense[0]["kernel"].shape[0]
    nodes: List[Any] = []
    inits: List[Any] = []
    x = "observation"
    if normalize:
        inits += [
            _tensor(normalizer.mean, "obs_mean"),
            _tensor(normalizer.std, "obs_std"),
        ]
        nodes += [
            helper.make_node("Sub", [x, "obs_mean"], ["centered"]),
            helper.make_node("Div", ["centered", "obs_std"], ["normalized"]),
        ]
        x = "normalized"
    for i, layer in enumerate(dense):
        inits += [
            _tensor(layer["kernel"], f"kernel_{i}"),
            _tensor(layer["bias"], f"bias_{i}"),
        ]
        nodes += [
            helper.make_node("MatMul", [x, f"kernel_{i}"], [f"matmul_{i}"]),
            helper.make_node("Add", [f"matmul_{i}", f"bias_{i}"], [f"dense_{i}"]),
        ]
        x = f"dense_{i}"
        if i < len(dense) - 1:
            nodes += _activation(activation, x, f"act_{i}")
            x = f"act_{i}"
    # The last layer outputs the mean and scale of the action distribution;
    # the deterministic action is tanh(mean).
    inits += [
        numpy_helper.from_array(np.array([0]), "mean_start"),
        numpy_helper.from_array(np.array([ACTION_SIZE]), "mean_end"),
    ]
    nodes += [
        helper.make_node("Slice", [x, "mean_start", "mean_end"], ["mean"]),
        helper.make_node("Tanh", ["mean"], ["action"]),
    ]
    graph = helper.make_graph(
        nodes,
        "cyberrunner_policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [obs_size])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [ACTION_SIZE])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def export(run_dir, out_path, ckpt=None) -> pathlib.Path:
    """Write the ONNX policy of a run's checkpoint (default: the best one) and
    return that checkpoint."""
    import onnx

    run_dir = pathlib.Path(run_dir)
    agent = OmegaConf.load(run_dir / ".hydra" / "config.yaml").agent
    ckpt = ckpt or best_checkpoint(run_dir)
    _, (normalizer, policy_params) = load_policy_params(run_dir, ckpt)
    model = policy_graph(
        normalizer, policy_params, agent.activation, agent.normalize_observations
    )
    onnx.save(model, str(out_path))
    return ckpt
