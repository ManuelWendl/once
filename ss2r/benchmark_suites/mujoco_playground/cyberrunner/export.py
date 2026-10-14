"""ONNX export of CyberRunner PPO policies for the robot's ONNX runner."""

import pathlib
from typing import Any, List

import numpy as np
import onnx
from omegaconf import OmegaConf
from onnx import TensorProto, helper, numpy_helper

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import evaluation

ACTION_SIZE = 2
_ACTIVATIONS = {"relu": "Relu", "tanh": "Tanh", "elu": "Elu", "sigmoid": "Sigmoid"}


def _activation(name: str, x: str, out: str) -> List[onnx.NodeProto]:
    if name in ("swish", "silu"):
        return [
            helper.make_node("Sigmoid", [x], [f"{out}_sigmoid"]),
            helper.make_node("Mul", [x, f"{out}_sigmoid"], [out]),
        ]
    return [helper.make_node(_ACTIVATIONS[name], [x], [out])]


def _tensor(value: Any, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)


def policy_graph(
    normalizer: Any, policy_params: Any, activation: str, normalize: bool
) -> onnx.ModelProto:
    """observation [obs] -> normalizer -> MLP -> tanh(mean) = action [2]."""
    layers = policy_params["params"]
    dense = [layers[f"hidden_{i}"] for i in range(len(layers))]
    obs_size = dense[0]["kernel"].shape[0]
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
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
    run_dir = pathlib.Path(run_dir)
    agent = OmegaConf.load(run_dir / ".hydra" / "config.yaml").agent
    ckpt = ckpt or evaluation.best_checkpoint(run_dir)
    _, (normalizer, policy_params) = evaluation.load_policy_params(run_dir, ckpt)
    model = policy_graph(
        normalizer, policy_params, agent.activation, agent.normalize_observations
    )
    onnx.save(model, str(out_path))
    return ckpt
