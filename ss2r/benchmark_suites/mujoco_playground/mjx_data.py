"""Initial MJX data, as mujoco_playground's mjx_env.init (removed in playground 0.1)."""

from typing import Optional

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import sensor


def init(
    model: mjx.Model,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
) -> mjx.Data:
    data = mjx.make_data(model, impl=model.impl)
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    if act is not None:
        data = data.replace(act=act)
    if mocap_pos is not None:
        data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
    if mocap_quat is not None:
        data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
    data = mjx.forward(model, data)
    # On GPU, mjx 3.5 can return NaN accelerations for an initial state (the next
    # steps recompute them): use the unconstrained ones, else zero, for its sensors.
    failed = jp.isnan(data.qacc).any()
    fallback = jp.where(
        jp.isnan(data.qacc_smooth).any(), jp.zeros_like(data.qacc), data.qacc_smooth
    )
    data = data.replace(qacc=jp.where(failed, fallback, data.qacc))
    return jax.lax.cond(
        failed, lambda d: sensor.sensor_acc(model, d), lambda d: d, data
    )
