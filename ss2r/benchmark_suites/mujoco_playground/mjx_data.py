"""Initial MJX data, as mujoco_playground's mjx_env.init (removed in playground 0.1)."""

from typing import Optional

import jax
from mujoco import mjx


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
    return mjx.forward(model, data)
