from typing import Any

import jax
import jax.numpy as jnp
from brax.training.types import PRNGKey

from ss2r.algorithms.once.types import RingBufferState


def init(dummy_data_sample: Any, capacity: int) -> RingBufferState:
    data = jax.tree_util.tree_map(
        lambda x: jnp.zeros((capacity,) + x.shape, x.dtype), dummy_data_sample
    )
    return RingBufferState(  # type: ignore
        data=data,
        insert_position=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
    )


def insert(buffer_state: RingBufferState, samples: Any) -> RingBufferState:
    """Inserts samples with a (static) leading batch dimension."""
    capacity = jax.tree_util.tree_leaves(buffer_state.data)[0].shape[0]
    num_samples = jax.tree_util.tree_leaves(samples)[0].shape[0]
    indices = (buffer_state.insert_position + jnp.arange(num_samples)) % capacity
    data = jax.tree_util.tree_map(
        lambda x, y: x.at[indices].set(y.astype(x.dtype)), buffer_state.data, samples
    )
    return RingBufferState(  # type: ignore
        data=data,
        insert_position=(buffer_state.insert_position + num_samples) % capacity,
        size=jnp.minimum(buffer_state.size + num_samples, capacity),
    )


def sample(buffer_state: RingBufferState, key: PRNGKey, batch_size: int) -> Any:
    indices = jax.random.randint(
        key, (batch_size,), 0, jnp.maximum(buffer_state.size, 1)
    )
    return jax.tree_util.tree_map(lambda x: x[indices], buffer_state.data)
