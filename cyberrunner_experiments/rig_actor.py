"""ONCE's actor on the rig's Mac, with the learner on another machine. With the same
overrides as the learner, the bridge up, and from the repository's root:

    ONCE_REMOTE_TOKEN=... PYTHONPATH=$PWD python cyberrunner_experiments/rig_actor.py \
        +experiment=cyberrunner_once_rig
"""

import logging
import signal
import threading

import hydra
import jax
from omegaconf import OmegaConf

from ss2r.algorithms.once.actor import actor_loop, make_actor_policy
from ss2r.algorithms.once.remote import RobotServer
from ss2r.benchmark_suites import mujoco_cpu
from train_once import get_prior_cfg

# Without a request of the learner for this long, the actor stops.
LEARNER_TIMEOUT_S = 60.0


@hydra.main(version_base=None, config_path="../ss2r/configs", config_name="train_once")
def main(cfg) -> None:
    jax.config.update("jax_platforms", "cpu")
    to_container = lambda x: OmegaConf.to_container(x, resolve=True)
    environment_cfg = to_container(cfg.environment)
    agent_cfg = to_container(cfg.agent)
    prior_agent_cfg = to_container(get_prior_cfg(cfg).agent)
    env = mujoco_cpu.make(
        environment_cfg, cfg.training.action_repeat, cfg.training.seed
    )
    policy_step, num_params = make_actor_policy(env, agent_cfg, prior_agent_cfg)
    server = RobotServer(agent_cfg["remote_connect"], num_params)
    env.on_pause = server.set_paused
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def should_stop() -> bool:
        if server.seconds_since_request() > LEARNER_TIMEOUT_S:
            logging.error("The learner is silent: stopping.")
            return True
        return stop.is_set()

    try:
        actor_loop(
            env,
            policy_step,
            agent_cfg,
            cfg.training.seed,
            server,
            server,
            should_stop,
            cfg.training.store_trajectory,
        )
    finally:
        env.close()
        server.close()


if __name__ == "__main__":
    main()
