import functools
import logging
import os

import hydra
import jax
from brax.training.agents.sac import checkpoint
from omegaconf import OmegaConf

from ss2r import benchmark_suites
from ss2r.algorithms import once, sac
from ss2r.common.logging import TrainingLogger
from ss2r.common.wandb import get_state_path, get_wandb_checkpoint

_LOG = logging.getLogger(__name__)


def report(logger, offset, prefix, num_steps, metrics):
    metrics = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
    logger.log(metrics, offset + num_steps)


def get_prior_cfg(cfg):
    """The prior is configured as a regular train_brax experiment."""
    return hydra.compose(
        "train_brax",
        overrides=[
            f"+experiment={cfg.prior.experiment}",
            f"training.seed={cfg.training.seed}",
            *cfg.prior.overrides,
        ],
    )


def train_recovery_prior(prior_cfg, logger):
    """Trains the recovery prior in simulation, with domain randomization."""
    train_env_wrap_fn, eval_env_wrap_fn = benchmark_suites.get_wrap_env_fn(prior_cfg)
    train_env, eval_env = benchmark_suites.make(
        prior_cfg, train_env_wrap_fn, eval_env_wrap_fn
    )
    train_fn = sac.get_train_fn(
        prior_cfg,
        restore_checkpoint_path=None,
        checkpoint_path=f"{get_state_path()}/prior",
    )
    _, params, _ = train_fn(
        environment=train_env,
        eval_env=eval_env,
        progress_fn=functools.partial(report, logger, 0, "prior/"),
    )
    return params


def get_recovery_prior(cfg, prior_cfg, logger):
    if cfg.prior.checkpoint_path is not None:
        return checkpoint.load(cfg.prior.checkpoint_path), 0
    if cfg.prior.wandb_id is not None:
        path = get_wandb_checkpoint(cfg.prior.wandb_id, cfg.wandb.entity)
        return checkpoint.load(path), 0
    params = train_recovery_prior(prior_cfg, logger)
    return params, prior_cfg.training.num_timesteps


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_once")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    prior_cfg = get_prior_cfg(cfg)
    with jax.disable_jit(not cfg.jit):
        prior_params, prior_steps = get_recovery_prior(cfg, prior_cfg, logger)
        _LOG.info("Starting online training.")
        train_fn = once.get_train_fn(
            cfg,
            prior_cfg,
            checkpoint_path=get_state_path() if cfg.training.store_checkpoint else None,
            trajectory_path=f"{os.getcwd()}/trajectory"
            if cfg.training.store_trajectory
            else None,
        )
        train_fn(
            prior_params=prior_params,
            progress_fn=functools.partial(report, logger, prior_steps, ""),
        )
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
