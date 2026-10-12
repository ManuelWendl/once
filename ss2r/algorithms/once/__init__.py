from omegaconf import OmegaConf


def get_train_fn(cfg, prior_cfg, checkpoint_path):
    import functools

    import ss2r.algorithms.once.train as once

    to_container = lambda x: OmegaConf.to_container(x, resolve=True)
    return functools.partial(
        once.train,
        environment_cfg=to_container(cfg.environment),
        agent_cfg=to_container(cfg.agent),
        prior_agent_cfg=to_container(prior_cfg.agent),
        num_timesteps=cfg.training.num_timesteps,
        action_repeat=cfg.training.action_repeat,
        seed=cfg.training.seed,
        log_every=cfg.training.log_every,
        checkpoint_logdir=checkpoint_path,
    )
