"""CyberRunner labyrinth environment.

The observation has the real robot's layout: 5 frames of
[alpha, beta, ball_x, ball_y, action], then 10 path points relative to the ball.
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import dm_control_suite
from mujoco_playground._src import mjx_env

from ss2r.benchmark_suites.mujoco_playground.cyberrunner.sim import (
    builder,
    constants,
    dynamics,
    maze,
)

BALL_POS_NOISE = 0.001  # 1 mm
JOINT_ANGLE_NOISE = 0.25 * np.pi / 180  # 0.25 deg
PROGRESS_SCALE = 1.0
GOAL_BONUS = 10.0
GOAL_THRESHOLD = 0.004  # 4 mm
BALL_START_HEIGHT = 0.0793  # marble resting on the flat board
FRAME_SIZE = 6  # joint(2) + ball(2) + action(2)
# MuJoCo Warp buffers, as in Thomas's env: contacts for all envs, rows per env.
NACON_PER_ENV = 8
NJMAX = 32
# Progress of the whole hard-maze path: its length in metres times 10.
TOTAL_PROGRESS = 10.0 * float(
    maze.compute_waypoint_distances(maze.get_layout("hard").waypoints)[1][-1]
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=constants.TIMESTEP * constants.FRAME_SKIP,  # 60 Hz
        sim_dt=constants.TIMESTEP,  # 600 Hz
        episode_length=2000,
        action_repeat=1,
        random_start=True,  # False: always start at the beginning, as on the robot
        layout="hard",  # maze: "hard", "medium" or "easy"
        impl="jax",  # "warp" for MuJoCo Warp (CUDA only)
        num_envs_hint=1,  # envs per batch, sizes the Warp contact buffer
        delay_ms=[0.0, 0.0],  # command delay range, drawn once per episode
        history_length=5,
        num_path_points=10,
        path_step=50,
        path_distance=0.0001,
        num_rays=32,
        action_rate_penalty=0.0,  # reward cost per squared command change
        kick_accel=0.0,  # largest random push on the ball, m/s^2 (0: no pushes)
        kick_interval_s=[1.0, 3.0],
        kick_duration_s=[0.05, 0.2],
        privileged_obs=False,  # dict observation with a privileged_state for the critic
        ball_stiction=False,  # a still ball stays until the tilt exceeds a threshold
        ctrl_limit=list(
            constants.CTRL_LIMIT
        ),  # alpha, beta command cap ([1, 1]: no cap)
        motor_dc=False,  # voltage-limited motor current and breakaway joint friction, no cap
    )


class CyberRunner(mjx_env.MjxEnv):
    """Roll the marble along the maze path to the goal."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        layout = maze.get_layout(self._config.layout)
        seg_lengths, cum_distances = maze.compute_waypoint_distances(layout.waypoints)
        self.total_progress = 10.0 * float(cum_distances[-1])
        self._mj_model = builder.build_model(
            layout,
            ball_stiction=self._config.ball_stiction,
            ctrl_limit=(self._config.ctrl_limit[0], self._config.ctrl_limit[1]),
            motor_dc=self._config.motor_dc,
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._board_body_id = self._mj_model.body("board").id
        self._marble_body_id = self._mj_model.body("marble").id
        self._marble_geom_id = int(self._mj_model.body_geomadr[self._marble_body_id])
        self._actuator_ids = jp.array(
            [self._mj_model.actuator(n).id for n in ("alpha_motor", "beta_motor")]
        )
        self._joint_ids = [
            self._mj_model.joint(n).id for n in ("alpha_joint", "beta_joint")
        ]
        self._dof_ids = jp.array(
            [self._mj_model.jnt_dofadr[j] for j in self._joint_ids]
        )
        segments, half_widths = maze.wall_segments(layout)
        self._walls = (jp.asarray(segments), jp.asarray(half_widths))

        self._waypoints = jp.asarray(layout.waypoints)
        self._walls_h = jp.asarray(layout.walls_h)
        self._walls_v = jp.asarray(layout.walls_v)
        self._holes = jp.asarray(layout.holes)
        self._seg_lengths = jp.asarray(seg_lengths)
        self._cum_distances = jp.asarray(cum_distances)
        self._path_points = jp.asarray(
            maze.interpolate_path(layout.waypoints, self._config.path_distance)
        )
        substep_ms = 1000.0 * self.sim_dt
        low, high = (int(round(ms / substep_ms)) for ms in self._config.delay_ms)
        self._delay_substeps = (low, high)
        # Commands that can still be pending: sent up to `high` substeps ago.
        self._num_commands = -(-high // self.n_substeps) + 1

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_spawn, k_bias_ball, k_bias_joint, k_noise = jax.random.split(rng, 5)
        delay = self._draw_delay(jax.random.fold_in(k_spawn, 1))
        if self._config.random_start:  # any waypoint but the goal
            spawn = self._waypoints[
                jax.random.randint(k_spawn, (), 0, self._waypoints.shape[0] - 1)
            ]
        else:
            spawn = self._waypoints[0]
        qpos = jp.asarray(self._mj_model.qpos0)
        qpos = qpos.at[2:4].set(spawn)
        qpos = qpos.at[4].set(BALL_START_HEIGHT)
        qpos = qpos.at[5:9].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        data = self._make_data().replace(qpos=qpos)
        data = mjx.forward(self.mjx_model, data)

        bias_ball = jax.random.uniform(
            k_bias_ball, (2,), minval=-BALL_POS_NOISE, maxval=BALL_POS_NOISE
        )
        bias_joint = jax.random.uniform(
            k_bias_joint, (2,), minval=-JOINT_ANGLE_NOISE, maxval=JOINT_ANGLE_NOISE
        )
        ball = self.ball_position(data)
        start_progress, *_, found = self._path_progress(ball)
        start_progress = jp.where(found, start_progress, -1.0)
        frame, ball_seen = self._measure(
            data, ball, jp.zeros(2), bias_ball, bias_joint, k_noise
        )
        history = jp.tile(frame, (self._config.history_length, 1))
        info = {
            "rng": rng,
            "bias_ball": bias_ball,
            "bias_joint": bias_joint,
            "history": history,
            "prev_progress": start_progress,
            # Restored by step() after an auto-reset.
            "start_history": history,
            "start_progress": start_progress,
            "delay": delay,
            "commands": jp.zeros((self._num_commands, 2)),
            "cost": jp.zeros(()),
            "kick_wait": self._draw_steps(
                jax.random.fold_in(k_spawn, 2), self._config.kick_interval_s
            ),
            "kick_left": jp.zeros((), jp.int32),
            "kick_force": jp.zeros(3),
            "stuck": jp.zeros((), bool),
        }
        metrics = {
            "success": jp.zeros(()),
            "progress": jp.zeros(()),
            "action_rate": jp.zeros(()),
        }
        obs = self._observation(history, ball_seen)
        if self._config.privileged_obs:
            obs = self._with_privileged(
                obs, data, delay, start_progress, jp.zeros(3), info["stuck"]
            )
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info: Dict[str, jax.Array] = dict(state.info)
        # The episode wrapper zeroes `steps` after an auto-reset.
        new_episode = info.get("steps", jp.ones(())) == 0
        history = jp.where(new_episode, info["start_history"], info["history"])
        prev_progress = jp.where(
            new_episode, info["start_progress"], info["prev_progress"]
        )
        rng, k_noise = jax.random.split(info["rng"])
        delay = jp.where(
            new_episode,
            self._draw_delay(jax.random.fold_in(k_noise, 1)),
            info["delay"],
        )
        commands = jp.where(new_episode, 0.0, info["commands"])

        action = jp.clip(action, -1.0, 1.0)
        action_rate = jp.sum((action - commands[0]) ** 2)
        commands = jp.concatenate([action[None], commands[:-1]])  # newest first
        kick = jp.zeros(3)
        if self._config.kick_accel > 0:
            kick = self._kick(info, new_episode, jax.random.fold_in(k_noise, 2))
        stuck = jp.where(new_episode, False, info["stuck"])
        data, stuck = self._physics(state.data, commands, delay, kick, stuck)

        ball = self.ball_position(data)
        curr_progress, *_, found = self._path_progress(ball)
        curr_progress = jp.where(found, curr_progress, -1.0)
        both_valid = (curr_progress >= 0) & (prev_progress >= 0)
        progress_reward = jp.where(
            both_valid, (curr_progress - prev_progress) * PROGRESS_SCALE, 0.0
        )
        at_goal = jp.linalg.norm(ball - self._waypoints[-1]) < GOAL_THRESHOLD
        in_hole = jp.any(
            jp.linalg.norm(self._holes - ball, axis=1) < constants.HOLE_RADIUS
        )
        reward = (
            progress_reward
            + jp.where(at_goal, GOAL_BONUS, 0.0)
            - self._config.action_rate_penalty * action_rate
        )
        done = (in_hole | at_goal).astype(jp.float32)

        frame, ball_seen = self._measure(
            data, ball, action, info["bias_ball"], info["bias_joint"], k_noise
        )
        history = jp.concatenate([history[1:], frame[None]], axis=0)
        info.update(
            rng=rng,
            delay=delay,
            commands=commands,
            history=history,
            prev_progress=jp.where(found, curr_progress, prev_progress),
            stuck=stuck,
        )
        metrics = {
            **state.metrics,
            "success": at_goal.astype(jp.float32),
            "progress": progress_reward / self.total_progress,
            "action_rate": action_rate,
        }
        obs = self._observation(history, ball_seen)
        if self._config.privileged_obs:
            obs = self._with_privileged(
                obs, data, delay, info["prev_progress"], kick, stuck
            )
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _draw_delay(self, key: jax.Array) -> jax.Array:
        low, high = self._delay_substeps
        return jax.random.randint(key, (), low, high + 1)

    def _draw_steps(self, key: jax.Array, seconds) -> jax.Array:
        low, high = (int(round(s / self.dt)) for s in seconds)
        return jax.random.randint(key, (), low, high + 1)

    def _kick(self, info: Dict[str, jax.Array], new_episode, key) -> jax.Array:
        """Force on the ball this step; updates the kick timers in info."""
        k_wait, k_len, k_dir, k_size = jax.random.split(key, 4)
        wait = jp.where(
            new_episode,
            self._draw_steps(k_wait, self._config.kick_interval_s),
            info["kick_wait"],
        )
        left = jp.where(new_episode, 0, info["kick_left"])
        start = (wait <= 0) & (left <= 0)
        angle = jax.random.uniform(k_dir, (), maxval=2 * jp.pi)
        accel = jax.random.uniform(k_size, (), maxval=self._config.kick_accel)
        mass = self.mjx_model.body_mass[self._marble_body_id]
        new = mass * accel * jp.array([jp.cos(angle), jp.sin(angle), 0.0])
        force = jp.where(start, new, jp.where(new_episode, 0.0, info["kick_force"]))
        left = jp.where(
            start, self._draw_steps(k_len, self._config.kick_duration_s), left
        )
        wait = jp.where(
            start, self._draw_steps(k_wait, self._config.kick_interval_s), wait - 1
        )
        active = left > 0
        info.update(
            kick_wait=wait, kick_left=left - active.astype(jp.int32), kick_force=force
        )
        return jp.where(active, force, 0.0)

    def _with_privileged(
        self, obs, data, delay, prog, kick, stuck
    ) -> Dict[str, jax.Array]:
        """The actor's observation, and for the critic also the clean state and physics."""
        m, act, dof = self.mjx_model, self._actuator_ids, self._dof_ids
        board = data.xmat[self._board_body_id].reshape(3, 3)
        ball_vel = (board.T @ data.qvel[2:5])[:2]
        g = self._marble_geom_id
        extra = [
            data.qpos[:2],
            data.qvel[:2],
            self.ball_position(data),
            ball_vel,
            data.act,
            data.ctrl,
            jp.atleast_1d(delay * self.sim_dt),
            m.actuator_gear[act, 0],
            m.actuator_dynprm[act, 0],
            self._speed_limit(m),
            m.dof_damping[dof],
            m.dof_frictionloss[dof],
            jp.stack(
                [
                    m.geom_friction[g, 0],
                    m.body_mass[self._marble_body_id],
                    m.geom_solref[g, 1],
                ]
            ),
            m.opt.gravity[:2],
            jp.atleast_1d(prog / self.total_progress),
            kick[:2],
        ]
        if self._config.ball_stiction:
            extra += [m.geom_friction[g, 1:], jp.atleast_1d(stuck)]
        privileged = jp.concatenate([obs] + [jp.ravel(x) for x in extra])
        return {"state": obs, "privileged_state": privileged.astype(jp.float32)}

    def _physics(self, data, commands, delay, kick, stuck):
        """Substeps where each command acts from `delay` substeps after it was sent."""
        n = self.n_substeps
        pushed = self._config.kick_accel > 0 or self._config.ball_stiction

        def substep(carry, j):
            d, stuck = carry
            force = kick
            if self._config.ball_stiction:
                friction, stuck = self._stiction(d, stuck)
                force = force + friction
            if pushed:
                xfrc = jp.zeros_like(d.xfrc_applied)
                d = d.replace(xfrc_applied=xfrc.at[self._marble_body_id, :3].set(force))
            # Age (in control steps) of the newest command that has arrived.
            age = jp.clip((delay - j + n - 1) // n, 0, commands.shape[0] - 1)
            if self._config.motor_dc:
                d = self._dc_motor(d)
            d = mjx.step(self.mjx_model, d.replace(ctrl=commands[age]))
            return (d, stuck), None

        return jax.lax.scan(substep, (data, stuck), jp.arange(n))[0]

    def _dc_motor(self, d: mjx.Data) -> mjx.Data:
        """Motor currents within the voltage limit; joint friction eased to sliding when moving."""
        m, act, dof = self.mjx_model, self._actuator_ids, self._dof_ids
        prm = m.actuator_gainprm[act]
        speed = d.qvel[dof] * jp.sign(m.actuator_gear[act, 0])
        low, high = dynamics.current_bounds(speed, prm[:, 1], prm[:, 2])
        assist = dynamics.friction_assist(
            d.qvel[dof], m.dof_frictionloss[dof], prm[:, 3], prm[:, 4]
        )
        return d.replace(
            act=d.act.at[act].set(jp.clip(d.act[act], low, high)),
            qfrc_applied=jp.zeros_like(d.qfrc_applied).at[dof].set(assist),
        )

    def _speed_limit(self, m: mjx.Model) -> jax.Array:
        """What limits the board speed per axis: the no-load speed or the command cap."""
        act = self._actuator_ids
        if self._config.motor_dc:
            return m.actuator_gainprm[act, 2]
        return m.actuator_ctrlrange[act, 1]

    def _stiction(self, d: mjx.Data, stuck: jax.Array):
        """Static and rolling friction on the ball (world force) and whether it is stuck."""
        m, g = self.mjx_model, self._marble_geom_id
        board = d.xmat[self._board_body_id].reshape(3, 3)
        alpha, beta = self._joint_ids
        spin = (
            d.qvel[self._dof_ids[0]] * d.xaxis[alpha]
            + d.qvel[self._dof_ids[1]] * d.xaxis[beta]
        )
        lever = (
            d.xpos[self._marble_body_id] - d.xanchor[alpha]
        )  # both axes pass through it
        velocity = board.T @ (d.qvel[2:5] - jp.cross(spin, lever))
        friction, stuck = dynamics.stiction_force(
            self.ball_position(d),
            velocity[:2],
            board.T @ m.opt.gravity,
            stuck,
            m.geom_friction[g, 1],
            m.geom_friction[g, 2],
            m.geom_friction[g, 0],
            m.body_mass[self._marble_body_id],
            self._walls,
            self.sim_dt,
        )
        return board[:, :2] @ friction, stuck

    def _make_data(self) -> mjx.Data:
        if self._config.impl == "warp":
            return mjx.make_data(
                self._mj_model,
                impl="warp",
                naconmax=NACON_PER_ENV * self._config.num_envs_hint,
                njmax=NJMAX,
            )
        return mjx.make_data(self._mj_model)

    def ball_position(self, data: mjx.Data) -> jax.Array:
        """Marble position in board coordinates."""
        board_pos = data.xpos[self._board_body_id]
        board_mat = data.xmat[self._board_body_id].reshape(3, 3)
        marble_pos = data.xpos[self._marble_body_id]
        return (board_mat.T @ (marble_pos - board_pos))[:2]

    def _measure(self, data, ball, action, bias_ball, bias_joint, rng):
        """Noisy history frame and noisy ball position."""
        k_ball, k_joint = jax.random.split(rng)
        noise_ball = jax.random.uniform(
            k_ball, (2,), minval=-BALL_POS_NOISE, maxval=BALL_POS_NOISE
        )
        noise_joint = jax.random.uniform(
            k_joint, (2,), minval=-JOINT_ANGLE_NOISE, maxval=JOINT_ANGLE_NOISE
        )
        joints = data.qpos[:2] + bias_joint + noise_joint
        ball_seen = ball + bias_ball + noise_ball
        frame = jp.concatenate([joints, ball_seen, action]).astype(jp.float32)
        return frame, ball_seen.astype(jp.float32)

    def _observation(self, history: jax.Array, ball_seen: jax.Array) -> jax.Array:
        rel_path = maze.relative_path(
            self._path_points,
            ball_seen,
            self._config.num_path_points,
            self._config.path_step,
        )
        return jp.concatenate([history.reshape(-1), rel_path]).astype(jp.float32)

    def _path_progress(self, ball: jax.Array) -> Tuple[jax.Array, ...]:
        return maze.compute_path_progress_jax(
            ball,
            self._waypoints,
            self._seg_lengths,
            self._cum_distances,
            self._walls_h,
            self._walls_v,
            self._holes,
            num_rays=self._config.num_rays,
        )

    @property
    def waypoints(self) -> jax.Array:
        return self._waypoints

    @property
    def holes(self) -> jax.Array:
        return self._holes

    @property
    def xml_path(self) -> str:
        raise NotImplementedError("CyberRunner is built with MjSpec, not from XML.")

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


dm_control_suite.register_environment("CyberRunner", CyberRunner, default_config)
