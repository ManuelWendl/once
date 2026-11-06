"""CyberRunner on the MuJoCo CPU backend, for non-episodic online learning."""

import functools
from typing import Any, Callable, Mapping, Sequence, Tuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import (
    cyberrunner,
    randomize,
    recovery,
)
from ss2r.benchmark_suites.mujoco_playground.cyberrunner.sim import (
    builder,
    constants,
    dynamics,
    maze,
)

CTRL_DT = constants.TIMESTEP * constants.FRAME_SKIP  # 60 Hz
HISTORY_LENGTH = 5
# Model fields that randomize.domain_randomization sets.
_RANDOMIZED_FIELDS = (
    "actuator_gear",
    "actuator_dynprm",
    "actuator_ctrlrange",
    "dof_damping",
    "dof_frictionloss",
    "geom_friction",
    "body_mass",
    "body_inertia",
    "geom_solref",
)


def path_progress_fn(layout: maze.Layout, num_rays: int) -> Callable:
    """Path progress of a ball, as in the task's reward."""
    seg_lengths, cum_distances = maze.compute_waypoint_distances(layout.waypoints)
    return functools.partial(
        maze.compute_path_progress_jax,
        waypoints=jp.asarray(layout.waypoints),
        seg_lengths=jp.asarray(seg_lengths),
        cum_distances=jp.asarray(cum_distances),
        walls_h=jp.asarray(layout.walls_h),
        walls_v=jp.asarray(layout.walls_v),
        holes=jp.asarray(layout.holes),
        num_rays=num_rays,
    )


def relative_path_fn(
    layout: maze.Layout,
    num_path_points: int,
    path_step: int,
    path_distance: float,
    num_rays: int,
    visible_path: bool,
) -> Callable:
    """The path points of a ball. With `visible_path` they start from the stretch of path
    the task's reward sees rather than from the nearest point, which can be behind a wall
    and switches with the ball's noise between two stretches of the path. The nearest
    point remains where nothing is visible and behind the path's start, where the rays
    miss the first segment, collinear with the ball."""
    points = jp.asarray(maze.interpolate_path(layout.waypoints, path_distance))
    if not visible_path:
        return lambda ball: maze.relative_path(points, ball, num_path_points, path_step)
    progress = path_progress_fn(layout, num_rays)
    counts = maze.segment_point_counts(layout.waypoints, path_distance)
    offsets = jp.asarray(np.r_[0, np.cumsum(counts)[:-1]])
    counts = jp.asarray(counts)

    def relative_path(ball: jax.Array) -> jax.Array:
        _, segment, along, _, found = progress(ball)
        nearest = maze.nearest_point(points, ball)
        visible = offsets[segment] + jp.floor(along * counts[segment]).astype(int)
        index = jp.where(found & (nearest > 0), visible, nearest)
        return maze.path_from(points, index, ball, num_path_points, path_step)

    return relative_path


class CyberRunner:
    """The maze task of playground's CyberRunner, on any system: observation, task reward,
    goal and R. Subclasses move the ball and measure it.

    An episode never ends, unless the ball falls into a hole. In this case `terminated`
    is returned and the caller should reset. At the goal the ball is put back at the
    start, as on the robot, and also after `max_episode_steps` without reaching it, if set.
    """

    action_size = 2

    def __init__(
        self,
        layout: str = "easy",
        history_length: int = HISTORY_LENGTH,
        num_path_points: int = 10,
        path_step: int = 50,
        path_distance: float = 0.0001,
        num_rays: int = 32,
        visible_path: bool = False,
        max_episode_steps: int | None = None,
        recovery_hole_distance: float = recovery.RECOVERY_HOLE_DISTANCE,
        recovery_speed: float = recovery.RECOVERY_SPEED,
        action_repeat: int = 1,
    ):
        self._layout = maze.get_layout(layout)
        self._history_length = history_length
        self._max_episode_steps = (
            float("inf") if max_episode_steps is None else max_episode_steps
        )
        self._recovery_hole_distance = recovery_hole_distance
        self._recovery_speed = recovery_speed
        self._action_repeat = action_repeat

        # The playground env's functions, jitted for single states.
        _, cum_distances = maze.compute_waypoint_distances(self._layout.waypoints)
        self._total_progress = 10.0 * float(cum_distances[-1])
        self._path_progress = jax.jit(path_progress_fn(self._layout, num_rays))
        self._relative_path = jax.jit(
            relative_path_fn(
                self._layout,
                num_path_points,
                path_step,
                path_distance,
                num_rays,
                visible_path,
            )
        )
        self.observation_size = history_length * cyberrunner.FRAME_SIZE + 2 * (
            num_path_points
        )
        self.dt = CTRL_DT * action_repeat

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def _physics(self, action: np.ndarray) -> None:
        """Moves the system by one control step."""
        raise NotImplementedError

    def _ball_position(self) -> np.ndarray:
        """Ball position in board coordinates."""
        raise NotImplementedError

    def _measure(self, ball: np.ndarray, action: np.ndarray):
        """History frame and ball position, as the policy sees them."""
        raise NotImplementedError

    def _start_episode(self) -> np.ndarray:
        """An episode from the current measurement: a history of it, the path from it."""
        self._steps = 0
        ball = self._ball_position()
        self._ball_before = ball
        self._prev_progress = self._progress(ball)
        frame, ball_seen = self._measure(ball, np.zeros(2))
        self._history = np.tile(frame, (self._history_length, 1))
        return self._observation(ball_seen)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        reward = 0.0
        for _ in range(self._action_repeat):
            step_reward, ball_seen, in_hole, at_goal, in_r = self._control_step(action)
            reward += step_reward
            if in_hole or at_goal or self._steps >= self._max_episode_steps:
                break
        # Task metrics: how far along the path the ball is (logged as its maximum), and
        # the goals (counted).
        info = {
            "in_recovery_set": in_r,
            "maxima": {"progress": self._path_fraction()},
            "counts": {"goals": float(at_goal)},
        }
        if not in_hole and (at_goal or self._steps >= self._max_episode_steps):
            obs = self.reset()
        else:
            obs = self._observation(ball_seen)
        return obs, reward, in_hole, info

    def _control_step(
        self, action: np.ndarray
    ) -> Tuple[float, np.ndarray, bool, bool, bool]:
        """One step of playground's CyberRunner.step: the reward, the observed ball, and
        whether the ball is in a hole, at the goal and in R."""
        self._physics(action)
        ball = self._ball_position()
        progress = self._progress(ball)
        reward = 0.0
        if progress >= 0 and self._prev_progress >= 0:
            reward = (progress - self._prev_progress) * cyberrunner.PROGRESS_SCALE
        if progress >= 0:
            self._prev_progress = progress
        holes = self._layout.holes
        at_goal = bool(
            np.linalg.norm(ball - self._layout.waypoints[-1])
            < cyberrunner.GOAL_THRESHOLD
        )
        in_hole = bool(recovery.violates(ball, holes))
        reward += cyberrunner.GOAL_BONUS if at_goal else 0.0
        velocity = (ball - self._ball_before) / CTRL_DT
        self._ball_before = ball
        in_r = bool(
            recovery.recovery_reward_fn(
                ball,
                velocity,
                holes,
                self._recovery_hole_distance,
                self._recovery_speed,
            )
        )
        frame, ball_seen = self._measure(ball, action)
        self._history = np.concatenate([self._history[1:], frame[None]], axis=0)
        self._steps += 1
        return float(reward), ball_seen, in_hole, at_goal, in_r

    def _progress(self, ball: np.ndarray) -> float:
        """Path progress of the ball, -1 where no part of the path is visible."""
        progress, *_, found = self._path_progress(ball)
        return float(progress) if bool(found) else -1.0

    def _path_fraction(self) -> float:
        """The share of the path the ball has covered, at its last visible progress."""
        return max(self._prev_progress, 0.0) / self._total_progress

    def _observation(self, ball_seen: np.ndarray) -> np.ndarray:
        rel_path = np.asarray(self._relative_path(ball_seen))
        return np.concatenate([self._history.reshape(-1), rel_path]).astype(np.float32)


class CyberRunnerCPU(CyberRunner):
    """Same model as playground's CyberRunner, on the MuJoCo CPU backend."""

    def __init__(
        self,
        layout: str = "easy",
        ball_stiction: bool = True,
        ctrl_limit: Sequence[float] = constants.CTRL_LIMIT,
        delay_ms: float = 0.0,
        real_params: Mapping[str, float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ):
        super().__init__(layout, **kwargs)
        self.model = builder.build_model(
            self._layout,
            ball_stiction=ball_stiction,
            ctrl_limit=(ctrl_limit[0], ctrl_limit[1]),
        )
        self.model.opt.timestep = constants.TIMESTEP
        self._apply_real_params(real_params or {})
        self.data = mujoco.MjData(self.model)
        self._ball_stiction = ball_stiction
        self._board_body_id = self.model.body("board").id
        self._marble_body_id = self.model.body("marble").id
        self._marble_geom_id = int(self.model.body_geomadr[self._marble_body_id])
        self._joint_ids = [
            self.model.joint(n).id for n in ("alpha_joint", "beta_joint")
        ]
        self._dof_ids = [int(self.model.jnt_dofadr[j]) for j in self._joint_ids]
        self._delay = int(round(delay_ms / (1000.0 * constants.TIMESTEP)))
        self._num_commands = -(-self._delay // constants.FRAME_SKIP) + 1
        self._rng = np.random.default_rng(seed)
        self._stiction_force = jax.jit(dynamics.stiction_force, static_argnames="dt")
        segments, half_widths = maze.wall_segments(self._layout)
        self._walls = (jp.asarray(segments), jp.asarray(half_widths))

    def _apply_real_params(self, params: Mapping[str, float]) -> None:
        """Sets the nominal model's physics with the semantics of the sim randomization:
        each of its ranges collapsed to the given value."""
        if not params:
            return
        values = {name: 1.0 for name in randomize.SCALE_GROUPS}
        values.update(
            {
                randomize.TILT_GROUP: 0.0,
                randomize.STATIC_GROUP: 0.0,
                randomize.ROLLING_GROUP: 0.0,
            }
        )
        unknown = set(params) - set(values)
        if unknown:
            raise ValueError(f"Unknown real_params {sorted(unknown)}")
        values.update(params)
        ranges = {name: [value, value] for name, value in values.items()}
        model, _, _ = randomize.domain_randomization(
            mjx.put_model(self.model),
            jax.random.split(jax.random.PRNGKey(0), 1),
            ranges,
        )
        for name in _RANDOMIZED_FIELDS:
            getattr(self.model, name)[:] = np.asarray(getattr(model, name))[0]
        self.model.opt.gravity[:] = np.asarray(model.opt.gravity)[0]

    def reset(self) -> np.ndarray:
        """The board level and still, the ball at rest at the maze's start."""
        qpos = self.model.qpos0.copy()
        qpos[2:4] = self._layout.waypoints[0]
        qpos[4] = cyberrunner.BALL_START_HEIGHT
        qpos[5:9] = [1.0, 0.0, 0.0, 0.0]
        return self.set_state(qpos, np.zeros(self.model.nv))

    def get_state(self) -> dict[str, np.ndarray]:
        """The state of the simulation, as taken by `set_state`."""
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Starts an episode from this state: new observation biases, no pending
        commands, a history of the current measurement."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self._bias_ball = self._rng.uniform(
            -cyberrunner.BALL_POS_NOISE, cyberrunner.BALL_POS_NOISE, 2
        )
        self._bias_joint = self._rng.uniform(
            -cyberrunner.JOINT_ANGLE_NOISE, cyberrunner.JOINT_ANGLE_NOISE, 2
        )
        self._commands = np.zeros((self._num_commands, 2))
        self._stuck = False
        return self._start_episode()

    def _physics(self, action: np.ndarray) -> None:
        """Substeps where each command acts from `delay` substeps after it was sent."""
        self._commands = np.concatenate([action[None], self._commands[:-1]])
        n = constants.FRAME_SKIP
        for j in range(n):
            if self._ball_stiction:
                self.data.xfrc_applied[self._marble_body_id, :3] = self._stiction()
            age = min(max((self._delay - j + n - 1) // n, 0), self._num_commands - 1)
            self.data.ctrl[:] = self._commands[age]
            mujoco.mj_step(self.model, self.data)

    def _stiction(self) -> np.ndarray:
        """Static and rolling friction on the ball (world force); updates `stuck`."""
        m, d, g = self.model, self.data, self._marble_geom_id
        board = d.xmat[self._board_body_id].reshape(3, 3)
        alpha, beta = self._joint_ids
        spin = (
            d.qvel[self._dof_ids[0]] * d.xaxis[alpha]
            + d.qvel[self._dof_ids[1]] * d.xaxis[beta]
        )
        lever = d.xpos[self._marble_body_id] - d.xanchor[alpha]
        velocity = board.T @ (d.qvel[2:5] - np.cross(spin, lever))
        friction, stuck = self._stiction_force(
            self._ball_position(),
            velocity[:2],
            board.T @ m.opt.gravity,
            self._stuck,
            m.geom_friction[g, 1],
            m.geom_friction[g, 2],
            m.geom_friction[g, 0],
            m.body_mass[self._marble_body_id],
            self._walls,
            dt=constants.TIMESTEP,
        )
        self._stuck = bool(stuck)
        return board[:, :2] @ np.asarray(friction)

    def _ball_position(self) -> np.ndarray:
        """Marble position in board coordinates."""
        board_pos = self.data.xpos[self._board_body_id]
        board_mat = self.data.xmat[self._board_body_id].reshape(3, 3)
        marble_pos = self.data.xpos[self._marble_body_id]
        return (board_mat.T @ (marble_pos - board_pos))[:2]

    def _measure(self, ball: np.ndarray, action: np.ndarray):
        """Noisy history frame and noisy ball position."""
        noise_ball = self._rng.uniform(
            -cyberrunner.BALL_POS_NOISE, cyberrunner.BALL_POS_NOISE, 2
        )
        noise_joint = self._rng.uniform(
            -cyberrunner.JOINT_ANGLE_NOISE, cyberrunner.JOINT_ANGLE_NOISE, 2
        )
        joints = self.data.qpos[:2] + self._bias_joint + noise_joint
        ball_seen = ball + self._bias_ball + noise_ball
        frame = np.concatenate([joints, ball_seen, action]).astype(np.float32)
        return frame, ball_seen.astype(np.float32)


def observed_ball(obs: Any, history_length: int = HISTORY_LENGTH) -> Tuple[Any, Any]:
    """The newest observed ball position, and its mean velocity over the history."""
    frames = obs[..., : history_length * cyberrunner.FRAME_SIZE].reshape(
        obs.shape[:-1] + (history_length, cyberrunner.FRAME_SIZE)
    )
    balls = frames[..., 2:4]
    velocity = (balls[..., -1, :] - balls[..., 0, :]) / ((history_length - 1) * CTRL_DT)
    return balls[..., -1, :], velocity


def next_observation_fn(task_params: Mapping[str, Any]) -> Callable:
    """Corrects a batch of predicted observations: with `visible_path` their path points
    are recomputed from their ball, as a jump of the path can't be predicted."""
    if not task_params.get("visible_path", False):
        return lambda obs: obs
    history_length = task_params.get("history_length", HISTORY_LENGTH)
    relative_path = jax.vmap(
        relative_path_fn(
            maze.get_layout(task_params.get("layout", "easy")),
            task_params.get("num_path_points", 10),
            task_params.get("path_step", 50),
            task_params.get("path_distance", 0.0001),
            task_params.get("num_rays", 32),
            True,
        )
    )
    frames_size = history_length * cyberrunner.FRAME_SIZE

    def next_observation(obs: jax.Array) -> jax.Array:
        ball, _ = observed_ball(obs, history_length)
        return jp.concatenate([obs[..., :frames_size], relative_path(ball)], axis=-1)

    return next_observation


def recovery_fns(task_params: Mapping[str, Any]) -> Tuple[Callable, Callable]:
    """The recovery reward and the constraint violation of an observation, as in the
    recovery task, with the ball's velocity averaged over the history: the newest frame
    alone would be about as noisy as R's speed bound."""
    holes = maze.get_layout(task_params.get("layout", "easy")).holes
    history_length = task_params.get("history_length", HISTORY_LENGTH)
    distance = task_params.get(
        "recovery_hole_distance", recovery.RECOVERY_HOLE_DISTANCE
    )
    speed = task_params.get("recovery_speed", recovery.RECOVERY_SPEED)

    def recovery_reward_fn(obs):
        ball, velocity = observed_ball(obs, history_length)
        return recovery.recovery_reward_fn(ball, velocity, holes, distance, speed)

    def violates_fn(obs):
        ball, _ = observed_ball(obs, history_length)
        return recovery.violates(ball, holes)

    return recovery_reward_fn, violates_fn
