"""Go to goal on the MuJoCo CPU backend, for non-episodic online learning."""

from typing import Any, Mapping

import mujoco
import numpy as np

from ss2r.benchmark_suites.safety_gym import go_to_goal, lidar, recovery

_SENSOR_COMPONENTS = go_to_goal._ROBOT_TO_SENSOR_TO_COMPONENTS["point"]
_N_SUBSTEPS = 2
_HAZARD_SIZE = 0.2
_MAX_PLACEMENT_TRIES = 10000


def _compute_lidar(
    robot_pos: np.ndarray, robot_mat: np.ndarray, targets_pos: np.ndarray
) -> np.ndarray:
    """Numpy version of `lidar.compute_lidar`."""
    num_bins = lidar.NUM_LIDAR_BINS
    bin_size = 2 * np.pi / num_bins
    obs = np.zeros(num_bins)
    for pos in targets_pos:
        ego = (np.array([pos[0], pos[1], 0.0]) - robot_pos) @ robot_mat
        dist = np.hypot(ego[0], ego[1])
        angle = np.arctan2(ego[1], ego[0]) % (2 * np.pi)
        bin_ = int(angle / bin_size) % num_bins
        sensor = max(0.0, lidar.LIDAR_MAX_DIST - dist) / lidar.LIDAR_MAX_DIST
        alias = (angle - bin_size * bin_) / bin_size
        bin_plus, bin_minus = (bin_ + 1) % num_bins, (bin_ - 1) % num_bins
        obs[bin_] = max(obs[bin_], sensor)
        obs[bin_plus] = max(obs[bin_plus], alias * sensor)
        obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
    return obs


def _draw_until_valid(
    rng: np.random.Generator,
    keepout: float,
    other_xy: list[np.ndarray],
    other_keepout: list[float],
) -> np.ndarray:
    """Numpy version of `go_to_goal.draw_until_valid`."""
    xmin, ymin, xmax, ymax = go_to_goal.constrain_placement(
        go_to_goal._EXTENTS, keepout
    )
    for _ in range(_MAX_PLACEMENT_TRIES):
        xy = rng.uniform([xmin, ymin], [xmax, ymax])
        if all(
            np.linalg.norm(xy - other) >= keepout + other_keepout_
            for other, other_keepout_ in zip(other_xy, other_keepout)
        ):
            break
    return xy


def _sample_layout(
    rng: np.random.Generator, objects_spec: Mapping[str, go_to_goal.ObjectSpec]
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Numpy version of `go_to_goal._sample_layout`."""
    placements: list[np.ndarray] = []
    keepouts: list[float] = []
    layout: dict[str, list[tuple[int, np.ndarray]]] = {}
    for name, object_spec in objects_spec.items():
        layout[name] = []
        for _ in range(object_spec.num_objects):
            xy = _draw_until_valid(rng, object_spec.keepout, placements, keepouts)
            layout[name].append((len(placements), xy))
            placements.append(xy)
            keepouts.append(object_spec.keepout)
    return layout


class GoToGoalCPU:
    """Same model, observation and task reward as the (simulated) go to goal.

    The observation is the one of `GoToGoalObservationWrapper`. An episode
    never ends, unless the hard constraint on the distance to the obstacles is
    violated. In this case `terminated` is returned and the caller should reset.
    """

    def __init__(
        self,
        num_hazards: int = 8,
        num_vases: int = 1,
        goal_size: float = 0.3,
        recovery_distance: float = recovery.RECOVERY_DISTANCE,
        recovery_speed: float = recovery.RECOVERY_SPEED,
        hard_distance: float = recovery.HARD_DISTANCE,
        max_distance: float | None = recovery.MAX_DISTANCE,
        recovery_max_distance: float | None = recovery.RECOVERY_MAX_DISTANCE,
        stable_steering: bool = False,
        linear_thrust: bool = False,
        action_repeat: int = 1,
        real_params: Mapping[str, Any] | None = None,
        seed: int = 0,
    ):
        self._rng = np.random.default_rng(seed)
        self._goal_size = goal_size
        self._spec = go_to_goal.make_objects_spec(num_hazards, num_vases, goal_size)
        mj_spec = mujoco.MjSpec.from_file(filename=str(go_to_goal._XML_PATH), assets={})
        if stable_steering:
            go_to_goal.stabilize_steering(mj_spec)
        if linear_thrust:
            go_to_goal.linearize_thrust(mj_spec)
        go_to_goal.build_arena(
            mj_spec, layout=_sample_layout(self._rng, self._spec), goal_size=goal_size
        )
        self.model = mj_spec.compile()
        self._apply_real_params(real_params or {})
        self.data = mujoco.MjData(self.model)
        model = self.model
        self._action_repeat = action_repeat
        self._recovery_distance = recovery_distance
        self._recovery_speed = recovery_speed
        self._hard_distance = hard_distance
        self._max_distance = max_distance
        self._recovery_max_distance = recovery_max_distance
        self._robot_body_id = model.body("robot").id
        self._robot_geom_ids = {model.geom("robot").id, model.geom("pointarrow").id}
        self._robot_qpos_ids = go_to_goal._get_qpos_ids(model, ["x", "y", "z"])
        vases = [f"vase_{i}" for i in range(num_vases)]
        hazards = [f"hazard_{i}" for i in range(num_hazards)]
        self._vase_geom_ids = {model.geom(name).id for name in vases}
        self._vases_qpos_ids = [go_to_goal._get_qpos_ids(model, [n]) for n in vases]
        self._vases_dof_adr = [model.joint(name).dofadr[0] for name in vases]
        self._vase_body_ids = [model.body(name).id for name in vases]
        self._hazard_body_ids = [model.body(name).id for name in hazards]
        self._hazard_mocap_ids = [model.body(name).mocapid[0] for name in hazards]
        self._goal_mocap_id = model.body("goal").mocapid[0]
        self._last_goal_distance = 0.0
        self.observation_size = 2 * lidar.NUM_LIDAR_BINS + sum(
            len(_SENSOR_COMPONENTS[sensor]) for sensor in go_to_goal.BASE_SENSORS
        )
        self.action_size = model.nu
        self.dt = model.opt.timestep * _N_SUBSTEPS * action_repeat

    def _apply_real_params(self, params: Mapping[str, Any]) -> None:
        """Perturbs the nominal model, with the semantics of the sim randomization."""
        model = self.model
        damping = params.get("damping", {})
        for i, axis in enumerate("xyz"):
            model.dof_damping[i] *= damping.get(axis, 1.0)
        gear = params.get("gear", {})
        model.actuator_gear[0, 0] += gear.get("x", 0.0)
        model.actuator_gear[1, 0] += gear.get("z", 0.0)
        mass = params.get("mass", 1.0)
        model.body_mass[go_to_goal._ROBOT_ID] *= mass
        model.body_inertia[go_to_goal._ROBOT_ID] *= mass**3

    def reset(self) -> np.ndarray:
        layout = _sample_layout(self._rng, self._spec)
        mujoco.mj_resetData(self.model, self.data)
        data = self.data
        data.mocap_quat[:] = [1.0, 0.0, 0.0, 0.0]
        data.mocap_pos[self._goal_mocap_id] = [
            *layout["goal"][0][1],
            self._goal_size / 2.0 + 1e-2,
        ]
        for (_, xy), mocap_id in zip(layout["hazards"], self._hazard_mocap_ids):
            data.mocap_pos[mocap_id] = [*xy, 0.02]
        for (_, xy), qpos_ids in zip(layout["vases"], self._vases_qpos_ids):
            rotation = self._rng.uniform(-np.pi, np.pi)
            quat = [np.cos(rotation / 2), 0.0, 0.0, np.sin(rotation / 2)]
            data.qpos[qpos_ids] = [*xy, 0.1 - 1e-4, *quat]
        data.qpos[self._robot_qpos_ids] = [
            *layout["robot"][0][1],
            self._rng.uniform(-np.pi, np.pi),
        ]
        mujoco.mj_forward(self.model, data)
        self._lag_accelerometer()
        self._last_goal_distance = self._goal_distance()
        return self._get_obs()

    def set_state(
        self, qpos: np.ndarray, qvel: np.ndarray, mocap_pos: np.ndarray
    ) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.mocap_pos[:] = mocap_pos
        self.data.mocap_quat[:] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self._lag_accelerometer()
        self._last_goal_distance = self._goal_distance()
        return self._get_obs()

    def get_state(self) -> dict[str, np.ndarray]:
        """The state of the simulation, as taken by `set_state`."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "mocap_pos": self.data.mocap_pos.copy(),
        }

    def _lag_accelerometer(
        self, qvel: np.ndarray | None = None, qacc: np.ndarray | None = None
    ) -> None:
        """Evaluates the accelerometer as the simulation does.

        MJX computes the acceleration sensors of a step before solving for its
        accelerations, hence with the ones of the previous step. `qvel` and
        `qacc` are the ones before the last step, by default the state at rest.
        """
        data = self.data
        current_qvel, current_qacc = data.qvel.copy(), data.qacc.copy()
        data.qvel[:] = current_qvel if qvel is None else qvel
        data.qacc[:] = 0.0 if qacc is None else qacc
        mujoco.mj_rnePostConstraint(self.model, data)
        mujoco.mj_sensorAcc(self.model, data)
        data.qvel[:], data.qacc[:] = current_qvel, current_qacc

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        lower, upper = self.model.actuator_ctrlrange.T
        ctrl = (action + 1.0) / 2.0 * (upper - lower) + lower
        reward = 0.0
        cost = 0.0
        for _ in range(self._action_repeat):
            self.data.ctrl[:] = ctrl
            for _ in range(_N_SUBSTEPS):
                previous = self.data.qvel.copy(), self.data.qacc.copy()
                mujoco.mj_step(self.model, self.data)
            goal_distance = self._goal_distance()
            reward += self._last_goal_distance - goal_distance
            if goal_distance <= self._goal_size + 1e-2:
                reward += 1.0
                self._resample_goal()
                goal_distance = self._goal_distance()
            self._last_goal_distance = goal_distance
            cost += self._get_cost()
        self._lag_accelerometer(*previous)
        obs = self._get_obs()
        terminated = bool(
            recovery.violates(obs, self._hard_distance, self._max_distance)
        )
        info: dict[str, Any] = {
            "in_recovery_set": bool(
                recovery.in_recovery_set(
                    obs,
                    self._recovery_distance,
                    self._recovery_speed,
                    self._max_distance,
                    self._recovery_max_distance,
                )
            ),
            "cost": cost,
        }
        if self._max_distance is not None:
            info["out_of_arena"] = bool(recovery.out_of_arena(obs, self._max_distance))
        return obs, reward, terminated, info

    def _goal_distance(self) -> float:
        robot_to_goal = (
            self.data.mocap_pos[self._goal_mocap_id]
            - self.data.xpos[self._robot_body_id]
        )
        return float(np.linalg.norm(robot_to_goal[:2]))

    def _resample_goal(self) -> None:
        hazards_xy = self.data.xpos[self._hazard_body_ids][:, :2]
        vases_xy = self.data.xpos[self._vase_body_ids][:, :2]
        keepouts = [self._spec["hazards"].keepout] * len(hazards_xy)
        keepouts += [self._spec["vases"].keepout] * len(vases_xy)
        xy = _draw_until_valid(
            self._rng, self._spec["goal"].keepout, [*hazards_xy, *vases_xy], keepouts
        )
        self.data.mocap_pos[self._goal_mocap_id, :2] = xy

    def _get_cost(self) -> float:
        """Whether the robot is in a hazard, touches a vase, or a vase moves."""
        data = self.data
        for contact in data.contact[: data.ncon]:
            geoms = {contact.geom1, contact.geom2}
            touches = geoms & self._robot_geom_ids and geoms & self._vase_geom_ids
            if touches and contact.dist < 0:
                return 1.0
        robot_xy = data.xpos[self._robot_body_id][:2]
        hazards_xy = data.xpos[self._hazard_body_ids][:, :2]
        if (np.linalg.norm(hazards_xy - robot_xy, axis=1) <= _HAZARD_SIZE).any():
            return 1.0
        vases_velocity = [data.qvel[adr : adr + 3] for adr in self._vases_dof_adr]
        if vases_velocity and np.linalg.norm(vases_velocity) > 5e-2:
            return 1.0
        return 0.0

    def _get_obs(self) -> np.ndarray:
        data = self.data
        robot_pos = data.xpos[self._robot_body_id]
        robot_mat = data.xmat[self._robot_body_id].reshape(3, 3)
        obstacles = data.xpos[self._vase_body_ids + self._hazard_body_ids]
        goal = data.mocap_pos[[self._goal_mocap_id]]
        sensors = [
            data.sensor(sensor).data[_SENSOR_COMPONENTS[sensor]]
            for sensor in go_to_goal.BASE_SENSORS
        ]
        return np.hstack(
            [
                _compute_lidar(robot_pos, robot_mat, obstacles),
                _compute_lidar(robot_pos, robot_mat, goal),
                *sensors,
            ]
        ).astype(np.float32)
