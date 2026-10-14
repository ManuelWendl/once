"""MuJoCo model of the CyberRunner board, marble and tilt motors."""

from typing import Optional, Tuple

import mujoco
import numpy as np

from ss2r.benchmark_suites.mujoco_playground.cyberrunner.layout import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    Layout,
)

# ============================================================================
# PHYSICS PARAMETERS (from system identification)
# ============================================================================

# Actuator parameters
GEAR_ALPHA = 0.6378096703815441
GEAR_BETA = 1.1503276919996794
DYNPRM_TAU_ALPHA = 0.017652846705560145
DYNPRM_TAU_BETA = 0.0381459302540759
# Larger commands add no speed on the robot (motor PWM limit), fitted on rig data:
# alpha, beta.
CTRL_LIMIT = (0.45, 0.55)
# DC motor (env option motor_dc, replaces the cap): the tilt motors (XL330-M077 at 5 V) are
# current controlled under the driver's voltage (PWM) limit, so the motor current (filter
# state, command units) is bounded by DC_CURRENT_LIMIT * (+-1 - speed / DC_NO_LOAD_SPEED).
# Command at which the tilt motors reach their PWM limit at standstill, measured on the rig
# from the motors' own PWM; alpha, beta.
DC_CURRENT_LIMIT = (0.92, 0.93)
# Board torque per command (N m) and board speed (rad/s) at which the back-EMF uses up the
# voltage limit; alpha, beta. Fitted, with the sliding friction, on half-second replays of
# hard-maze rig policy episodes.
DC_GEAR = (0.813, 1.022)
DC_NO_LOAD_SPEED = (1.63, 1.04)
# Joint dry friction with motor_dc: the breakaway torque (N m; the rig's dead band times the
# gear), and the sliding torque as a fraction of it, reached above the Stribeck speed (rad/s).
STATIC_FRICTION = (0.168, 0.178)
SLIDING_FRACTION = (0.74, 0.14)
STRIBECK_SPEED = 0.01
# Stiffer joint dry friction (constraint time constant, s): with MuJoCo's default 0.02 a board
# held below the breakaway torque creeps at ~0.3 deg/s, enough to start the sliding level.
FRICTION_TIMECONST = 0.005

# Joint parameters (angles in radians)
RANGE_ALPHA = (-0.15847916128302914, 0.15847916128302914)  # ~±9.08°
RANGE_BETA = (-0.10424974775885551, 0.10424974775885551)  # ~±5.97°
DAMPING_ALPHA = 0.13716407911582046
DAMPING_BETA = 0.4263556183589426
FRICTIONLOSS_ALPHA = 0.0496853803940722
FRICTIONLOSS_BETA = 0.09905565254081455

# Board parameters
BOARD_MASS = 1.5
BOARD_INERTIA = (0.021, 0.021, 0.021)

# Marble parameters
MARBLE_MASS = 0.009
MARBLE_RADIUS = 0.0063
MARBLE_FRICTION = (0.250129382, 0.07549734, 0.00275288)  # slide, spin, roll
MARBLE_SOLREF = (0.02, 1.25)  # timeconst, dampingratio
# Ball stiction (env option ball_stiction): a still ball starts rolling above this board
# tilt, and rolls against a resistance equal to this tilt; median of the rig's ramps.
BALL_STATIC_DEG = 1.3
BALL_ROLLING_DEG = 0.6

WALL_RADIUS = 0.0025
HOLE_RADIUS = 0.0075

# Simulation
TIMESTEP = 0.00166666666  # 600Hz physics
FRAME_SKIP = 10  # 60Hz control

# MJX checks the ball only against this many nearest boxes and cylinders.
MAX_GEOM_PAIRS = 10


def check_endpoint_connected(
    point_x: float,
    point_y: float,
    walls_h: np.ndarray,
    walls_v: np.ndarray,
    is_vertical_wall: bool,
    wall_index: int,
    tol: float = 0.003,
) -> bool:
    """Check if a wall endpoint is connected to another wall or board edge."""
    # Check board edges
    if (
        abs(point_x) < tol
        or abs(point_x - BOARD_WIDTH) < tol
        or abs(point_y) < tol
        or abs(point_y - BOARD_HEIGHT) < tol
    ):
        return True

    if is_vertical_wall:
        # Check connection to horizontal walls
        for h_wall in walls_h:
            x_start, x_end, y = h_wall
            if abs(y - point_y) < tol:
                if x_start - tol <= point_x <= x_end + tol:
                    return True
        # Check connection to other vertical walls
        for i, v_wall in enumerate(walls_v):
            if i == wall_index:
                continue
            y_start, y_end, x = v_wall
            if abs(x - point_x) < tol:
                if abs(y_start - point_y) < tol or abs(y_end - point_y) < tol:
                    return True
    else:
        # Check connection to vertical walls
        for v_wall in walls_v:
            y_start, y_end, x = v_wall
            if abs(x - point_x) < tol:
                if y_start - tol <= point_y <= y_end + tol:
                    return True
        # Check connection to other horizontal walls
        for i, h_wall in enumerate(walls_h):
            if i == wall_index:
                continue
            x_start, x_end, y = h_wall
            if abs(y - point_y) < tol:
                if abs(x_start - point_x) < tol or abs(x_end - point_x) < tol:
                    return True

    return False


def build_model(
    layout: Layout,
    max_geom_pairs: Optional[int] = MAX_GEOM_PAIRS,
    ball_stiction: bool = False,
    ctrl_limit: Tuple[float, float] = CTRL_LIMIT,
    motor_dc: bool = False,
) -> mujoco.MjModel:
    """Build the MuJoCo model using mjSpec; ctrl_limit caps the alpha and beta commands,
    unless motor_dc bounds the motor current instead (in the env)."""
    walls_h, walls_v, waypoints = layout.walls_h, layout.walls_v, layout.waypoints
    spec = mujoco.MjSpec()
    spec.modelname = "cyberrunner"
    spec.compiler.autolimits = True
    spec.option.timestep = TIMESTEP

    world = spec.worldbody

    # Floor (visual only)
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10, 10, 10]
    floor.rgba = [0.8, 0.9, 0.8, 1]
    floor.contype = 0
    floor.conaffinity = 0

    # Light
    light = world.add_light()
    light.pos = [0, 0, 1.3]
    light.dir = [0, 0, -1.3]

    # Base (static)
    base = world.add_body()
    base.name = "base"
    base_geom = base.add_geom()
    base_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    base_geom.size = [0.15, 0.15, 0.01]
    base_geom.pos = [0.138, 0.1155, 0.0]
    base_geom.rgba = [1.0, 1.0, 1.0, 1.0]
    base_geom.contype = 0
    base_geom.conaffinity = 0

    # Link body (alpha joint - rotation around Y axis)
    link = world.add_body()
    link.name = "link"
    link.pos = [0, 0, 0]

    link_joint = link.add_joint()
    link_joint.name = "alpha_joint"
    link_joint.type = mujoco.mjtJoint.mjJNT_HINGE
    link_joint.pos = [0.138, 0.1155, 0.0835]
    link_joint.axis = [0, -1, 0]
    link_joint.range = [np.degrees(RANGE_ALPHA[0]), np.degrees(RANGE_ALPHA[1])]
    link_joint.limited = True
    link_joint.damping = DAMPING_ALPHA
    link_joint.frictionloss = FRICTIONLOSS_ALPHA

    link_geom = link.add_geom()
    link_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    link_geom.size = [0.02, 0.02, 0.001]
    link_geom.pos = [0.138, 0.1155, 0.0]
    link_geom.rgba = [0.5, 0.5, 0.5, 1.0]
    link_geom.contype = 0
    link_geom.conaffinity = 0
    link_geom.mass = 0.05

    # Board body (beta joint - rotation around X axis)
    board = link.add_body()
    board.name = "board"
    board.pos = [0, 0, 0]

    board_joint = board.add_joint()
    board_joint.name = "beta_joint"
    board_joint.type = mujoco.mjtJoint.mjJNT_HINGE
    board_joint.pos = [0.138, 0.1155, 0.0835]
    board_joint.axis = [1, 0, 0]
    board_joint.range = [np.degrees(RANGE_BETA[0]), np.degrees(RANGE_BETA[1])]
    board_joint.limited = True
    board_joint.damping = DAMPING_BETA
    board_joint.frictionloss = FRICTIONLOSS_BETA

    # Board inertial properties
    board.ipos = [0.138, 0.1155, 0.0835]
    board.mass = BOARD_MASS
    board.inertia = list(BOARD_INERTIA)

    # Board edges
    _add_board_edges(board)

    # Board floor
    floor_geom = board.add_geom()
    floor_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    floor_geom.size = [0.145, 0.1225, 0.0025]
    floor_geom.pos = [0.138, 0.1155, 0.0705]
    floor_geom.contype = 2
    floor_geom.conaffinity = 0

    # Maze walls
    _add_maze_walls(board, walls_h, walls_v)

    # Actuators
    _add_actuators(spec, (1.0, 1.0) if motor_dc else ctrl_limit)

    # Marble
    _add_marble(world, waypoints[0], ball_stiction)

    if max_geom_pairs is not None:
        numeric = spec.add_numeric()
        numeric.name = "max_geom_pairs"
        numeric.data = [max_geom_pairs]

    model = spec.compile()
    if motor_dc:
        _set_dc_motors(model)
    return model


def _set_dc_motors(model: mujoco.MjModel):
    """Breakaway joint friction, the DC gear, and the DC motor parameters in the fixed gain's
    unused slots (read by the env and randomized with the motors): current limit, no-load
    speed, sliding fraction, Stribeck speed."""
    for axis, name in enumerate(("alpha", "beta")):
        dof = model.jnt_dofadr[model.joint(f"{name}_joint").id]
        act = model.actuator(f"{name}_motor").id
        model.dof_frictionloss[dof] = STATIC_FRICTION[axis]
        model.dof_solref[dof, 0] = FRICTION_TIMECONST
        model.actuator_gear[act, 0] = DC_GEAR[axis]
        model.actuator_gainprm[act, 1:5] = (
            DC_CURRENT_LIMIT[axis],
            DC_NO_LOAD_SPEED[axis],
            SLIDING_FRACTION[axis],
            STRIBECK_SPEED,
        )


def _add_board_edges(board):
    """Add collision boxes for board edges."""
    edges = [
        ([0.138, 0.2345, 0.0835], [0.145, 0.0035, 0.0105]),  # Top
        ([0.138, -0.0035, 0.0835], [0.145, 0.0035, 0.0105]),  # Bottom
        ([-0.0035, 0.1155, 0.0835], [0.0035, 0.1155, 0.0105]),  # Left
        ([0.2795, 0.1155, 0.0835], [0.0035, 0.1155, 0.0105]),  # Right
    ]
    for pos, size in edges:
        edge = board.add_geom()
        edge.type = mujoco.mjtGeom.mjGEOM_BOX
        edge.size = size
        edge.pos = pos
        edge.contype = 2
        edge.conaffinity = 0


def _add_maze_walls(board, walls_h: np.ndarray, walls_v: np.ndarray):
    """Add maze walls with rounded edges."""
    wall_r = WALL_RADIUS
    visual_wall_r = max(wall_r, 0.0025)
    wall_height = 0.0075
    z_pos = 0.073 + wall_height
    wall_rgba = [0.9, 0.3, 0.1, 1.0]

    # Vertical walls
    for i, wall in enumerate(walls_v):
        y_start, y_end, x = float(wall[0]), float(wall[1]), float(wall[2])

        needs_cap_start = not check_endpoint_connected(
            x, y_start, walls_h, walls_v, True, i
        )
        needs_cap_end = not check_endpoint_connected(
            x, y_end, walls_h, walls_v, True, i
        )

        box_y_start = (y_start + visual_wall_r) if needs_cap_start else y_start
        box_y_end = (y_end - visual_wall_r) if needs_cap_end else y_end
        box_length = box_y_end - box_y_start

        if box_length > 0:
            box = board.add_geom()
            box.name = f"vwall{i}"
            box.type = mujoco.mjtGeom.mjGEOM_BOX
            box.pos = [x, (box_y_start + box_y_end) / 2.0, z_pos]
            box.size = [visual_wall_r, box_length / 2.0, wall_height]
            box.rgba = wall_rgba
            box.contype = 2
            box.conaffinity = 0

        if needs_cap_start:
            cap = board.add_geom()
            cap.name = f"vwall{i}_cap_s"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x, y_start + visual_wall_r, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0

        if needs_cap_end:
            cap = board.add_geom()
            cap.name = f"vwall{i}_cap_e"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x, y_end - visual_wall_r, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0

    # Horizontal walls
    for i, wall in enumerate(walls_h):
        x_start, x_end, y = float(wall[0]), float(wall[1]), float(wall[2])

        needs_cap_start = not check_endpoint_connected(
            x_start, y, walls_h, walls_v, False, i
        )
        needs_cap_end = not check_endpoint_connected(
            x_end, y, walls_h, walls_v, False, i
        )

        box_x_start = (x_start + visual_wall_r) if needs_cap_start else x_start
        box_x_end = (x_end - visual_wall_r) if needs_cap_end else x_end
        box_length = box_x_end - box_x_start

        if box_length > 0:
            box = board.add_geom()
            box.name = f"hwall{i}"
            box.type = mujoco.mjtGeom.mjGEOM_BOX
            box.pos = [(box_x_start + box_x_end) / 2.0, y, z_pos]
            box.size = [box_length / 2.0, visual_wall_r, wall_height]
            box.rgba = wall_rgba
            box.contype = 2
            box.conaffinity = 0

        if needs_cap_start:
            cap = board.add_geom()
            cap.name = f"hwall{i}_cap_s"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x_start + visual_wall_r, y, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0

        if needs_cap_end:
            cap = board.add_geom()
            cap.name = f"hwall{i}_cap_e"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x_end - visual_wall_r, y, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0


def _add_actuators(spec, ctrl_limit: Tuple[float, float]):
    """Add motor actuators for the two joints."""
    limit_alpha, limit_beta = ctrl_limit
    # Alpha actuator
    act_alpha = spec.add_actuator()
    act_alpha.name = "alpha_motor"
    act_alpha.target = "alpha_joint"
    act_alpha.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act_alpha.gear = [GEAR_ALPHA, 0, 0, 0, 0, 0]
    act_alpha.ctrlrange = [-limit_alpha, limit_alpha]
    act_alpha.ctrllimited = True
    act_alpha.dyntype = mujoco.mjtDyn.mjDYN_FILTER
    act_alpha.dynprm = [DYNPRM_TAU_ALPHA, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Beta actuator
    act_beta = spec.add_actuator()
    act_beta.name = "beta_motor"
    act_beta.target = "beta_joint"
    act_beta.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act_beta.gear = [GEAR_BETA, 0, 0, 0, 0, 0]
    act_beta.ctrlrange = [-limit_beta, limit_beta]
    act_beta.ctrllimited = True
    act_beta.dyntype = mujoco.mjtDyn.mjDYN_FILTER
    act_beta.dynprm = [DYNPRM_TAU_BETA, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _add_marble(world, start_pos: np.ndarray, stiction: bool = False):
    """Add the marble to the world."""
    marble = world.add_body()
    marble.name = "marble"
    marble.pos = [start_pos[0], start_pos[1], 0.0793]

    # Inertial properties
    marble.ipos = [0, 0, 0]
    marble.mass = MARBLE_MASS
    inertia_val = 2.0 / 5.0 * MARBLE_MASS * MARBLE_RADIUS**2
    marble.inertia = [inertia_val, inertia_val, inertia_val]

    # Free joint
    joint = marble.add_joint()
    joint.type = mujoco.mjtJoint.mjJNT_FREE

    # Sphere geometry
    geom = marble.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [MARBLE_RADIUS, MARBLE_RADIUS, MARBLE_RADIUS]
    geom.rgba = [0.0, 0.4, 1.0, 1.0]
    geom.contype = 4
    geom.conaffinity = 7
    # The condim-3 contact uses only the sliding friction; the other two slots hold the
    # env's ball stiction coefficients (zero without it).
    degrees = (BALL_STATIC_DEG, BALL_ROLLING_DEG) if stiction else (0.0, 0.0)
    geom.friction = [MARBLE_FRICTION[0], *np.tan(np.radians(degrees))]
    geom.priority = 1
    geom.solref = list(MARBLE_SOLREF)
