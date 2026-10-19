"""Physical parameters of the CyberRunner robot, from system identification."""

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
