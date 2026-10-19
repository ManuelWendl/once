# CyberRunner on the rig

How to run the CyberRunner policies trained in this repository on the real labyrinth: setting up
the rig's software, starting the hardware, evaluating a policy, testing the safe backup, and
exporting new policies from training runs. Everything below is run from a Mac connected to the
rig.

The rig runs Thomas Bi's CyberRunner ROS 2 nodes (camera, state estimator, Dynamixel motors),
ported to macOS in [cyberrunner-macos](https://github.com/BruceDLee/cyberrunner-macos). The
scripts in this folder use those nodes and Thomas's ONNX runner (`cyberrunner_dreamer.eval_onnx`),
replacing some of its methods when they are imported.

| File | What it does |
|---|---|
| `rig_eval.py` | Runs an ONNX policy on the maze given by `LAYOUT` (`easy`, `medium` or `hard`). Easy and medium run without the hard maze's goal assist; hard is Thomas's runner as it is. |
| `level_eval.py` | The episode start used in simulation: the board holds `WAIT_DEG` while the ball rolls into the start corner. `rig_eval.py` uses it when `WAIT_DEG` is set. |
| `shield_eval.py` | The safe backup (recovery prior): the backup alone, or a policy shielded by the backup. |
| `calib_record.py` | Records the still ball in the plate's corners, for the camera-to-plate calibration. |
| `calib_fit.py` | Fits that calibration from the recording. |
| `easy_plate.json` | The camera-to-plate calibration of the easy plate, used in all easy-maze tests (13–19 October 2026). |

## 1. One-time setup

Requirements: an Apple Silicon Mac, the Xcode command-line tools (`xcode-select --install`) and
[Miniforge](https://github.com/conda-forge/miniforge).

**The rig's software:**

```sh
git clone https://github.com/BruceDLee/cyberrunner-macos.git ~/cyberrunner-macos
cd ~/cyberrunner-macos
macos/setup.sh   # conda env in .conda and build of Thomas's ROS packages; no hardware needed
```

`setup.sh` writes absolute paths into `.conda` and `install`: if the folder moves, run it again.
It also deletes the saved markers, so redo the markers (below) after every run of `setup.sh`.
`macos/setup_dreamer.sh` is only needed for Thomas's own DreamerV3 policy, not for ONNX
policies.

**This repository:** clone it as usual. The scripts in this folder run in the rig's conda
environment, not in this repository's venv; `calib_fit.py` and the export and simulation tools
(sections 7 and 8) run in this repository's venv.

**Hardware:** the camera and the motors' USB adapter (U2D2) go into the Mac; the motor power
supply and the lamp go into the mains. The motors are Dynamixels with IDs 1 and 2 (the board's
two axes) and 3 (the lift that brings the ball back to the start).

**Markers** (the estimator finds the board through 8 blue dots):

```sh
cd ~/cyberrunner-macos
macos/run.sh markers
```

A camera window opens.
1. Level the board.
2. Click the **outer** dots, on the fixed enclosure: lower left, lower right, upper right, upper left.
3. Click the **inner** dots, at the board's corners, in the same order.
4. Click the **centre** of each dot. After the 8th click, click the window, press any key, then
   Ctrl-C in the Terminal.

Redo the markers when the rig, the camera or a dot has moved. Every dot must look round to the
camera: a dot partly hidden by the enclosure wall is rejected.

## 2. Before each session

- **Surface:** the rig must stand on an even, level surface.
- **Light:** switch the lamp on.
- **Vision check (no motors):** `cd ~/cyberrunner-macos && macos/run.sh vision`. In the
  detection window a mark must sit on all 8 dots and on the ball, and the image must not look
  bluish. The Terminal prints `ball: (x, y) | (a, b): (...) [deg]`: steady angles near 0 with the
  board level, a smooth ball position. `missing corner N` or `Did not find all outer markers`
  mean the markers or the colours need fixing. Stop with Ctrl-C.
- **White balance**, if the image is bluish or orange (in a Terminal with `source macos/env.sh`):

  ```sh
  python -m cyberrunner_camera.uvc_controls set white_balance_automatic 0
  python -m cyberrunner_camera.uvc_controls set white_balance_temperature 3000   # K; 3000 worked with the lamp
  ```

  Lower temperatures make the image bluer. The setting holds until the camera is unplugged.

## 3. Starting the hardware

Terminal 1, leave it running:

```sh
cd ~/cyberrunner-macos
macos/run.sh hw
```

This sets the motor adapter's USB latency to 1 ms and starts the camera, the motor node and the
estimator. It prints `Stack running. Ctrl-C to stop.` once they are up. The nodes' logs are in
`~/cyberrunner-macos/logs/<date-time>-hw/` (`camera.log`, `dynamixel.log`, `estimator.log`).

Terminal 2, for the scripts:

```sh
source ~/cyberrunner-macos/macos/env.sh
cd <this repository>/cyberrunner_experiments
```

The lift can be started and stopped by hand from Terminal 2:

```sh
ros2 service call /cyberrunner_dynamixel/start_reload std_srvs/srv/Trigger
ros2 service call /cyberrunner_dynamixel/stop_reload std_srvs/srv/Trigger
```

## 4. Policies

The scripts take ONNX files:

- **Behaviour policies** (one `.onnx` file): 50 inputs, the 30 values of the history (5 frames of
  tilts, ball position and command) and 20 values of the path ahead; 2 outputs, the commands.
- **Backups** (a folder with `backup_policy.onnx` and `backup_q.onnx`): 30 inputs, the history.
  `backup_q.onnx` gives Q_R(observation, action), the lowest of the critics.

The policies used in the tests so far:

| File | What it is | On the rig |
|---|---|---|
| `easy_cap_newball_thomas_priv_deadband_s31.onnx` | PPO, seed 31, best checkpoint | 19 goals in 24 episodes |
| `easy_sac_pq_s30_54M.onnx` | SAC (pessimistic Q), seed 30, 54M steps | 19 in 20 |
| `easy_sac_pq_ar001_s32_41M.onnx` | SAC with action-rate penalty 0.01, seed 32, 41M steps | 15 in 20 |
| `backup_nominal_s1_9.8M/` | recovery prior, seed 1, 9.8M steps, the best-calibrated one | see section 6 |

Filippo shares these files; put them in one folder, e.g. `~/cyberrunner-policies/`. New ones
are exported from training runs (section 7).

## 5. Evaluating a policy on the easy maze

The settings of all easy-maze tests:

```sh
P=~/cyberrunner-policies
LAYOUT=easy LEVEL_AT_START=0 CALIB=$PWD/easy_plate.json WAIT_DEG=-2.5,-0.75 START=148,222 START_MM=15 \
    python rig_eval.py --onnx $P/easy_cap_newball_thomas_priv_deadband_s31.onnx \
    --num-episodes 20 --output-dir runs/ppo_s31
```

An episode:
1. The board holds the tilt `WAIT_DEG` (alpha, beta in degrees) while the lift brings the ball
   back, so that the ball rolls into the start corner.
2. The episode starts once the ball has rested for 1 s within `START_MM` (mm) of `START` (x, y
   in mm). With `LEVEL_AT_START=0` it starts from that tilt, as in training. The script prints
   the start: angles, ball position, and `(NOT level)` if the board is off level.
3. The policy plays at 60 Hz until the goal (`SUCCESS`), until the ball disappears (it fell into
   a hole, or the camera lost it), or `--max-steps` (default 3000, 50 s).
4. Thomas's reset tilts the board and the lift brings the ball back for the next episode.

While it waits, the script prints every 5 s why the episode has not started (ball seen, at the
start, still). Every 30 s without a start, Thomas's runner shakes and resets the board.

| Variable | Meaning |
|---|---|
| `LAYOUT` | `easy`, `medium` or `hard` (required) |
| `CALIB` | camera-to-plate calibration (JSON); without it the camera's positions are used as they are |
| `WAIT_DEG` | tilt held while the ball arrives; without it, Thomas's own start |
| `START`, `START_MM` | start corner (mm) and how close to it the ball must rest |
| `LEVEL_AT_START` | `1` (default) levels the board before the start, `0` starts from the `WAIT_DEG` tilt |

**Outputs** in `--output-dir`: per episode `episode_NNN.npz` and `episode_NNN.csv` (time,
commands, tilts alpha and beta in rad, ball x and y in m), and `summary.json` (success rate,
episode lengths). To compare policies, alternate them a few episodes at a time (we ran turns of
5 episodes per policy), so that slow changes of the rig affect all of them alike.

**Other mazes:**

```sh
LAYOUT=medium START=170,223 WAIT_DEG=-2.5,-0.75 CALIB=$PWD/easy_plate.json \
    python rig_eval.py --onnx policy.onnx --num-episodes 10 --output-dir runs/medium
LAYOUT=hard python rig_eval.py --onnx policy.onnx --num-episodes 20 --output-dir runs/hard
```

## 6. Testing the safe backup (`shield_eval.py`)

`shield_eval.py` takes the same variables as `rig_eval.py`, plus `BACKUP` (the backup's folder)
and `MODE`:

| `MODE` | What acts |
|---|---|
| `shield` | the `--onnx` policy while Q_R(observation, its action) ≥ `TAU`; below, the backup takes over for `BACKUP_STEPS` steps (default 60, 1 s), then Q_R decides again |
| `behavior` | the `--onnx` policy alone, with Q_R logged: the baseline of `shield` |
| `backup` | the backup alone, from a thrown ball |
| `zero` | zero commands from a thrown ball: the baseline of `backup` |

A shielded policy, with the easy-maze settings of section 5:

```sh
P=~/cyberrunner-policies
LAYOUT=easy LEVEL_AT_START=0 CALIB=$PWD/easy_plate.json WAIT_DEG=-2.5,-0.75 START=148,222 START_MM=15 \
MODE=shield BACKUP=$P/backup_nominal_s1_9.8M TAU=0.85 \
    python shield_eval.py --onnx $P/easy_cap_newball_thomas_priv_deadband_s31.onnx \
    --num-episodes 10 --output-dir runs/shield_s31
```

The backup alone, from a thrown ball (without `WAIT_DEG`: the ball does not start in the corner):

```sh
LAYOUT=easy CALIB=$PWD/easy_plate.json MODE=backup BACKUP=$P/backup_nominal_s1_9.8M \
    python shield_eval.py --onnx $P/backup_nominal_s1_9.8M/backup_policy.onnx \
    --num-episodes 10 --max-steps 300 --num-wait-steps 1 --output-dir runs/backup_alone
```

After each episode, pick the ball up, keep it out of the camera's view for 0.5 s, then throw it
onto the board: the episode starts at the first frame the ball is seen more than `START_MM`
(default 30 mm) from `START` (default 154, 224 mm), so balls brought back by the lift are
skipped.

Besides the runner's own files, the script writes `<output-dir>/shield/<date-time>/episode_NNN.npz`,
row by row with the runner's `episode_NNN.npz`: the command sent, the policy's command, Q_R,
whether the backup acted, and the number of its takeover in the episode (0: none).

What we saw so far (19 October 2026): shielding PPO seed 31, the backup took control 27–93% of
the time and the falls happened while the backup was in control; alone, it stopped 11 of 12
thrown balls.

## 7. Exporting policies from training runs

In this repository's venv, from a training run's Hydra directory:

```sh
python utility_scripts/cyberrunner_export_onnx.py RUN_DIR policy.onnx                # behaviour policy
python utility_scripts/cyberrunner_export_onnx.py RUN_DIR backup_dir/ --recovery     # backup and Q_R
```

`--checkpoint best` (default) or `last`. For a recovery prior, "best" is the checkpoint chosen by
the calibration (`utility_scripts/cyberrunner_recovery_calibration.py`).

## 8. Simulation tools in this repository

- `ss2r/configs/experiment/cyberrunner_recovery_prior.yaml`: the recovery prior (SAC with SPiDR
  and the pessimistic recovery reward); `cyberrunner_recovery_prior_adversary.yaml`: the same,
  trained against a handover adversary; `cyberrunner_ppo.yaml`: PPO behaviour policies.
- `utility_scripts/cyberrunner_eval.py RUN_DIR ... --out results.jsonl`: behaviour policies on
  fixed simulated conditions.
- `utility_scripts/cyberrunner_recovery_calibration.py RUN_DIR --behavior PPO_RUN_DIR
  --near-holes 25 13.8 --plot --out DIR`: Q_R of every checkpoint against the backup's outcome
  in simulation, on the prior's own starts and on states of a behaviour policy.

## 9. Plate calibration

The camera sees the ball on the raised plate slightly off from where it is on the board.
`easy_plate.json` corrects for it (a scale about the camera's nadir, fitted on easy-maze
trajectories). Redo it if the camera or the rig moved, with the stack running:

```sh
python calib_record.py calib.npz             # rig environment: put the still ball in each corner for about 5 s
python calib_fit.py calib.npz new_plate.json  # this repository's venv
```

then use `CALIB=new_plate.json`.

## 10. Stopping

1. Ctrl-C in Terminal 2 stops the script; Ctrl-C in Terminal 1 stops the lift and all nodes.
2. Check that nothing is left: `pgrep -fl ~/cyberrunner-macos/install` must print nothing.
3. The board's motors keep their last command. To switch the torque of all three motors off
   (or switch off the motor power):

   ```sh
   cd ~/cyberrunner-macos
   .conda/bin/python -c "
   import glob
   from dynamixel_sdk import PortHandler, PacketHandler
   p = PortHandler(glob.glob('/dev/cu.usbserial-*')[0]); ph = PacketHandler(2.0)
   p.openPort(); p.setBaudRate(2000000)
   [ph.write1ByteTxRx(p, i, 64, 0) for i in (1, 2, 3)]; p.closePort(); print('torque off')"
   ```

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The ball falls into the first hole right away, or the ball position jumps | Markers, colours or light: run the vision check (section 2) |
| `missing corner`, `Did not find all outer markers` | A dot is partly hidden or not blue enough: make it fully visible, adjust the white balance, redo the markers |
| The motors stop, `TxRxResult` or `failed to read/write` in `dynamixel.log` | A motor stopped answering, most often the lift (ID 3), and all motors lose torque: stop and restart the stack, then restart the lift with `start_reload` (section 3) |
| The motors do nothing after their power was switched off and on | Torque is off after a power cycle: restart the stack |
| The lift keeps running after a crash | `stop_reload` (section 3), or the torque-off command (section 10) |
| Episodes end early with the ball near the top-right edge of the board (around x 220–260 mm, y 225 mm) | The camera loses the ball there: improve the light on that edge |
| The episode never starts | The script prints why every 5 s: the ball must rest within `START_MM` of `START` |
| `(NOT level)` at the start | Expected with `LEVEL_AT_START=0`; otherwise check the surface and the markers |
| Camera at 120 Hz or odd timing | Two camera nodes are running: stop everything and check with `pgrep` |
| The wrong camera (e.g. the Mac's own) | `CYBERRUNNER_CAMERA=1 macos/run.sh hw` |
| The motor adapter is not found | `CYBERRUNNER_PORT=/dev/cu.usbserial-XXXX macos/run.sh hw` |
