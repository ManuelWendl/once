# CyberRunner on the rig

Scripts that run the CyberRunner policies trained here on the real labyrinth. They drive the
rig through its ROS 2 stack (camera, state estimator, Dynamixel motors: Thomas's CyberRunner
with the macOS port, `cyberrunner-macos`) and build on Thomas's ONNX runner
(`cyberrunner_dreamer.eval_onnx`), replacing some of its methods when they are imported.

They run in the rig's environment, not in this repository's venv (except `calib_fit.py`):

```bash
macos/run.sh hw        # in cyberrunner-macos: camera, estimator and motors; leave it running
source macos/env.sh    # in a second terminal, then cd to this folder
```

| File | What it does |
|---|---|
| `rig_eval.py` | Thomas's runner on the maze given by `LAYOUT` (`easy`, `medium` or `hard`, required). Easy and medium run without the hard maze's goal assist; hard is the runner as it is. |
| `level_eval.py` | The start used in the sim: the board holds `WAIT_DEG` while the ball rolls into the start corner, then a P+I loop levels it. `rig_eval.py` uses it when `WAIT_DEG` is set. |
| `shield_eval.py` | The safe backup (a recovery prior exported with `utility_scripts/cyberrunner_export_onnx.py --recovery`): `MODE=backup`, `zero`, `shield` or `behavior`. |
| `calib_record.py` | Records the still ball in the plate's corners, for the camera-to-plate calibration. |
| `calib_fit.py` | Fits that calibration from the recording (in this repository's venv); `rig_eval.py` applies it with `CALIB=file`. |

```bash
LAYOUT=easy WAIT_DEG=-3,-1 python rig_eval.py --onnx policy.onnx --num-episodes 20 --output-dir runs/easy
LAYOUT=medium START=170,223 WAIT_DEG=-2.5,-0.75 CALIB=easy_plate.json \
    python rig_eval.py --onnx policy.onnx --num-episodes 10 --output-dir runs/medium
LAYOUT=hard python rig_eval.py --onnx policy.onnx --num-episodes 20 --output-dir runs/hard
LAYOUT=easy MODE=shield BACKUP=backup/ TAU=0.9 WAIT_DEG=-3,-1 \
    python shield_eval.py --onnx policy.onnx --num-episodes 20 --output-dir runs/shield

python calib_record.py calib.npz                                      # rig environment
python cyberrunner_experiments/calib_fit.py calib.npz easy_plate.json  # this repository's venv
```

The other settings are environment variables, described in each script's docstring: `START`,
`START_MM` and `LEVEL_AT_START` (`level_eval.py`), `TAU` and `BACKUP` (`shield_eval.py`). The
runner saves `episode_NNN.npz`, `episode_NNN.csv` and `summary.json` under `--output-dir`.
