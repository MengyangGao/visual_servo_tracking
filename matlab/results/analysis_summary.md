# Analysis Summary

This file records the latest numeric results from the MATLAB demo after the visual and repository cleanup.

## Simulation

| Task | Metric | Value |
|---|---:|---:|
| T1 calibration | Mean reprojection error | 0.4226 px |
| T2 fixed-camera | Final position error | 0.000198 m |
| T2 fixed-camera | RMSE position error | 0.000294 m |
| T2 eye-in-hand | Final position error | 0.000347 m |
| T2 eye-in-hand | RMSE position error | 0.000441 m |
| T3 IBVS | Final feature error | 0.013326 |
| T3 IBVS | RMSE feature error | 0.159721 |

## Real Camera

| Demo | Samples | Notes |
|---|---:|---|
| Follow | 24 | Reused the saved `cameraParams.mat` and the printed ChArUco board |
| IBVS | 24 | Reused the saved `cameraParams.mat` and the same board geometry |

## Repository Policy

- PNG, MP4, and PDF artifacts are generated locally only.
- Raw calibration photos are not tracked in Git.
- Text logs remain available locally for inspection, but the public repository keeps the source tree compact.

## Reference Logs

The latest local run produced text logs under:
- `results/run_log.txt`
- `results/t2_real_camera_follow/real_camera_follow_log.txt`
- `results/t3_real_camera_ibvs/real_camera_ibvs_log.txt`
