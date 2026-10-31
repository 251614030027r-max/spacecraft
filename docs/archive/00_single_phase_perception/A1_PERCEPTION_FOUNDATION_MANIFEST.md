# A1 perception foundation manifest

Date: 2026-09-02
Branch: `claude/sac-mpc-coupling-design-ns7g6i`
Base: `43caf4dacacdbd8f3706dda621387a01d4828dd3`

## Single factor

Add an optional geometric camera and relative-state EKF observation path to the
frozen S1-v2 `single_phase` task. `perception=None` preserves the original path;
perception mode changes observation generation only. Truth remains the source
for RK45 dynamics, reward, termination, constraints and evaluation errors.

## Fixed specification

- Five target-fixed features: four at `x=-1.50 m`, one non-coplanar feature at
  `(-1.70,+0.18,-0.10) m`.
- Chaser +x camera: 1024x1024 px, `(cu,cv)=(512,512) px`, `f=430 px`, 50 deg
  half-FOV, 30 m range, 1 px independent Gaussian coordinate noise, 10 Hz.
- Visibility: positive depth, image bounds, cone, range and
  `(-1,0,0) dot (p_camera_T-p_feature_T) > 0` front-face test.
- EKF local error: `[dtheta, dp, domega, dv]`; initial standard deviations
  `5 deg, 0.50 m, 0.005 rad/s, 0.05 m/s`; per-step process standard deviations
  `2e-4 rad, 1e-3 m, 1e-4 rad/s, 5e-4 m/s`.
- Observation schema: `phase2_perception_v1_29d`, comprising an estimated 24D
  core, four `softsign(log(block_rms_std/P0_std))` values and visible fraction.

## Implementation and tests

- Camera: `env/perception.py`; EKF: `estimation/relative_ekf.py`.
- Integration: `env/se3_rendezvous_env.py`, `env/observation.py`,
  `env/phase2_env.py`.
- Aggregate tests: camera geometry/visibility, deterministic 20 s estimator
  chain, and disabled-path/schema compatibility.
- Full acceptance: `149 passed in 57.51s` using the verified Python 3.12.13
  runtime with `.venv/Lib/site-packages`.

## Smoke acceptance

Seed `260902`, existing scripted controller, perception configuration, 300 steps
and 30 s cap: observation shape 29; 301/301 frames used measurements; visible
count remained 5; all estimate/covariance diagnostics were finite; no early
termination and the cap produced normal truncation. Final diagnostic errors were
0.05745 rad attitude, 0.85088 m position, 0.01188 rad/s angular velocity and
0.21387 m/s velocity. These are smoke diagnostics, not formal performance
claims.

No training, A2 task, reward change, guidance removal, window, obstacle or
discrete-thruster model is included in this stage.
