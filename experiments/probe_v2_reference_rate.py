"""What does the V2 rate limit actually permit the reference to do?

V2 caps the two task axes separately: 0.5 m per decision on distance, 0.05 per
decision on commitment. The distance cap is in metres and binds. The commitment
cap is dimensionless, but its effect on the commanded point is a metric
displacement -- the direction slerps, so the reference sweeps an arc of about
``radius * angle(hold, capture) * 0.05``. At the 12-19 m staging radii that is
1-3 m per decision, and nothing bounds it.

The reason that matters is recorded in env/hybrid_env.py: every V1 episode that
completed held the commanded waypoint to a median 0.000-0.015 m per decision at
0.20-0.28 actuator usage, and every episode that timed out moved it 0.27-0.62 m
at 0.84-0.95. So a cap that permits 1-3 m per decision sits well above the band
this task associates with running the clock out.

This probe drives both axes to their far end and measures the reference
displacement the limiter actually allows. It is a limiter measurement, not a
performance arm: an absolute proposal pinned at maximum is adversarial by
construction and its completion count means nothing.
"""
import numpy as np
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config

rows = []
for seed in (262000, 262001, 262005, 262011):
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="task_state_v2",
            runtime_diagnostics=False,
            include_staging_direction_observation=True,
        ),
    )
    env.reset(seed=seed)
    previous = None
    motions, usages = [], []
    terminated = truncated = False
    decisions = 0
    # Absolute proposal at the far end of both axes: always ask for maximum
    # advance, so the limiter is the only thing deciding the step.
    action = np.array([1.0, 1.0])
    while not (terminated or truncated):
        waypoint = np.asarray(env.waypoint_from_action(action), dtype=np.float64)
        if previous is not None:
            motions.append(float(np.linalg.norm(waypoint - previous)))
        previous = waypoint
        _, _, terminated, truncated, info = env.step(action)
        decisions += 1
        usages.append(float(info.get("mean_actuator_usage", float("nan"))))
    completed = bool(info.get("completed", False))
    m = np.array(motions) if motions else np.array([np.nan])
    finite = [v for v in usages if np.isfinite(v)]
    u = np.array(finite) if finite else np.array([np.nan])
    rows.append((seed, completed, decisions, float(env.env.time_seconds),
                 float(np.median(m)), float(np.max(m)),
                 float(np.mean(u)) if len(u) else float("nan")))

print(f'{"seed":<8}{"ended":>8}{"decisions":>11}{"time_s":>9}'
      f'{"ref_move_median_m":>20}{"ref_move_max_m":>16}')
for r in rows:
    print(f'{r[0]:<8}{str(r[1]):>8}{r[2]:>11}{r[3]:>9.1f}{r[4]:>20.3f}{r[5]:>16.3f}')
print()
print("V1 reference, from env/hybrid_env.py: completing episodes held the")
print("waypoint to 0.000-0.015 m per decision at 0.20-0.28 actuator usage;")
print("timing-out episodes moved it 0.27-0.62 m at 0.84-0.95.")
