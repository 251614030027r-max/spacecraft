"""Is the upper-to-lower channel live for a whole episode, or does it go inert?

This is the check round 1 did not have. V1 passed every test it had and still
produced 0.114% and 0.484% effective intervention after 60,000 decisions per
seed, because the failure was dynamic: the commit ratchet closed on decision
zero and the remaining ~40 decisions could not change the reference at all. No
static mapping test catches that, and neither return nor completion rate showed
it -- both looked healthy while the capability went to zero.

What does catch it: drive the interface with a *random* policy, the crudest
stand-in for an untrained one, and measure over the decision sequence what
fraction of decisions actually change the task reference, whether that fraction
holds up in the second half, and how many distinct references the run produces.
No training and no model.

Preregistered pass criteria, all three required:

1. the whole-episode changed fraction is well above zero;
2. the second half is not markedly below the first (V1's disease is "dies as it
   goes");
3. the reference spans a radius range rather than collapsing to a point.

Measured 2026-09-22 over seeds 262000 / 262001 / 262005, 40 decisions each:

    arrival_condition (V1)   changed 0.026 / 0.025 / 0.050
                             second half 0.000 / 0.000 / 0.000
                             distinct references 2 / 2 / 3 of 40
    task_state_v2 (V2)       changed 1.000 / 1.000 / 1.000
                             second half 1.000 / 1.000 / 1.000
                             distinct references 40 / 40 / 40

V1 fails all three, V2 passes all three. This says the channel carries the
policy's output for a whole episode; it says nothing about whether a policy can
learn to use it, which only training answers.

Run before any training round on a changed interface. It costs minutes.
"""
import numpy as np
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


def run(parametrization, seed, rng):
    cfg = dict(
        horizon_steps=35,
        waypoint_parametrization=parametrization,
        runtime_diagnostics=False,
        include_staging_direction_observation=True,
    )
    if parametrization == "arrival_condition":
        cfg["baseline_anchored_residual"] = True   # V1 as trained
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(**cfg),
    )
    env.reset(seed=seed)
    desired = np.asarray(
        env.environment_config.precapture_task.desired_position, dtype=np.float64
    )
    changed, total, first_half, second_half = 0, 0, 0, 0
    refs = []
    terminated = truncated = False
    # 40 decisions is 80 s of mission time. V1 goes inert on decision 1, so the
    # collapse pattern and the first-versus-second-half comparison are both
    # visible well inside that; running to the 300 s cap only costs solves.
    while not (terminated or truncated) and total < 40:
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
        _, _, terminated, truncated, info = env.step(action)
        ref = np.asarray(info["hybrid_waypoint_target_frame"], dtype=np.float64)
        refs.append(ref)
        is_changed = float(np.linalg.norm(ref - desired)) > 1.0e-9
        changed += int(is_changed)
        total += 1
    half = max(total // 2, 1)
    for i, r in enumerate(refs):
        if float(np.linalg.norm(r - desired)) > 1.0e-9:
            if i < half:
                first_half += 1
            else:
                second_half += 1
    radii = np.linalg.norm(np.array(refs), axis=1)
    return dict(
        decisions=total,
        changed_frac=changed / total,
        first_half_frac=first_half / half,
        second_half_frac=second_half / max(total - half, 1),
        radius_min=float(radii.min()),
        radius_max=float(radii.max()),
        distinct_refs=len({tuple(np.round(r, 9)) for r in refs}),
    )


for parametrization, label in (("arrival_condition", "V1 (对照)"),
                               ("task_state_v2", "V2")):
    print(f"=== {label}  [{parametrization}] ===", flush=True)
    print(f'{"seed":<8}{"decisions":>10}{"changed":>9}{"1st half":>10}'
          f'{"2nd half":>10}{"radius range (m)":>20}{"distinct":>10}', flush=True)
    for seed in (262000, 262001, 262005):
        rng = np.random.default_rng(seed)
        r = run(parametrization, seed, rng)
        import sys
        print(f'{seed:<8}{r["decisions"]:>10}{r["changed_frac"]:>9.3f}'
              f'{r["first_half_frac"]:>10.3f}{r["second_half_frac"]:>10.3f}'
              f'{r["radius_min"]:>10.2f}{r["radius_max"]:>10.2f}'
              f'{r["distinct_refs"]:>10}', flush=True)
    print()
