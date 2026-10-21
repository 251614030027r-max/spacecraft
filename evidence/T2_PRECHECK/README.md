# T2_PRECHECK — sandbox checks of the T2 staged-approach interface (exploratory, not paper numbers)

Code: `env/t2_stage.py`, `experiments/t2.py`, `train/t2_mve_sac.py` (screen at `58551fd`). Training openings 1000000-1000007 only.

1. **Frozen MPC far-field hold defect** (`hold_test.py`; the hold runs used the T2 environment before the
   far-field tracker was added): asked to hold an inertially fixed point from rest (no thrust needed),
   the MPC drifts outward 1.84 m and spends 0.19 m/s in 20 s at 16 m (`hold_mpc_16m.log`), 0.97 m /
   0.10 m/s at 8 m (`hold_mpc_8m.log`). With the inertial plan-tracking law (far field > 7.5 m, MPC
   keeps the attitude and the near field): 0.00 m / 0.000 m/s (`hold_tracker_16m.log`).
2. **Fixed-policy screen** (`screen_58551fd/`, summary `python3 summarise_screen.py screen_58551fd/screen.log c_0_0,rally,c_8_0,c_8_8`):
   neutral action (the untrained policy's deterministic action) capture 4/8, mean -29.1; best uniform
   policy (deadline-aware rule) 7/8, +42.0; per-opening best +68.6 (+26.6); best policy per opening
   c_0_0 3, c_8_8 2, c_8_0 2, rally 1.
3. **Training smoke** (`smoke_e_mve_80/`, `experiments.t2 train --arm e_mve --steps 80 --envs 2 --smoke`): 12 episodes,
   22 / 80 stage transitions carried usable model candidates (66 candidates; the factual gate admitted them
   after the per-environment calibration had 10 stages); model term active in 5 / 6 probes with
   lambda_eff 0.38-1.0 and lambda_eff * |g_model| <= 0.5 |g_real| in every probe; checkpoints save / load;
   the untrained deterministic action is ~(0.05, -0.02), i.e. the neutral action.
