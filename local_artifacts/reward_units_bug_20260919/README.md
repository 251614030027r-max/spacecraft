# Invalid adaptive-training evidence (2026-09-19)

These six small files are the retained evidence from the invalid first adaptive
training round (`adp_262410/411/412`). The reward safety-proximity rate was not
multiplied by the 0.1 s control time step, so legal captures were scored below
hovering and the learned critics are unusable.

Only each run's manifest and Monitor CSV are retained. Checkpoints, interrupted
models and TensorBoard event files were permanently removed because they encode
the inverted objective and must never be resumed. See
`docs/REWARD_UNITS_FIX_20260919.md`. The valid retrain uses fresh directories
`logs/adp_rf_262410/411/412`.
