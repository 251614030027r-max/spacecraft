# T1B_V2_DEVCHECK — upper sandbox check of V2 (`a1698ac`), E-SAC, seed 0, 2k steps (2026-10-19)

Formal V2 hyper-parameters (`experiments.t1b_v2.model_for`, learning_starts 500, batch 256); the only
difference from the lower run order is the teacher Replay: regenerated in the sandbox with the same
V1 rule (`sandbox_teacher_receipt.json`) because the V1 pickle lives on the lower machine
(`v2_formal_dev.py` swaps the expected SHA). Development check of the repair mechanics only; no
readout, no claim about returns.

| First review, 8 dev states (292140–147) | V1 E-SAC 2k | V2 E-SAC 2k (this check) |
|---|---|---|
| categorical entropy (max ln 3 = 1.099) | 4.5e-6 | 1.080 |
| actor gradient norm / 0k | 9.9e-6 / 2.2e-3 | 0.041 / 0.080 |
| p(earliest) | 1.000 | 0.40–0.44 |
| learning decisions non-earliest, steps 500–2000 | 0 % (after 1k) | 52 % (198 / 382) |

Q at 2k still differs little between candidates (≈0.13 per step, earliest highest); whether the
critic learns the state-dependent ranking (economy → later, `evidence/T1B_V1_AUDIT/`) is the
question of the lower pilot (gates A3, B1–B2).

Learning decisions by 500-step bucket (candidate index: count): {0: {0: 81, 1: 26, 2: 2}, 500: {0: 47, 1: 42, 2: 18}, 1000: {0: 52, 1: 52, 2: 12}, 1500: {1: 51, 0: 53, 2: 21}}
