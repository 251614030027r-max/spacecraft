# T1B_V6_PRECHECK — why the critic cannot learn the state × intent value, and the V6 mechanism (upper, 2026-10-20)

Input: lower V5.1 interim `evidence/T1B_V51_ROUND/key_review_20261020_1530/` (`9e4afff`; seed 0;
E-SAC ~16.3k, E-TCPF ~4.8k).

## Not the reward

The reward (+100 capture, −150 unsafe, −100 timeout/exhaustion, −0.05/s, −20 Δv) is the task's
real trade-off and is never changed. It makes the arrival-time decision a small difference (+2..+3:
later arrival saves descent Δv when the deadline allows) next to large terminal outcomes; that is a
property of the problem, which the learner has to handle, not a defect to tune away.

## The structural cause in the learner

1. **The decision value is hardly trained.** Learning decisions are 469 of 16.3k stored transitions
   (2.9 %, E-SAC) and 117 of 4.8k (2.4 %, E-TCPF): a uniform 256 batch carries ~6–7 decisions. The
   twin critic's gradient goes to continuation values of committed states; Q(s, ·) at decisions is
   shaped mostly by generalisation, which yields the state-independent "later is worse" slope
   (E-SAC −1.15 / −7.85 / −15.48 per candidate at 2k / 5k / 10k; 15/15 states earliest first).
2. **The MPC labels are diluted by the same 2 %.** V5.1 already produced 115 admitted full-horizon
   pairs in 4.8k (factual error ≤ 4.83; 40 say later is better, 18 earlier), but their loss is a mean
   over the whole batch: the share sat at the 20 % cap and could not move the decision values.
3. **Independent real outcomes are few.** One episode = one outcome (~150 per 10k steps). Real
   returns can fix the value level but cannot resolve +2..+3 intent differences under ±100
   variance; only the paired full-horizon MPC rollouts multiply outcomes per decision.

## V6 mechanism: intent-level replay with an MPC-paired outcome advantage (`train/t1_b_v6_sac.py`)

Each critic update keeps the V5.1 transition batch (real interval TD on all transitions) and adds
an **intent batch** of 256 closed learning decisions on the same twin critic:
- real interval target on Q(s, a_taken) (the Monte Carlo return of the committed intent when it
  ends at a terminal): the real return anchors the value level — both arms;
- E-TCPF only: the gated full-horizon MPC pairs of those decisions train the intent advantage
  Q(s, ã) − Q(s, a) → Y, averaged over the admitted pairs, weight 1 (no calibration, no cap).

E-SAC gets the same intent replay with real targets only, so the ablation isolates the MPC
feedback. Interface (V5.1), actor, temperature 0.5, exploration 0.5, task, G*, MPC, truth unchanged.

Smoke (E-TCPF, 500 steps, smoke hyper-parameters): intent-batch real-TD gradient 13–21 vs
transition-batch 1.0–2.7; 9–25 % of the intent batch carries an admitted pair; before the
per-pair averaging the MPC-advantage gradient was 0.4–0.8 (diluted), hence the pair-mean.
After the pair-mean (400 steps, `smoke_v6_pairmean_*`): MPC-advantage gradient 1.9–3.2 vs intent
real-TD 12–16 and transition TD 0.6–2.4, i.e. the MPC information now reaches the decision values
at ~15–25 % of the real anchor instead of ~3 %.
