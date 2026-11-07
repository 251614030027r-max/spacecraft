# T1B_V3_PRECHECK — why V2 still learns "earlier is better", and the V3 repair (upper, 2026-10-19)

Inputs: V2 pilot `evidence/T1B_V2_PILOT/` (`5c86561`; stopped at A1, E-SAC 5k), V1 audit
`evidence/T1B_V1_AUDIT/`. Sandbox checks on development openings only.

## 1. Root cause: the "intent" was re-decided every 6 s as a relative shift (`v2_decision_structure.json`)

V2 E-SAC: 13.9 learning decisions per episode, **12.7 of them with a live plan**. With a live plan
the verified support is [remaining − 15, remaining, remaining + 15], i.e. advance / keep / delay,
asked again every 3 steps. Consequences:

- A late arrival survives only if ~15 consecutive stochastic choices agree; one "advance" undoes
  it. The critic therefore learns the value of local shifts under the noisy current policy, not
  the value of an arrival intent. Its self-consistent solution is "advance" (V2: 66 % of all
  decisions advance/earliest, and the share rises with training while entropy falls — the A1 stop
  at 5k with 77 % of the 0k actor gradient: not saturation, a learned preference).
- Every advance/delay is a re-plan: fuel and resource exhaustion (17/77 training episodes)
  accumulate from churn, which further favours arriving sooner.
- The candidate truth scan (V1 audit) commits to the first choice and keeps it. V1/V2 Q and the
  scan described different decisions; the 5/15 ranking agreement is expected, not a learning bug.

## 2. V3 (`env/t1_b_feasible_v3.py`, `experiments.t1b_v2 --interface v3`)

A learning decision exists only when no plan is live (initial review; after a plan is lost),
over the G*-verified candidates [first, first + 20, first + 40]. While a plan is live the only
admissible request is its own arrival (keep; G* re-plans the same arrival after a deviation as
before). Learner, temperature 0.5, epsilon 0.10, hash CF alternative, task, D/B, G*, MPC and truth
are V2's.

- Value supervision: the real SMDP interval of a decision runs until the plan is lost or the
  episode ends; ending at a terminal its target is the **Monte Carlo return of the committed
  intent** (no bootstrapping, no truncation, consistent follow-up: the continuation is the
  commitment itself). The scan now measures exactly this decision (up to re-decisions after
  plan loss).
- MPC feedback: pairs are formed only at these decisions; both branches are fresh G* plans (no
  kept-vs-replanned asymmetry). The 6 s MPC prefix reward is nearly uninformative about the long
  term (`mpc_prefix_by_candidate.jsonl`: differences 0.0–0.5; at 292140 it even prefers 100 s,
  truth prefers 140 s by 186). The informative part is the bootstrap at the alternative's
  MPC-predicted successor: the critic's value of a **committed state**, trained by every real step
  of every committed trajectory (~70 per episode vs ~1–3 decisions). The relative target thus
  transfers well-supported state values to the unchosen intents at the same state — model-based
  counterfactual evaluation, no longer the decision-level Q ranking repeating itself. Whether this
  speeds up or improves the state-dependent choice is the E-TCPF − E-SAC question of the run.

## 3. Sandbox checks

- `tests/test_t1_b_v2.py` (5): V1 alternative rule unchanged; V2 hash rule; temperature; V3 decides
  only without a live plan and keeps the committed arrival (292140, 9 subsequent reviews).
- E-TCPF V3 smoke, 400 steps (smoke hyper-parameters): 10 learning decisions, **0 with a live
  plan**; 9/10 pairs valid (weights 0.59–0.94), both pair directions; intervals closed by
  terminal; CF share at the 10 % cap (`smoke_tcpf_400_*`).
