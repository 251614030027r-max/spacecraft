"""T2 learner: SAC on the 16 s stage decision, with Execution-Anchored Counterfactual Coupling (EACC, T2.3).

Real part (both arms): SB3 SAC (twin critic, fixed entropy coefficient, DEAD_ENDS D02) on the real
stage transitions; success / unsafe / timeout / resource exhaustion are terminal (no bootstrap).

Coupled part (``e_mve`` only). Per real transition the environment returns the executor model's raw
prediction of the executed intent (r^_a, s^'_a, d^_a) and K anchored candidate labels (a_k, r~_k, s~'_k, d~_k)
(``T2StageEnv._labels``: M1 executor option model, M2 hindsight execution anchoring). M3 robust precision
fusion, one rule in place of T2.2's gates and gradient cap:

    y(r, s', d)  = r + gamma (1 - d) V'(s'),   V'(s') = min_i Q'_i(s', a~) - alpha log pi(a~ | s'),  a~ ~ pi(.|s')
    e_a          = y(r, s', d) - y(r^_a, s^'_a, d^_a)          (the model's error on the executed intent, value units)
    w            = 1 / (1 + (e_a / sigma)^2)                    (sigma: running RMS of the real TD residual)
    L_Q          = L_real + beta * sum_{b,k} w_b 1/2 sum_i (Q_i(s_b, a_k) - y(r~_k, s~'_k, d~_k))^2 / #candidates

Every label is r + gamma (1 - d) V'(.) of a single next state, so the fused regression target is a
gamma-contraction; w |e_a| <= sigma / 2 bounds the influence of any model error (a Cauchy M-estimator).
The term is off until ``model_start`` real transitions are stored. beta = 0.5.

T2.12 EACC (arm ``e_rel``; learner 4.0): the absolute candidate regression above is replaced by the execution-anchored
relative labels of ``_relative_loss`` (module constant block REL_BETA): the executor model values each candidate and the
executed intent through the same treatment, only their difference D_j enters, anchored at the critic's own value of the
executed intent (no gradient), so real execution alone sets the value level and the model shapes the action dependence.
One-stage labels (the stage end is valued by the critic that reads the executor's judgements, reduced_priv). Arm
``e_priv``: the same critic view with no counterfactual feedback (ablation). T2.11's paths (``e_acc``) are unchanged.

Collection (``train``): asynchronous -- each worker process steps its own environment at its own pace and
the learner (this process) answers every action request with the current policy and performs
``UPDATES_PER_TRANSITION`` updates per stored transition. T2.2's lock-step vector environment waited
for the slowest environment (a bridge + descent macro-step, ~50 s) at every decision.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

from env.t2_stage import GAMMA_STAGE, OBS_DIM, OBS_DIM_PRIV

K_CANDIDATES = 8          # 3 structured alternatives + 1 global + 4 local (T2StageEnv.candidate_actions)
MODEL_BETA = 0.5
MODEL_START = 300
ENT_COEF = 0.05
SIGMA_DECAY = 0.99        # running mean of the squared real TD residual
#: critic / actor updates per stored real transition
UPDATES_PER_TRANSITION = 2
#: T2.7 executor guidance: at a stage decision the collector flies the executor's projected capture plan
#: (T2StageEnv.track_intent) instead of the policy's action with probability GUIDE_P0 * max(0, 1 - t / GUIDE_UNTIL),
#: t the stored transitions; the replay keeps the executed action (off-policy SAC learns from it like any behaviour).
#: Training only: evaluation flies the deterministic actor alone. Pre-registered with the T2.7 run order.
GUIDE_P0 = 0.5
GUIDE_UNTIL = 2000
#: T2.5 two-stage labels: the policy copy in each environment is refreshed every SYNC_UPDATES critic/actor updates;
#: a stage-2 part older than AGE_CAP stored transitions is not used in the Retrace correction (the label falls back to its
#: stage-1 target); its one-step stage-2 row (T2.5.1) has no age cap
SYNC_UPDATES = 100
AGE_CAP = 2000
#: learner version recorded in the manifest (T2.5.1: one-step stage-2 rows; T2.6: calibration shared by the workers;
#: T2.6.1: labels end at the deadline / budget, env t2-stage-0.4.2; env 0.4.3 adds the budget before a predicted entry;
#: T2.7: executor guidance in the collector (``guide_probability``; off by default), env 0.5.0;
#: T2.9: the decision-layer interface ``obs_profile`` passed to every environment and recorded, env 0.7.0;
#: T2.10: arm e_pref, the executor-calibrated opportunity preference on the critic, env 0.8.0;
#: T2.11: arm e_acc, EACC-A -- decision-level execution calibration and calibrated advantage to critic and actor, env 0.9.0;
#: T2.12: arm e_rel, EACC -- relative execution-anchored counterfactual value learning (``_relative_loss``), and arm e_priv,
#: the same critic view without counterfactual feedback; env 0.10.0;
#: 4.1 (EACC iteration): common random numbers in the bootstrap values of one decision's relative labels, and candidates
#: separated in the executor's stage-goal terms (env 0.11.0, ``cand_min_sep``; arm e_rel only);
#: 5.0: arms e_term / e_term2 -- executor-evaluable (terminal) counterfactual values only, ``_terminal_loss``)
LEARNER_VERSION = "t2-learner-6.1"
#: T2.6: every worker's executed stages and entries feed one stage / entry calibration (one executor, one model of it).
#: Per-worker calibrations were an artifact of the parallel collector: each worker waited for its own 20 entries, so the
#: labels started only after ~workers x 20 entries (T2.5.1: first model rows at 512 / 537 decisions with 3 workers).
POOL_CALIBRATION = True
#: T2.10 executor-calibrated opportunity preference (arm e_pref; research layer 2026-11-02: relative advantage / ranking
#: from the executor on the decision's candidates instead of more short Bellman labels). At every stage decision the
#: environment values PREF_GRID and the executed intent with T2StageEnv.opportunity_value (earliest feasible entry within
#: the time left, valued by the entry outcome model's historical execution replay). For every pair whose values differ by
#: at least PREF_DELTA the critic is asked to order them, by a margin that grows with the executor's value gap:
#:     L_pref = mean over such pairs and both critics of relu(min(PREF_KAPPA dv, PREF_MARGIN_CAP) - (Q_i(s, a_hi) - Q_i(s, a_lo)))
#: and the critic loss is L_real + PREF_LAMBDA * L_pref. A one-sided hinge: a pair the critic already separates by the
#: margin costs nothing, and no value is regressed on the executor's number, so its optimism cannot enter the Bellman
#: targets; the actor follows the critic as in SAC. PREF_DELTA = 50: only the executor's capture-versus-timeout calls
#: (P4, evidence/T2_8_P4_INFO: opp2 calls >= 50 apart are right 59/59 on the selection openings and 76/94 = 0.81 on the
#: held-out ones, breaking 0 / 1.5 % of the pairs the critic orders right; calls 5-20 apart are near chance). Scale:
#: the hinge's gradient is O(1) per pair while the TD regression's is O(TD error ~ 10-30), hence PREF_LAMBDA = 10
#: (sandbox integration, T2.10: with a fixed margin of 2 and weight 1 the term was 0.15 % of the critic loss).
PREF_LAMBDA = 10.0
PREF_DELTA = 50.0
PREF_KAPPA = 0.5
PREF_MARGIN_CAP = 100.0
PREF_K = 11            # PREF_GRID (9) + the executed intent + (T2.11) the policy's deterministic intent
PREF_EXEC, PREF_PI = 9, 10
#: T2.11 EACC-A (arm e_acc): the original EACC labels (M1-M3, two stages, model_beta) plus a decision-level execution-
#: calibrated advantage. Diagnosis (evidence/T2_8_P4_INFO_UPPER): the executor's planner judges an intent's capture
#: opportunity from the stage end as if it could enter directly, while the executor enters only within 12 m and 35 deg
#: -- 210 of the 221 capture-versus-timeout pairs the planner left undecided were such inconsistencies; a logistic
#: calibration of the planner's judgement against realized outcomes predicts capture at 0.93 on held-out openings (raw
#: 0.70). EACC-A learns that calibration online from the agent's own executed decisions (DecisionCalibration: the
#: executed intent's opportunity features -> did the episode end in capture), trusts it once its predict-then-observe
#: accuracy is >= ACC_MIN_ACC over >= ACC_MIN_LABELS outcomes, and at every decision compares the policy's own intent
#: with the grid and the executed intent: P_k = calibrated capture probability, dP_k = P_k - P_pi. Where dP_k >= ACC_DP
#: (a decisive, trusted advantage):
#:   critic  relu(ACC_KAPPA * ACC_SCALE * dP_k - (Q(s, a_k) - Q(s, a_pi))), weight ACC_LAMBDA_Q   (both critics)
#:   actor   ACC_ETA * dP_* * ||tanh(mu(s)) - a_*||^2 with a_* the most probable capture among the decisive candidates
#: so the information reaches the deployed actor directly, not only through the critic's local gradient (P4: labelled
#: critics ranked key decisions better while the actor chose no better). Training only; the actor alone is deployed.
ACC_F = 7              # calibration features (dec_features)
ACC_DP = 0.4
ACC_KAPPA = 0.5
ACC_SCALE = 200.0      # capture reward + timeout penalty
ACC_LAMBDA_Q = 10.0
ACC_ETA = 10.0
ACC_MIN_LABELS = 200
ACC_MIN_ACC = 0.8
#: T2.12 EACC (arm e_rel): execution-anchored relative counterfactual value learning. One mechanism replaces T2.11's
#: absolute candidate regression, two-stage Retrace, stage-2 rows, decision calibration and advantage terms.
#: At a decision s with executed intent a_f and real target y_f = r_f + gamma (1 - d_f) V'(s'_f), the executor model with
#: the candidates' treatment (M1 option model, M2 realized attitude and executed residual path, entry outcome offset)
#: predicts every candidate a_j and a_f itself (``fact.same``, env 0.10.0): Y(a) = r^_a + gamma (1 - d^_a) V'(s^'_a).
#: The model supplies only a difference to the executed intent; the value level stays with the real data:
#:   D_j = Y(a_j) - R_f,   R_f = Y(a_f) if c_j else y_f,   c_j = 1[outcome class of a_j's label = that of a_f's]
#:   L_rel = mean_{b,j} w_bj 1/2 sum_i (Q_i(s_b, a_j) - [sg Q_i(s_b, a_f) + D_bj])^2,   L_Q = L_real + REL_BETA L_rel
#: (outcome class: stage end / entry / deadline-budget end; sg: no gradient; the actor follows the critic as in SAC).
#: Comparable pair (c_j = 1): the executed intent's reference is the model's own same-treatment prediction, so every error
#: the two predictions share cancels (the decision-state PBRS potential, the entry outcome model's error on two entries
#: from the same state, the calibration's common bias); D_j = 0 for a_j = a_f. Pair of different classes (one opens the
#: entry, the other does not): their model errors are not shared, so the reference is the real executed value y_f (a
#: control variate is used only where it reduces error; sandbox truth branches, evidence/T2_12_PRECHECK: entry-entry pairs
#: rms error 20.0 absolute -> 0.9 relative, stage-stage equal, cross-class worse relative than absolute). Either way the
#: target is anchored at the critic's own value of the executed intent, so no label moves the value level.
#: Trust (M3, one Cauchy rule): w = 1 / (1 + (eps / sigma)^2), eps = |e_raw| + b 1[k_j = 2] (1 - c_j 1[k_f = 2]),
#: e_raw = y_f - Y_raw(a_f) the calibration-only model's error on the executed transition, b the current entry outcome-model
#: bias, charged to an unanchored entry label unless its reference is an unanchored entry of the same decision (b cancels).
#: Common random numbers (learner 4.1): the soft values V'(s^'_a) of one decision's candidates and of its reference
#: Y(a_f) are evaluated with the same policy noise (one standard-normal draw per decision, ``_soft_value_noise``), so the
#: sampling noise of the bootstrap cancels in D_j as the exogenous target motion does in the labels (M2); every row keeps
#: the marginal of SAC's soft value, so each target is unchanged in expectation. Two candidates with the same predicted
#: transition get D_j = 0 exactly.
REL_BETA = 0.5
#: Learner 5.0 (arms e_term, e_term2): executor-evaluable counterfactual values. T2.12 (development round 2026-11-03,
#: midterm at 1000 decisions): the critic that reads the executor's judgements without any counterfactual feedback
#: (e_priv) learned clearly better than the same critic with the relative labels (curve AUC 14.9 against -5.7, both
#: seeds) and no worse than T2.11's absolute labels. Most e_rel rows are stage ends whose value is the critic's own
#: V'(s^') at a state the executor model predicted and the critic was never trained on: the model hands the critic its
#: own extrapolation back as an action preference. Learner 5.0 keeps only the rows whose value the executor determines
#: without the critic -- candidates whose modelled continuation reaches a terminal event (the entry flown to its replayed
#: outcome, or the deadline / budget end): within stage 1 (e_term), or within stage 1 or a second stage flown under the
#: policy's deterministic intent (e_term2). Target, a constant:
#:   Y_j = r^_j (stage-1 terminal)  or  r^_j1 + gamma r^_j2 (stage-2 terminal; a2 = the policy mean at s^_j1)
#:   T_j = Y_j + c_j (y_f - Y(a_f)),  c_j = 1 when the executed intent's same-treatment label is a terminal of the same
#:         kind (entry / deadline-budget end) as a_j's stage-1 label: the model's error on the executed outcome, which
#:         the two share, is removed (y_f is the real Bellman target of the executed transition); else 0
#:   L_term = mean_{rows} w_j 1/2 sum_i (Q_i(s, a_j) - T_j)^2,   L_Q = L_TD + TERM_BETA L_term
#: Trust: w = 1 / (1 + (eps / sigma)^2), eps = |e_raw| + b 1[unanchored entry] (1 - c_j 1[k_f = 2]). No target bootstraps on
#: the critic at a modelled state, so the fused regression stays a gamma-contraction in the tabular case (every target is
#: a constant or a real one-step target).
TERM_BETA = 0.5
#: Learner 6.0 (arms e_pi, e_pit): EACC as an execution-consistent policy improvement. Development rounds T2.11-T2.14:
#: every way of regressing the critic on counterfactual values (absolute, relative, terminal-only) learned worse than
#: the same critic trained on real transitions alone (e_priv), although the terminal labels are accurate against truth
#: branches -- the failure is where the feedback enters (the critic's value landscape, its bootstraps), not only what
#: it says. Learner 6.0 keeps e_priv's critic exactly (real soft Bellman targets only, the executor's judgements in its
#: view) and lets the executor's counterfactual evidence act on the policy improvement step only:
#:   the executor predicts, through one treatment (M1, M2, common random numbers), the candidates a_j and the policy's
#:   own deterministic intent a_pi at the decision (env 0.12.0 ``ref``);  Y(a) = r^_a + gamma (1 - d^_a) V'_xi(s^'_a)
#:   critic advantage   A_Q(a_j) = Q(z, a_j) - Q(z, a_pi)        (the real-data critic, no gradient)
#:   executor advantage A_E(a_j) = Y(a_j) - Y(a_pi)  (e_pi: every candidate; e_pit: only the candidates the executor
#:                      evaluates to a terminal event)
#:   correction field   C_j = rho_j (A_E(a_j) - A_Q(a_j)),   rho_j = sigma^2 / (sigma^2 + eps_j^2)
#: rho is the inverse-variance weight of two estimates of the same advantage (the critic's, error scale sigma = the RMS
#: real TD residual; the executor's, error scale eps_j = |e_raw| + unshared entry bias): A_mix = A_Q + rho (A_E - A_Q)
#: is their minimum-variance combination when the errors are independent and unbiased (the Cauchy weight of M3 is this
#: weight). The policy follows grad_a A_mix: SAC's reparameterized gradient of Q plus the slope g of the correction field near
#: a_pi, applied at the actor's mean intent:
#:   L_pi = L_SAC + mean_{feedback rows} -sg(g) . tanh(mu(o)),   ||g|| <= FB_CLIP x RMS ||grad_a Q|| of those rows
#: (a trust region on the step). Learner 6.1 (research-layer review of 6.0, 2026-11-04):
#:   - the slope is a centred weighted least-squares fit WITH an intercept over the points (candidate offsets from a_pi,
#:     their corrections) and a_pi itself (correction 0 by definition): an error common to every point gives no
#:     gradient; a row is used only when its points span the action plane enough to identify a direction (>= 2 points,
#:     total spread >= FB_MIN_SPREAD). Both arms compare against the policy's own intent; e_pit keeps only the
#:     candidates the executor resolves to a terminal event (6.1 coverage check: with absolute terminal values and no
#:     reference point a row was identifiable in ~2 % of the feedback rows);
#:   - the feedback rows come from a separate minibatch of the FB_WINDOW most recently stored transitions (FB_BATCH
#:     rows), whose a_pi is the current policy's intent up to a few hundred updates -- a uniform replay batch is mostly
#:     old labels around intents the policy left long ago (6.0 self-check: the gate closed every row within 300 updates);
#:     a row is applied while the current mean intent is within FB_RADIUS of its a_pi.
#: With no evidence in the buffer the feedback path draws nothing (no random numbers) and L_pi is SAC's actor loss, the
#: critic is e_priv's: the update rule and its random stream are e_priv's. The critic never regresses on a model value.
FB_RADIUS = 0.75
FB_RIDGE = 0.05
FB_CLIP = 1.0
FB_WINDOW = 256
FB_BATCH = 64
FB_MIN_SPREAD = 0.01


def centred_slope(d, c, w, ridge: float = FB_RIDGE, min_spread: float = FB_MIN_SPREAD):
    """Weighted least-squares slope with an intercept, per row: d (B, P, 2) point offsets, c (B, P) values, w (B, P)
    weights (0 = absent). Returns (g (B, 2), identifiable (B,)): a constant c gives g = 0; rows with < 2 points or a
    total weighted spread below ``min_spread`` are not identifiable."""
    ws = w.sum(dim=1, keepdim=True).clamp(min=1e-9)
    dbar = (w[..., None] * d).sum(dim=1, keepdim=True) / ws[..., None]
    cbar = (w * c).sum(dim=1, keepdim=True) / ws
    dc, cc = d - dbar, c - cbar
    A = th.einsum('bp,bpi,bpj->bij', w, dc, dc)
    spread = A.diagonal(dim1=1, dim2=2).sum(dim=1) / ws.reshape(-1)
    g = th.linalg.solve(A + ridge * th.eye(d.shape[-1])[None], th.einsum('bp,bp,bpi->bi', w, cc, dc).unsqueeze(-1)).squeeze(-1)
    ok = ((w > 0).sum(dim=1) >= 2) & (spread >= min_spread)
    return th.where(ok[:, None], g, th.zeros_like(g)), ok
CAL_REFIT = 50
CAL_WINDOW = 3000
CAL_L2 = 1e-2


def dec_features(f) -> list:
    """Calibration features of one opportunity judgement (T2StageEnv.opportunity_features): bias, feasible, slack of the
    planned arrival to the time left (two scales), the entry outcome model's success share, range / 15, urgency. The
    same features as the offline check (evidence/T2_8_P4_INFO_UPPER/calib.py)."""
    if f is None:
        return [0.0] * ACC_F
    rem = float(f['remaining_s']); tau = float(f['tau']) if f.get('feasible') else 300.0
    return [1.0, float(bool(f.get('feasible'))), float(np.clip((rem - tau) / 100.0, -3, 3)), float(np.clip((rem - tau - 90.0) / 50.0, -3, 3)),
            float(f.get('p_success') or 0.0) if f.get('feasible') else 0.0, float(f['range_m']) / 15.0, float(bool(f.get('urgent')))]


class DecisionCalibration:
    """T2.11 decision-level execution calibration: logistic P(capture | opportunity features) fitted on executed decisions
    and their realized episode outcomes (sliding window CAL_WINDOW, refit every CAL_REFIT outcomes, L2 CAL_L2). Accuracy
    is measured predict-then-observe on each outcome before it is added (the last ACC_MIN_LABELS outcomes)."""

    def __init__(self):
        self.x, self.y, self.hits, self.w, self.since = [], [], [], np.zeros(ACC_F), 0

    def observe(self, feats, captured: bool) -> None:
        x = np.asarray(feats, np.float64)
        if self.w.any():
            self.hits = (self.hits + [float((float(x @ self.w) > 0) == bool(captured))])[-ACC_MIN_LABELS:]
        self.x = (self.x + [x])[-CAL_WINDOW:]; self.y = (self.y + [float(captured)])[-CAL_WINDOW:]
        self.since += 1
        if self.since >= CAL_REFIT:
            self.fit(); self.since = 0

    def fit(self, iters: int = 300, lr: float = 0.5) -> None:
        if len(set(self.y)) < 2:
            return
        X, Y, w = np.asarray(self.x), np.asarray(self.y), self.w.copy()
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(X @ w)))
            w -= lr * (X.T @ (p - Y) / len(Y) + CAL_L2 * w)
        self.w = w

    @property
    def accuracy(self):
        return float(np.mean(self.hits)) if self.hits else None

    @property
    def trusted(self) -> bool:
        return len(self.hits) >= ACC_MIN_LABELS and self.accuracy >= ACC_MIN_ACC


class StageReplay(ReplayBuffer):
    def __init__(self, *args, k_candidates: int = K_CANDIDATES, **kwargs):
        super().__init__(*args, **kwargs)
        k, d, a = k_candidates, self.obs_shape[0], self.action_dim
        self.k = k
        shape = (self.buffer_size, self.n_envs)
        self.fact_r = np.zeros(shape, np.float32); self.fact_obs = np.zeros((*shape, d), np.float32)
        self.fact_done = np.zeros(shape, np.float32); self.fact_valid = np.zeros(shape, np.float32)
        #: T2.12: the executed intent through the candidates' treatment (anchor of the relative labels)
        self.same_r = np.zeros(shape, np.float32); self.same_obs = np.zeros((*shape, d), np.float32)
        self.same_done = np.zeros(shape, np.float32); self.same_kind = np.zeros(shape, np.float32); self.same_valid = np.zeros(shape, np.float32)
        #: learner 6.0: the policy's own deterministic intent at the decision through the same treatment (env 0.12.0 ``ref``)
        self.ref_a = np.zeros((*shape, self.action_dim), np.float32); self.ref_r = np.zeros(shape, np.float32)
        self.ref_obs = np.zeros((*shape, d), np.float32); self.ref_done = np.zeros(shape, np.float32)
        self.ref_kind = np.zeros(shape, np.float32); self.ref_valid = np.zeros(shape, np.float32)
        self.cand_a = np.zeros((*shape, k, a), np.float32); self.cand_r = np.zeros((*shape, k), np.float32)
        self.cand_obs = np.zeros((*shape, k, d), np.float32); self.cand_done = np.zeros((*shape, k), np.float32)
        self.cand_valid = np.zeros((*shape, k), np.float32)
        #: label kind: 0 stage end, 1 entry with the executed entry's realized outcome anchored, 2 entry on the outcome model alone
        self.cand_kind = np.zeros((*shape, k), np.float32)
        #: stage 2 of a non-terminal candidate (T2.5): a2 ~ mu(.|o1), log mu, shaped r2, o2, terminal flag, kind
        self.c2_valid = np.zeros((*shape, k), np.float32); self.c2_a = np.zeros((*shape, k, a), np.float32)
        self.c2_logp = np.zeros((*shape, k), np.float32); self.c2_r = np.zeros((*shape, k), np.float32)
        self.c2_obs = np.zeros((*shape, k, d), np.float32); self.c2_done = np.zeros((*shape, k), np.float32)
        self.c2_kind = np.zeros((*shape, k), np.float32)
        self.created = np.zeros(shape, np.float32)             # stored transitions at label time (label age diagnostic)
        #: T2.10 opportunity preference: intents and their executor-calibrated opportunity values at the decision
        self.pref_a = np.zeros((*shape, PREF_K, a), np.float32); self.pref_v = np.zeros((*shape, PREF_K), np.float32)
        self.pref_valid = np.zeros((*shape, PREF_K), np.float32)
        self.pref_f = np.zeros((*shape, PREF_K, ACC_F), np.float32)     # T2.11 calibration features per candidate
        self.stored = 0
        self.model_counts = {'transitions': 0, 'with_candidates': 0, 'candidates': 0, 'entry_candidates': 0, 'reason': {}}
        self._last = None

    def add(self, obs, next_obs, action, reward, done, infos):
        i = self.pos
        self.fact_valid[i] = 0.0; self.same_valid[i] = 0.0; self.ref_valid[i] = 0.0; self.cand_valid[i] = 0.0; self.cand_done[i] = 0.0; self.cand_kind[i] = 0.0
        self.c2_valid[i] = 0.0; self.c2_done[i] = 0.0; self.c2_kind[i] = 0.0; self.pref_valid[i] = 0.0
        self.created[i] = self.stored; self.stored += 1
        for e, info in enumerate(infos):
            pref = info.get('pref')
            if pref:
                feats = pref.get('f') or [None] * len(pref['a'])
                for j, (pa, pv, pf) in enumerate(zip(pref['a'][:PREF_K], pref['v'][:PREF_K], feats[:PREF_K])):
                    if pv is not None:
                        self.pref_a[i, e, j] = pa; self.pref_v[i, e, j] = pv; self.pref_valid[i, e, j] = 1.0
                        self.pref_f[i, e, j] = dec_features(pf)
                self.model_counts['pref_decisions'] = self.model_counts.get('pref_decisions', 0) + 1
            model = info.get('model') or {}
            fact, cands = model.get('fact'), (model.get('cands') or [])[:self.k]
            reason = (model.get('diag') or {}).get('reason', 'none')
            c = self.model_counts; c['transitions'] += 1; c['reason'][reason] = c['reason'].get(reason, 0) + 1
            c['resampled'] = c.get('resampled', 0) + int((model.get('diag') or {}).get('resampled') or 0)
            if fact is None or not cands:
                continue
            self.fact_r[i, e] = fact['reward']; self.fact_obs[i, e] = fact['next_obs']; self.fact_done[i, e] = float(fact['done'])
            self.fact_valid[i, e] = 1.0
            ref = model.get('ref')
            if ref is not None:
                self.ref_a[i, e] = ref['a']; self.ref_r[i, e] = ref['reward']; self.ref_obs[i, e] = ref['next_obs']
                self.ref_done[i, e] = float(ref['done']); self.ref_kind[i, e] = float(ref.get('kind', 0)); self.ref_valid[i, e] = 1.0
                c['ref'] = c.get('ref', 0) + 1
            same = fact.get('same')
            if same is not None:
                self.same_r[i, e] = same['reward']; self.same_obs[i, e] = same['next_obs']; self.same_done[i, e] = float(same['done'])
                self.same_kind[i, e] = float(same.get('kind', 0)); self.same_valid[i, e] = 1.0
            c['with_candidates'] += 1; c['candidates'] += len(cands)
            for j, x in enumerate(cands):
                self.cand_a[i, e, j] = x['a']; self.cand_r[i, e, j] = x['reward']; self.cand_obs[i, e, j] = x['next_obs']
                self.cand_done[i, e, j] = float(x['done']); self.cand_valid[i, e, j] = 1.0; self.cand_kind[i, e, j] = float(x.get('kind', 0))
                c['entry_candidates'] += int(x['done'] and x.get('kind', 0) > 0)
                c['cutoff_candidates'] = c.get('cutoff_candidates', 0) + int(x['done'] and x.get('kind', 0) == 0)   # T2.6: deadline / budget
                y2 = x.get('s2')
                if y2 is not None and not x['done']:
                    self.c2_valid[i, e, j] = 1.0; self.c2_a[i, e, j] = y2['a']; self.c2_logp[i, e, j] = y2['logp']
                    self.c2_r[i, e, j] = y2['reward']; self.c2_obs[i, e, j] = y2['next_obs']; self.c2_done[i, e, j] = float(y2['done'])
                    self.c2_kind[i, e, j] = float(y2.get('kind', 0))
                    c['two_stage'] = c.get('two_stage', 0) + 1; c['two_stage_entry'] = c.get('two_stage_entry', 0) + int(y2['done'] and y2.get('kind', 0) == 2)
                    c['two_stage_cutoff'] = c.get('two_stage_cutoff', 0) + int(y2['done'] and y2.get('kind', 0) == 0)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        self._last = (batch_inds, env_indices)
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (self._normalize_obs(self.observations[batch_inds, env_indices, :], env), self.actions[batch_inds, env_indices, :], next_obs,
                (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env))
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def recent_sample(self, n: int, window: int, env=None):
        """Learner 6.1: n transitions drawn uniformly from the ``window`` most recently stored ones (single environment
        slot); sets the index for ``model_batch`` like ``sample``."""
        size = self.buffer_size if self.full else self.pos
        w = max(1, min(int(window), size))
        idx = (self.pos - 1 - np.random.randint(0, w, size=n)) % self.buffer_size
        return self._get_samples(idx, env=env)

    def model_batch(self):
        i, e = self._last
        return {k: self.to_torch(getattr(self, k)[i, e]) for k in
                ('fact_r', 'fact_obs', 'fact_done', 'fact_valid', 'same_r', 'same_obs', 'same_done', 'same_kind', 'same_valid',
                 'ref_a', 'ref_r', 'ref_obs', 'ref_done', 'ref_kind', 'ref_valid', 'cand_a', 'cand_r', 'cand_obs', 'cand_done', 'cand_valid', 'cand_kind', 'created',
                 'c2_valid', 'c2_a', 'c2_logp', 'c2_r', 'c2_obs', 'c2_done', 'c2_kind', 'pref_a', 'pref_v', 'pref_valid', 'pref_f')}


class T2SAC(SAC):
    def __init__(self, *args, model_beta: float = 0.0, model_start: int = MODEL_START, pref_lambda: float = 0.0,
                 acc: bool = False, relative: bool = False, terminal: int = 0, feedback: str | None = None, **kwargs):
        self.model_beta, self.model_start = float(model_beta), int(model_start)
        self.relative = bool(relative)               # T2.12 EACC: relative labels in place of the absolute ones
        self.terminal = int(terminal)                # learner 5.0: executor-evaluable counterfactuals, horizon in stages (0 = off)
        self.feedback = feedback                     # learner 6.0: actor-side execution advantage, 'rel' (e_pi) / 'term' (e_pit) / None
        self.pref_lambda = float(pref_lambda)
        self.acc = bool(acc)                         # T2.11 EACC-A
        self.dec_cal = DecisionCalibration() if self.acc else None
        self.model_stats = []
        self.td_ms = None          # running mean of the squared real TD residual (sigma^2)
        self.entry_bias_now = None # current entry outcome-model bias (mean over environments that know it), None = unknown
        kwargs.setdefault('replay_buffer_class', StageReplay)
        kwargs.setdefault('ent_coef', ENT_COEF)   # fixed (DEAD_ENDS D02)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'model_stats', 'dec_cal']

    def _soft_value(self, obs, ent_coef):
        a, logp = self.actor.action_log_prob(obs)
        return th.min(th.cat(self.critic_target(obs, a), dim=1), dim=1, keepdim=True).values - ent_coef * logp.reshape(-1, 1)

    def _soft_value_noise(self, obs, noise, ent_coef):
        """Soft value of the target critic with a given standard-normal noise for the squashed-Gaussian policy sample
        (the same formula as SB3's SquashedDiagGaussianDistribution: tanh(mu + std * noise), log-prob with the tanh
        correction). With independent noise per row it is ``_soft_value``."""
        mean, log_std, _ = self.actor.get_action_dist_params(obs)
        std = log_std.exp()
        u = mean + std * noise
        a = th.tanh(u)
        logp = (-0.5 * noise ** 2 - log_std - 0.5 * float(np.log(2 * np.pi))).sum(dim=1, keepdim=True)
        logp = logp - th.log(1.0 - a ** 2 + 1e-6).sum(dim=1, keepdim=True)
        return th.min(th.cat(self.critic_target(obs, a), dim=1), dim=1, keepdim=True).values - ent_coef * logp

    def _log_pi(self, obs, actions):
        """log pi(a | obs) of given (squashed) actions under the current actor."""
        mean, log_std, kwargs = self.actor.get_action_dist_params(obs)
        dist = self.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
        return dist.log_prob(actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)).reshape(-1, 1)

    def _q_target_min(self, obs, actions):
        return th.min(th.cat(self.critic_target(obs, actions), dim=1), dim=1, keepdim=True).values

    def _candidate_targets(self, m, mask, err, ent_coef):
        """Targets and weights of the candidate labels selected by ``mask`` (see ``_model_loss``)."""
        k = mask.shape[1]
        o1 = m['cand_obs'][mask]; d1 = m['cand_done'][mask].reshape(-1, 1)
        y = m['cand_r'][mask].reshape(-1, 1) + (1.0 - d1) * self.gamma * self._soft_value(o1, ent_coef)
        age = (float(self.replay_buffer.stored) - m['created']).reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1)
        two = (m['c2_valid'][mask].reshape(-1, 1) > 0) & (d1 == 0) & (age <= AGE_CAP)
        c = th.zeros_like(y); delta = th.zeros_like(y)
        idx = two.reshape(-1)
        if bool(idx.any()):
            a2 = m['c2_a'][mask][idx]; o1s = o1[idx]
            c[idx] = th.exp(self._log_pi(o1s, a2) - m['c2_logp'][mask][idx].reshape(-1, 1)).clamp(max=1.0)
            d2 = m['c2_done'][mask][idx].reshape(-1, 1)
            delta[idx] = (m['c2_r'][mask][idx].reshape(-1, 1) + (1.0 - d2) * self.gamma * self._soft_value(m['c2_obs'][mask][idx], ent_coef)
                          - self._q_target_min(o1s, a2))
        y = y + self.gamma * c * delta
        kind1 = m['cand_kind'][mask].reshape(-1, 1); kind2 = m['c2_kind'][mask].reshape(-1, 1)
        b_now = float(np.sqrt(self.td_ms)) if self.entry_bias_now is None else abs(float(self.entry_bias_now))
        ea = err.abs().reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1)
        eps = ea + (kind1 == 2).float() * b_now + self.gamma * c * (ea + (kind2 == 2).float() * b_now)
        wk = 1.0 / (1.0 + eps ** 2 / self.td_ms)
        return y, c, delta, wk, idx, two, d1, age, kind1

    def _stage2_rows(self, m, mask, err, ent_coef):
        """T2.5.1 stage-2 rows: the stage-2 part of every non-terminal candidate as its own one-step label at the
        predicted stage-1 state, Q(o1, a2) <- r2 + gamma (1 - d2) V(o2). A one-step soft Bellman target is valid for any
        action, so these rows need no importance ratio and no age cap; their information reaches Q(s, a) through V(o1).
        Weight as every model row: eps2 = |e_a| + b2 (b2 the current entry bias for an unanchored entry, else 0).
        Also returns |Qbar(o1, a2) - Qbar(o1, a_pi)| / sigma, how far the label's action lies from the current policy's in value,
        d2 and the stage-2 kind (2 entry, 0 stage end or, T2.6.1, the deadline / budget end)."""
        k = mask.shape[1]
        d1 = m['cand_done'][mask].reshape(-1, 1)
        sel = ((m['c2_valid'][mask].reshape(-1, 1) > 0) & (d1 == 0)).reshape(-1)
        if not bool(sel.any()):
            return None
        o1 = m['cand_obs'][mask][sel]; a2 = m['c2_a'][mask][sel]; d2 = m['c2_done'][mask][sel].reshape(-1, 1)
        y2 = m['c2_r'][mask][sel].reshape(-1, 1) + (1.0 - d2) * self.gamma * self._soft_value(m['c2_obs'][mask][sel], ent_coef)
        b_now = float(np.sqrt(self.td_ms)) if self.entry_bias_now is None else abs(float(self.entry_bias_now))
        ea = err.abs().reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1)[sel]
        eps2 = ea + (m['c2_kind'][mask][sel].reshape(-1, 1) == 2).float() * b_now
        w2 = 1.0 / (1.0 + eps2 ** 2 / self.td_ms)
        a_pi, _ = self.actor.action_log_prob(o1)
        gap = (self._q_target_min(o1, a2) - self._q_target_min(o1, a_pi)).abs() / float(np.sqrt(self.td_ms))
        return o1, a2, y2, w2, gap, d2, m['c2_kind'][mask][sel].reshape(-1, 1)

    def _model_loss(self, data, target, ent_coef):
        """Counterfactual labels fused into the critic regression (module docstring).

        Stage-1 target (every candidate): y1 = r1 + gamma (1 - d1) V(o1), V the soft value of the target critic.
        Two-stage soft Retrace target (T2.5; non-terminal candidates with a stage 2):
            y = y1 + gamma c delta,  delta = r2 + gamma (1 - d2) V(o2) - Qbar(o1, a2),  c = min(1, pi(a2|o1) / mu(a2|o1)),
        a2 ~ mu the environment's policy copy at label time; c = 0 when the stage-2 part is older than AGE_CAP (y = y1).
        E_{a2~pi}[delta] = 0 when Qbar is the soft Q of pi, so the target is the soft Bellman target of pi; with stale
        mu the truncated ratio keeps it a contraction to Q^pi (Retrace, Munos et al. 2016).
        Weight: w_k = 1 / (1 + (eps_k / sigma)^2), eps_k = |e_a| + b1_k + gamma c (|e_a| + b2_k): the factual model
        error stands for each model stage's error, b the current entry outcome-model bias of entry parts resting on
        the outcome model alone (one sigma while unknown), 0 otherwise. eps_k is an estimate, not a proven bound.
        T2.5.1: the stage-2 part of every non-terminal candidate also enters as its own one-step row at the predicted
        stage-1 state (``_stage2_rows``), Q(o1, a2) <- r2 + gamma (1 - d2) V(o2). The Retrace trace c collapses once the
        narrow SAC policy has moved (measured c 0.01-0.55 at label ages 50-400), while the critic's value at the label's
        action differs from that at the current policy's action by a small fraction of sigma; the one-step row keeps
        that information without a ratio and passes it to Q(s, a) through V(o1). The loss is the mean over all model rows."""
        m = self.replay_buffer.model_batch()
        rows = m['fact_valid'] > 0
        if not bool(rows.any()) or self.td_ms is None:
            return None, {}
        with th.no_grad():
            y_fact = m['fact_r'].reshape(-1, 1) + (1.0 - m['fact_done'].reshape(-1, 1)) * self.gamma * self._soft_value(m['fact_obs'], ent_coef)
            err = (target - y_fact).reshape(-1)
            w = 1.0 / (1.0 + err ** 2 / self.td_ms)
            b, k = m['cand_valid'].shape
            mask = (m['cand_valid'] > 0) & rows.reshape(-1, 1)
            y, c, delta, wk, idx, two, d1, age, kind1 = self._candidate_targets(m, mask, err, ent_coef)
            entry = (kind1 > 0)
            rows2 = self._stage2_rows(m, mask, err, ent_coef)
        s = data.observations[:, None, :].expand(-1, k, -1)[mask]
        qs = self.critic(s, m['cand_a'][mask])
        total = (sum((wk * (q - y) ** 2).sum() for q in qs))
        n_rows = int(mask.sum())
        if rows2 is not None:                                        # every model row counts once (stage 1 and stage 2 alike)
            o1, a2, y2, w2, gap, d2, _ = rows2
            total = total + sum((w2 * (q - y2) ** 2).sum() for q in self.critic(o1, a2))
            n_rows += int(o1.shape[0])
        loss = 0.5 * total / max(n_rows, 1)
        e = err[rows]
        n2 = int(idx.sum())
        dec = (m['c2_done'][mask].reshape(-1, 1) > 0) & (m['c2_kind'][mask].reshape(-1, 1) == 2) & two   # stage 2 opens the entry
        return loss, {'model_rows': int(mask.sum()), 'w_mean': float(wk.mean()), 'abs_err_median': float(e.abs().median()),
                      'w_err_max': float((w[rows] * e.abs()).max()), 'entry_share': float(entry.float().mean()),
                      'entry_unanchored_share': float((kind1 == 2).float().mean()), 'entry_bias_now': self.entry_bias_now,
                      'label_age_mean': float(self.replay_buffer.stored - m['created'][rows].mean()),
                      'two_stage_share': n2 / max(int(mask.sum()), 1), 'decisive_share': float(dec.float().mean()),
                      'c_mean': float(c[idx].mean()) if n2 else None, 'c_low_share': float((c[idx] < 0.5).float().mean()) if n2 else None,
                      'increment_abs_mean': float((self.gamma * c[idx] * delta[idx]).abs().mean()) if n2 else None,
                      'age_capped_share': float(((m['c2_valid'][mask].reshape(-1, 1) > 0) & (d1 == 0) & (age > AGE_CAP)).float().mean()),
                      'stage2_rows': int(rows2[0].shape[0]) if rows2 is not None else 0,
                      'w2_mean': float(rows2[3].mean()) if rows2 is not None else None,
                      'pi_gap_sigma_median': float(rows2[4].median()) if rows2 is not None else None,
                      'stage2_entry_share': float((rows2[6] == 2).float().mean()) if rows2 is not None else None,
                      'stage2_cutoff_share': float(((m['c2_done'][mask] > 0) & (m['c2_kind'][mask] == 0) & (m['c2_valid'][mask] > 0)).float().mean()),
                      'stage1_cutoff_share': float(((m['cand_done'][mask] > 0) & (m['cand_kind'][mask] == 0)).float().mean())}

    def _relative_loss(self, data, current, target, ent_coef):
        """T2.12 EACC relative execution-anchored labels (module constants, REL_BETA). ``current``: the critics' values at
        the executed intents of the batch (the anchors, used without gradient); ``target``: the real Bellman targets.
        Returns (loss or None, readings)."""
        m = self.replay_buffer.model_batch()
        rows = (m['fact_valid'] > 0) & (m['same_valid'] > 0)
        if not bool(rows.any()) or self.td_ms is None:
            return None, {}
        b, k = m['cand_valid'].shape
        mask = (m['cand_valid'] > 0) & rows.reshape(-1, 1)
        if not bool(mask.any()):
            return None, {}
        with th.no_grad():
            g = self.gamma
            y_raw = m['fact_r'].reshape(-1, 1) + (1.0 - m['fact_done'].reshape(-1, 1)) * g * self._soft_value(m['fact_obs'], ent_coef)
            noise = th.randn((b, self.actor.action_dist.action_dim))          # common random numbers: one draw per decision
            y_same = m['same_r'].reshape(-1, 1) + (1.0 - m['same_done'].reshape(-1, 1)) * g * self._soft_value_noise(m['same_obs'], noise, ent_coef)
            e_raw = (target - y_raw).reshape(-1); e_same = (target - y_same).reshape(-1)
            o1 = m['cand_obs'][mask]; d1 = m['cand_done'][mask].reshape(-1, 1)
            n1 = noise[:, None, :].expand(-1, k, -1)[mask]
            y_c = m['cand_r'][mask].reshape(-1, 1) + (1.0 - d1) * g * self._soft_value_noise(o1, n1, ent_coef)
            expand = lambda x: x.reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1)
            kind_j = m['cand_kind'][mask].reshape(-1, 1); kind_f = expand(m['same_kind'])
            cls = lambda done, kind: th.where(done > 0, th.where(kind > 0, th.ones_like(kind), 2.0 * th.ones_like(kind)), th.zeros_like(kind))
            comp = (cls(d1, kind_j) == cls(expand(m['same_done']), kind_f)).float()
            ref = comp * expand(y_same) + (1.0 - comp) * expand(target)
            delta = y_c - ref
            b_now = float(np.sqrt(self.td_ms)) if self.entry_bias_now is None else abs(float(self.entry_bias_now))
            eps = expand(e_raw.abs()) + b_now * (kind_j == 2).float() * (1.0 - comp * (kind_f == 2).float())
            w = 1.0 / (1.0 + eps ** 2 / self.td_ms)
            anchors = [expand(q.detach()) for q in current]
        s = data.observations[:, None, :].expand(-1, k, -1)[mask]
        qs = self.critic(s, m['cand_a'][mask])
        n = int(mask.sum())
        loss = 0.5 * sum((w * (q - (anc + delta)) ** 2).sum() for q, anc in zip(qs, anchors)) / max(n, 1)
        with th.no_grad():
            sig = float(np.sqrt(self.td_ms))
            qmin = th.min(th.cat(qs, dim=1), dim=1, keepdim=True).values - th.min(th.cat(anchors, dim=1), dim=1, keepdim=True).values
            big = delta.abs() >= 0.25 * sig
        return loss, {'model_rows': n, 'w_mean': float(w.mean()), 'w_low_share': float((w < 0.5).float().mean()),
                      'abs_err_median': float(e_raw[rows.reshape(-1)].abs().median()), 'same_err_median': float(e_same[rows.reshape(-1)].abs().median()),
                      'same_err_p90': float(th.quantile(e_same[rows.reshape(-1)].abs(), 0.9)),
                      'delta_abs_mean_sigma': float(delta.abs().mean() / sig), 'entry_share': float((d1 > 0).float().mean()),
                      'class_differs_share': float(1.0 - comp.mean()),
                      'rank_agree': float(((qmin * delta) > 0)[big].float().mean()) if bool(big.any()) else None,
                      'entry_bias_now': self.entry_bias_now,
                      'label_age_mean': float(self.replay_buffer.stored - m['created'][rows].mean())}

    def _terminal_loss(self, data, target):
        """Learner 5.0 executor-evaluable counterfactual values (module constants, TERM_BETA). ``target``: the real Bellman
        targets of the batch. Returns (loss or None, readings)."""
        m = self.replay_buffer.model_batch()
        rows = (m['fact_valid'] > 0) & (m['same_valid'] > 0)
        if not bool(rows.any()) or self.td_ms is None:
            return None, {}
        b, k = m['cand_valid'].shape
        valid = (m['cand_valid'] > 0) & rows.reshape(-1, 1)
        t1 = valid & (m['cand_done'] > 0)
        t2 = (valid & (m['cand_done'] == 0) & (m['c2_valid'] > 0) & (m['c2_done'] > 0)) if self.terminal >= 2 else th.zeros_like(t1)
        sel = t1 | t2
        if not bool(sel.any()):
            return None, {'term_rows': 0}
        with th.no_grad():
            g = self.gamma
            y_raw = m['fact_r'].reshape(-1, 1) + (1.0 - m['fact_done'].reshape(-1, 1)) * g * self._soft_value(m['fact_obs'], self.ent_coef_tensor)
            e_raw = (target - y_raw).reshape(-1)
            ex = lambda x: x.reshape(-1, 1).expand(-1, k)[sel].reshape(-1, 1)
            is1 = t1[sel].reshape(-1, 1).float()
            y = m['cand_r'][sel].reshape(-1, 1) + (1.0 - is1) * g * m['c2_r'][sel].reshape(-1, 1)
            kind_j = m['cand_kind'][sel].reshape(-1, 1); kind_f = ex(m['same_kind'])
            comp = is1 * (ex(m['same_done']) > 0).float() * ((kind_j > 0) == (kind_f > 0)).float()   # same terminal kind
            corr = ex(target) - ex(m['same_r'])               # real executed value minus the model's (a terminal: no bootstrap)
            tgt = y + comp * corr
            unanchored = is1 * (kind_j == 2).float() + (1.0 - is1) * (m['c2_kind'][sel].reshape(-1, 1) == 2).float()
            b_now = float(np.sqrt(self.td_ms)) if self.entry_bias_now is None else abs(float(self.entry_bias_now))
            eps = ex(e_raw.abs()) + b_now * unanchored * (1.0 - comp * (kind_f == 2).float())
            w = 1.0 / (1.0 + eps ** 2 / self.td_ms)
        s = data.observations[:, None, :].expand(-1, k, -1)[sel]
        qs = self.critic(s, m['cand_a'][sel])
        n = int(sel.sum())
        loss = 0.5 * sum((w * (q - tgt) ** 2).sum() for q in qs) / max(n, 1)
        with th.no_grad():
            qmin = th.min(th.cat(qs, dim=1), dim=1, keepdim=True).values
        return loss, {'model_rows': n, 'term_rows': n, 'term2_rows': int(t2.sum()), 'term_share': n / max(int(valid.sum()), 1),
                      'comp_share': float(comp.mean()), 'w_mean': float(w.mean()), 'abs_err_median': float(e_raw[rows.reshape(-1)].abs().median()),
                      'term_target_mean': float(tgt.mean()), 'term_q_minus_target_mean': float((qmin - tgt).mean()),
                      'term_corr_abs_mean': float((comp * corr).abs().sum() / max(float(comp.sum()), 1.0)),
                      'entry_bias_now': self.entry_bias_now, 'label_age_mean': float(self.replay_buffer.stored - m['created'][rows].mean())}

    def _actor_correction(self):
        """Learner 6.1 (module constants FB_*): a minibatch of the most recent labelled transitions, and per row the
        centred least-squares slope g of the correction field C_j = rho_j (A_E(a_j) - A_Q(a_j)) near the policy's
        intent a_pi. Returns (observations, rows mask, g, a_pi, readings) or Nones and readings."""
        rb = self.replay_buffer
        if not rb.model_counts.get('ref') or self.td_ms is None:
            return None, None, None, None, {'fb_rows': 0}
        data = rb.recent_sample(FB_BATCH, FB_WINDOW)
        m = rb.model_batch()
        rows = (m['fact_valid'] > 0) & (m['ref_valid'] > 0)
        if not bool(rows.any()):
            return None, None, None, None, {'fb_rows': 0}
        b, k = m['cand_valid'].shape
        a_ref = m['ref_a']
        d = m['cand_a'] - a_ref[:, None, :]
        dist = d.norm(dim=2)
        valid = (m['cand_valid'] > 0) & rows.reshape(-1, 1) & (dist <= FB_RADIUS) & (dist > 1e-3)
        if self.feedback == 'term':
            valid = valid & (m['cand_done'] > 0)
        with th.no_grad():
            g_ = self.gamma
            target = data.rewards + (1 - data.dones) * g_ * self._soft_value(data.next_observations, self.ent_coef_tensor)
            noise = th.randn((b, self.actor.action_dist.action_dim))         # common random numbers per decision
            vn = lambda o, n: self._soft_value_noise(o, n, self.ent_coef_tensor)
            y_ref = m['ref_r'].reshape(-1, 1) + (1.0 - m['ref_done'].reshape(-1, 1)) * g_ * vn(m['ref_obs'], noise)
            y_raw = m['fact_r'].reshape(-1, 1) + (1.0 - m['fact_done'].reshape(-1, 1)) * g_ * self._soft_value(m['fact_obs'], self.ent_coef_tensor)
            e_raw = (target - y_raw).reshape(-1, 1)                           # model error on the executed transition
            ob = data.observations
            y_c = th.zeros((b, k)); q_c = th.zeros((b, k))
            if bool(valid.any()):
                o1 = m['cand_obs'][valid]; d1 = m['cand_done'][valid]
                y_c[valid] = m['cand_r'][valid] + (1.0 - d1) * g_ * vn(o1, noise[:, None, :].expand(-1, k, -1)[valid]).reshape(-1)
                q_c[valid] = th.min(th.cat(self.critic(ob[:, None, :].expand(-1, k, -1)[valid], m['cand_a'][valid]), dim=1), dim=1).values
            q_ref = th.min(th.cat(self.critic(ob, a_ref), dim=1), dim=1, keepdim=True).values
            b_now = float(np.sqrt(self.td_ms)) if self.entry_bias_now is None else abs(float(self.entry_bias_now))
            kind_j = m['cand_kind']; kind_r = m['ref_kind'].reshape(-1, 1)
            # both arms: the executor's advantage over the policy's own intent against the critic's (a level error common
            # to the candidate and the reference cancels); e_pit restricts the candidates to executor-complete terminals
            eps = e_raw.abs() + b_now * ((kind_j == 2).float() - (kind_r == 2).float()).abs()
            corr = (y_c - y_ref) - (q_c - q_ref)
            c_ref = th.zeros((b, 1)); w_ref = rows.reshape(-1, 1).float()   # zero at a_pi by definition
            rho = self.td_ms / (self.td_ms + eps ** 2)
            c = th.where(valid, rho * corr, th.zeros_like(corr))
            dd = th.cat([d, th.zeros((b, 1, d.shape[-1]))], dim=1)
            cc = th.cat([c, c_ref], dim=1); ww = th.cat([valid.float(), w_ref], dim=1)
            g, ok = centred_slope(dd, cc, ww)
            sel = ok & rows
        info = {'fb_rows': int(rows.sum()), 'fb_identified': int(sel.sum()), 'fb_cands': int(valid.sum()),
                'fb_rho_mean': float(rho[valid].mean()) if bool(valid.any()) else None,
                'fb_corr_mean': float(corr[valid].mean()) if bool(valid.any()) else None,
                'fb_corr_abs_mean': float(corr[valid].abs().mean()) if bool(valid.any()) else None,
                'fb_g_norm_mean': float(g[sel].norm(dim=1).mean()) if bool(sel.any()) else None,
                'fb_label_age_mean': float(rb.stored - m['created'][rows].mean())}
        return ob, sel, g, a_ref, info

    def _pref_loss(self, data):
        """T2.10 opportunity preference (module constants): hinge on the critic's order of every decisive pair of the
        decision's intents. Returns (loss or None, readings): pairs used and the share the critic already orders."""
        m = self.replay_buffer.model_batch()
        valid = m['pref_valid'] > 0                                          # (B, K)
        rows = valid.sum(dim=1) >= 2
        if not bool(rows.any()):
            return None, {'pref_pairs': 0}
        a, v, valid = m['pref_a'][rows], m['pref_v'][rows], valid[rows]
        b, k = v.shape
        s = data.observations[rows][:, None, :].expand(-1, k, -1).reshape(b * k, -1)
        qs = [q.reshape(b, k) for q in self.critic(s, a.reshape(b * k, -1))]
        dv = v[:, :, None] - v[:, None, :]                                   # (B, K, K): hi - lo
        mask = (dv >= PREF_DELTA) & valid[:, :, None] & valid[:, None, :]
        n = int(mask.sum())
        if n == 0:
            return None, {'pref_pairs': 0}
        margin = (PREF_KAPPA * dv).clamp(max=PREF_MARGIN_CAP)
        loss = sum(F.relu(margin - (q[:, :, None] - q[:, None, :]))[mask].mean() for q in qs) / len(qs)
        with th.no_grad():
            qmin = th.min(th.stack(qs), dim=0).values
            ordered = float(((qmin[:, :, None] - qmin[:, None, :]) > 0)[mask].float().mean())
        return loss, {'pref_pairs': n, 'pref_ordered': ordered, 'pref_states': int(rows.sum())}

    def _acc_terms(self, data):
        """T2.11 EACC-A (module constants): the critic's advantage hinge and the actor's pull target from the calibrated
        capture probabilities of the decision's candidates. Returns (critic loss or None, actor rows, actor targets,
        actor weights, readings)."""
        cal = self.dec_cal
        if cal is None or not cal.trusted:
            return None, None, None, None, {'acc_trusted': 0.0}
        m = self.replay_buffer.model_batch()
        valid = m['pref_valid'] > 0
        w = th.as_tensor(cal.w, dtype=th.float32)
        prob = th.sigmoid(m['pref_f'] @ w)                                    # (B, K)
        dp = prob - prob[:, PREF_PI:PREF_PI + 1]
        dec = (dp >= ACC_DP) & valid & valid[:, PREF_PI:PREF_PI + 1]
        dec[:, PREF_PI] = False
        rows = dec.any(dim=1)
        if not bool(rows.any()):
            return None, None, None, None, {'acc_trusted': 1.0, 'acc_rows': 0}
        b, k = dec.shape
        s = data.observations[:, None, :].expand(-1, k, -1)[dec]
        a_pi = m['pref_a'][:, PREF_PI, :][:, None, :].expand(-1, k, -1)[dec]
        s_pi = data.observations[:, None, :].expand(-1, k, -1)[dec]
        margin = (ACC_KAPPA * ACC_SCALE * dp[dec]).reshape(-1, 1)
        q_c = self.critic(s, m['pref_a'][dec]); q_p = self.critic(s_pi, a_pi)
        loss = sum(F.relu(margin - (qc - qp)).mean() for qc, qp in zip(q_c, q_p)) / len(q_c)
        score = th.where(dec, prob, th.full_like(prob, -1.0))
        best = score.argmax(dim=1)
        target = m['pref_a'][th.arange(b), best][rows]
        weight = dp[th.arange(b), best][rows].clamp(0.0, 1.0)
        with th.no_grad():
            ordered = float(((th.min(th.cat(q_c, dim=1), dim=1).values - th.min(th.cat(q_p, dim=1), dim=1).values) > 0).float().mean())
        return loss, rows, target, weight, {'acc_trusted': 1.0, 'acc_rows': int(rows.sum()), 'acc_pairs': int(dec.sum()),
                                            'acc_q_ordered': ordered, 'acc_dp_mean': float(dp[dec].mean())}

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        ent_coef = self.ent_coef_tensor
        stored = (self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos) * self.replay_buffer.n_envs
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actions_pi, log_prob = self.actor.action_log_prob(data.observations); log_prob = log_prob.reshape(-1, 1)
            with th.no_grad():
                target = data.rewards + (1 - data.dones) * self.gamma * self._soft_value(data.next_observations, ent_coef)
            current = self.critic(data.observations, data.actions)
            real = 0.5 * sum(F.mse_loss(q, target) for q in current)
            with th.no_grad():
                ms = float(sum(((q - target) ** 2).mean() for q in current) / len(current))
                self.td_ms = ms if self.td_ms is None else SIGMA_DECAY * self.td_ms + (1.0 - SIGMA_DECAY) * ms
            loss, mloss, info = real, None, {}
            if self.model_beta > 0 and stored >= self.model_start:
                mloss, info = (self._terminal_loss(data, target) if self.terminal
                               else self._relative_loss(data, current, target, ent_coef) if self.relative
                               else self._model_loss(data, target, ent_coef))
                if mloss is not None:
                    loss = real + self.model_beta * mloss
                    info['loss_share'] = float(self.model_beta * mloss.item() / max(real.item(), 1e-9))
                    if (self._n_updates + step) % 100 == 0:   # read-only: do the two critic gradients agree in direction?
                        params = [p for p in self.critic.parameters() if p.requires_grad]
                        g1 = th.autograd.grad(real, params, retain_graph=True, allow_unused=True)
                        g2 = th.autograd.grad(mloss, params, retain_graph=True, allow_unused=True)
                        v1 = th.cat([g.reshape(-1) for g in g1 if g is not None]); v2 = th.cat([g.reshape(-1) for g in g2 if g is not None])
                        info['grad_cos'] = float(F.cosine_similarity(v1, v2, dim=0))
            acc_rows = None
            if self.acc:
                aloss, acc_rows, acc_target, acc_w, ainfo = self._acc_terms(data)
                info.update(ainfo)
                info['cal_acc'] = self.dec_cal.accuracy; info['cal_n'] = len(self.dec_cal.y)
                if aloss is not None:
                    loss = loss + ACC_LAMBDA_Q * aloss
                    info['acc_q_loss'] = float(aloss.item()); info['acc_q_share'] = float(ACC_LAMBDA_Q * aloss.item() / max(real.item(), 1e-9))
            if self.pref_lambda > 0:
                ploss, pinfo = self._pref_loss(data)
                info.update(pinfo)
                if ploss is not None:
                    loss = loss + self.pref_lambda * ploss
                    info['pref_loss'] = float(ploss.item()); info['pref_share'] = float(self.pref_lambda * ploss.item() / max(real.item(), 1e-9))
            self.critic.optimizer.zero_grad(); loss.backward(); self.critic.optimizer.step()
            q_pi = th.min(th.cat(self.critic(data.observations, actions_pi), dim=1), dim=1, keepdim=True).values
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            if self.feedback and stored >= self.model_start:      # learner 6.1: execution-corrected action gradient
                ob_f, sel, g, a_ref, finfo = self._actor_correction()
                info.update(finfo)
                if sel is not None and bool(sel.any()):
                    mean_a, _, _ = self.actor.get_action_dist_params(ob_f)
                    mu = th.tanh(mean_a)
                    near = sel & ((mu.detach() - a_ref).norm(dim=1) <= FB_RADIUS)
                    if bool(near.any()):
                        a_q = a_ref.clone().requires_grad_(True)              # the critic's own action gradient, for the trust region
                        gq = th.autograd.grad(th.min(th.cat(self.critic(ob_f, a_q), dim=1), dim=1).values.sum(), a_q)[0]
                        scale = FB_CLIP * float(gq[near].norm(dim=1).pow(2).mean().sqrt())
                        gn = g.norm(dim=1, keepdim=True)
                        g_c = g * th.clamp(scale / gn.clamp(min=1e-9), max=1.0)
                        corr_loss = -(g_c.detach()[near] * mu[near]).sum() / ob_f.shape[0]
                        actor_loss = actor_loss + corr_loss
                        info.update(fb_applied=int(near.sum()), fb_applied_share=float(near.float().mean()), fb_gq_rms=float(scale / max(FB_CLIP, 1e-9)),
                                    fb_cos=float(F.cosine_similarity(g_c[near], gq[near], dim=1).mean()))
                    else:
                        info.update(fb_applied=0, fb_applied_share=0.0)
            if acc_rows is not None:                  # T2.11: pull the actor's mean intent to the trusted better candidate
                mean_a, _, _ = self.actor.get_action_dist_params(data.observations[acc_rows])
                dist = ((th.tanh(mean_a) - acc_target) ** 2).sum(dim=1)
                bc = (acc_w * dist).mean()
                actor_loss = actor_loss + ACC_ETA * bc
                info['acc_bc'] = float(bc.item()); info['acc_actor_dist'] = float(dist.sqrt().mean().item())
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            if (self._n_updates + step) % 25 == 0:
                self.model_stats.append({'steps': int(self.num_timesteps), 'update': int(self._n_updates + step), 'real_td': float(real.item()),
                                         'sigma': float(np.sqrt(self.td_ms)), 'model_td': float(mloss.item()) if mloss is not None else None,
                                         **info, 'actor_loss': float(actor_loss.item()), 'q_pi_mean': float(q_pi.mean().item())})
        self._n_updates += gradient_steps


#: arms whose critic reads the executor's judgements (reduced_priv interface) and arms that train on counterfactual labels
PRIV_CRITIC_ARMS = ('e_acc', 'e_rel', 'e_priv', 'e_term', 'e_term2', 'e_pi', 'e_pit')
LABEL_ARMS = ('e_mve', 'e_acc', 'e_rel', 'e_term', 'e_term2')
#: learner 5.0: counterfactual horizon (stages) of the executor-evaluable arms
TERM_ARMS = {'e_term': 1, 'e_term2': 2}
#: learner 6.0: actor-side execution advantage arms (counterfactual evidence computed, no critic labels)
FB_ARMS = {'e_pi': 'rel', 'e_pit': 'term'}


def make_model(env, arm: str, seed: int, smoke: bool = False, updates_per_transition: float = UPDATES_PER_TRANSITION,
               model_beta: float = MODEL_BETA) -> T2SAC:
    obs_dim = int(env.observation_space.shape[0])
    if obs_dim == OBS_DIM_PRIV:      # T2.11: split views; only EACC-A's critic reads the executor's judgements
        policy, kw = SplitViewPolicy, {'net_arch': [256, 256], 'actor_dim': OBS_DIM, 'critic_dim': OBS_DIM_PRIV if arm in PRIV_CRITIC_ARMS else OBS_DIM}
    else:
        policy, kw = 'MlpPolicy', {'net_arch': [256, 256]}
    return T2SAC(policy, env, seed=seed, device='cpu', verbose=0, policy_kwargs=kw,
                 gamma=GAMMA_STAGE, learning_rate=3e-4, buffer_size=100_000, batch_size=64 if smoke else 256,
                 learning_starts=16 if smoke else 200, train_freq=(1, 'step'), gradient_steps=max(1, int(round(updates_per_transition))),
                 ent_coef=ENT_COEF, tau=0.005, model_beta=(TERM_BETA if arm in TERM_ARMS else model_beta) if arm in LABEL_ARMS else 0.0, model_start=16 if smoke else MODEL_START,
                 pref_lambda=PREF_LAMBDA if arm == 'e_pref' else 0.0, acc=arm == 'e_acc', relative=arm == 'e_rel', terminal=TERM_ARMS.get(arm, 0),
                 feedback=FB_ARMS.get(arm))


# ---------------------------------------------------------------------- asynchronous collection
class SpacesEnv(gym.Env):
    """Observation / action spaces of the stage decision, for the learner process (no physics)."""

    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, obs_dim: int = OBS_DIM):
        self.obs_dim = int(obs_dim)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(self.obs_dim, np.float32), {}

    def step(self, action):
        return np.zeros(self.obs_dim, np.float32), 0.0, True, False, {}


class SliceExtractor(BaseFeaturesExtractor):
    """The first ``dim`` values of the observation (T2.11: the actor's reduced interface, or the critic's view)."""

    def __init__(self, observation_space, dim: int):
        super().__init__(observation_space, features_dim=int(dim))
        self.dim = int(dim)

    def forward(self, obs):
        return obs[..., :self.dim]


class SplitViewPolicy(SACPolicy):
    """T2.11: one observation (OBS_DIM_PRIV: reduced interface followed by the executor's judgements), two views. The
    actor reads the first ``actor_dim`` values -- the deployed policy never sees the executor's information; the critic
    reads the first ``critic_dim`` -- all of it for EACC-A (the executor's judgements as training-time information), the
    actor's view for the SAC baseline (so the baseline is SAC on the reduced interface, the same environment process)."""

    def __init__(self, *args, actor_dim: int = OBS_DIM, critic_dim: int = OBS_DIM, **kwargs):
        self.actor_dim, self.critic_dim = int(actor_dim), int(critic_dim)
        kwargs['share_features_extractor'] = False
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        return super().make_actor(SliceExtractor(self.observation_space, self.actor_dim))

    def make_critic(self, features_extractor=None):
        return super().make_critic(SliceExtractor(self.observation_space, self.critic_dim))

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'actor_dim': self.actor_dim, 'critic_dim': self.critic_dim}


def guide_probability(t: int, p0: float, until: int) -> float:
    """T2.7: probability that the collector flies the executor's projected plan at stored transition t (linear decay)."""
    return float(p0) * max(0.0, 1.0 - float(t) / float(until)) if p0 > 0 and until > 0 else 0.0


def _slim(info: dict) -> dict:
    """What the learner needs from a step's info (the model labels without their audit fields)."""
    model = info.get('model') or {}
    cands = [{**{k: c.get(k, 0) for k in ('a', 'reward', 'next_obs', 'done', 'kind')}, 's2': c.get('s2')} for c in model.get('cands') or []]
    diag = model.get('diag') or {}
    return {'model': {'fact': model.get('fact'), 'cands': cands, 'ref': model.get('ref'), 'diag': {k: diag.get(k) for k in ('reason', 'entry_bias', 'entry_records', 'resampled')}},
            **{k: info.get(k) for k in ('event', 'time_s', 'delta_v', 'mission', 'guided', 'pref', 'true_reward')}}


class CalibrationPool:
    """Learner side of the shared calibration: rows reported by the workers, forwarded to every other worker once."""

    def __init__(self):
        self.rows, self.sent = [], {}

    def add(self, worker: int, rows) -> None:
        for kind in ('stage', 'entry'):
            for r in (rows or {}).get(kind, ()):
                self.rows.append((worker, kind, r))

    def pending(self, worker: int):
        new = [(k, r) for w, k, r in self.rows[self.sent.get(worker, 0):] if w != worker]
        self.sent[worker] = len(self.rows)
        if not new:
            return None
        return {kind: [r for k, r in new if k == kind] for kind in ('stage', 'entry')}


def share_calibration(env) -> dict:
    """Worker side: from now on every row this environment adds to its own stage / entry calibration is also put in
    the returned outbox (sent to the learner and emptied after each step)."""
    outbox = {'stage': [], 'entry': []}
    for kind, obj in (('stage', env.calibration), ('entry', env.entry_calibration)):
        def add(*a, _orig=obj.add, _obj=obj, _kind=kind, **k):
            _orig(*a, **k); outbox[_kind].append(_obj.rows[-1])
        obj.add = add
    return outbox


def merge_calibration(env, rows) -> None:
    """Worker side: the other workers' rows, appended in arrival order (the same sliding window as the own rows)."""
    from env.t2_stage import CALIBRATION_WINDOW
    for kind, obj in (('stage', env.calibration), ('entry', env.entry_calibration)):
        if rows and rows.get(kind):
            obj.rows = (obj.rows + list(rows[kind]))[-CALIBRATION_WINDOW:]


def worker(conn, rank: int, seed: int, candidates: int, entry_stages: int, window: bool, urgent_deadline=None,
           obs_profile: str = 'engineered', pref: bool = False, cand_min_sep: float = 0.0, label_deterministic: bool = False,
           ref_labels: bool = False) -> None:
    """One environment: send (transition or None, observation to act on), receive the action; until 'stop'."""
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    th.set_num_threads(1)
    from env.t2_stage import T2Mission, T2StageEnv
    mission = T2Mission() if urgent_deadline is None else T2Mission(urgent_deadline_s=tuple(float(x) for x in urgent_deadline))
    env = T2StageEnv(seed_base=1_000_000 * (seed + 1) + rank * 100_000, candidates=candidates, shaping=True, seed=seed * 100 + rank,
                     entry_stages=entry_stages, window=window, mission=mission, obs_profile=obs_profile, pref=pref, cand_min_sep=cand_min_sep)
    env.ref_labels = bool(ref_labels)            # learner 6.0: the policy's own intent through the candidates' treatment
    obs, _ = env.reset(); transition = None
    actor = None
    guide_rng = np.random.default_rng([seed, rank, 2707])        # T2.7 guidance draws (own stream)
    outbox = share_calibration(env) if (POOL_CALIBRATION and (candidates > 0 or pref)) else None

    def label_policy(o):          # mu: this environment's copy of the policy (T2.5 stage 2 of a counterfactual label)
        with th.no_grad():
            if label_deterministic:              # learner 5.0 (e_term2): stage 2 under the policy's deterministic intent
                a = actor(th.as_tensor(np.asarray(o, np.float32)[None]), deterministic=True)
                return a[0].numpy(), 0.0
            a, logp = actor.action_log_prob(th.as_tensor(np.asarray(o, np.float32)[None]))
        return a[0].numpy(), float(logp.reshape(-1)[0])
    while True:
        conn.send((transition, obs, {k: list(v) for k, v in outbox.items()} if outbox is not None else None))
        if outbox is not None:
            outbox['stage'].clear(); outbox['entry'].clear()
        msg = conn.recv()
        if isinstance(msg, str):
            break
        action, payload, shared, guide_p, a_det = (*msg, None) if len(msg) == 4 else (msg if len(msg) == 5 else (*msg, 0.0, None))
        env.policy_action = a_det                 # T2.11: the policy's deterministic intent, a candidate of the advantage
        if shared is not None:
            merge_calibration(env, shared)
        if payload is not None:                   # ('actor', module) once, then ('state', state_dict) every SYNC_UPDATES
            if payload[0] == 'actor':
                actor = payload[1]; actor.set_training_mode(False); env.label_policy = label_policy
            else:
                actor.load_state_dict(payload[1])
        guided = False
        if guide_p > 0 and guide_rng.uniform() < guide_p:           # T2.7: the executor's projected capture plan flies instead
            a_exec = env.track_intent()
            if a_exec is not None:
                action, guided = np.asarray(a_exec, np.float32), True
        next_obs, reward, term, trunc, info = env.step(action)
        info = dict(info, guided=guided)
        transition = (obs, next_obs, np.asarray(action, np.float32), float(reward), bool(term), bool(trunc), _slim(info))
        obs = env.reset()[0] if (term or trunc) else next_obs
    conn.close()


def train(out: Path, arm: str, seed: int, steps: int, workers: int, marks: list[int], smoke: bool = False,
          updates_per_transition: float = UPDATES_PER_TRANSITION, entry_stages: int | None = None, window: bool = True,
          model_beta: float = MODEL_BETA, urgent_deadline=None, guide: bool = False, obs_profile: str = 'engineered') -> None:
    import multiprocessing as mp
    from multiprocessing.connection import wait
    from stable_baselines3.common.logger import configure
    from env.t2_stage import CANDIDATE_MIN_SEP_M_S, ENTRY_STAGES, T2_VERSION
    entry_stages = ENTRY_STAGES if entry_stages is None else int(entry_stages)
    if arm in ('e_rel', 'e_priv', 'e_term', 'e_term2', 'e_pi', 'e_pit') and obs_profile != 'reduced_priv':
        raise ValueError(f'arm {arm} needs --obs reduced_priv')
    if arm == 'e_rel':
        entry_stages = 1          # T2.12: one-stage relative labels; the stage end is valued by the critic that reads the executor
    if arm in TERM_ARMS:
        entry_stages = TERM_ARMS[arm]     # learner 5.0: the counterfactual horizon in stages
    if arm in FB_ARMS:
        entry_stages = 1                  # learner 6.0: one-stage executor evidence (terminal within the stage, or the stage end)
    out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    th.set_num_threads(1)
    model = make_model(SpacesEnv(OBS_DIM_PRIV if obs_profile == 'reduced_priv' else OBS_DIM), arm, seed, smoke,
                       updates_per_transition=updates_per_transition, model_beta=model_beta)
    model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
    candidates = K_CANDIDATES if (arm in LABEL_ARMS or arm in FB_ARMS) else 0      # the SAC arm (and e_priv) computes no model
    pref = arm in ('e_pref', 'e_acc')                        # T2.10 preference / T2.11 EACC-A opportunity judgements
    cand_min_sep = CANDIDATE_MIN_SEP_M_S if arm in ('e_rel', 'e_term', 'e_term2', 'e_pi', 'e_pit') else 0.0     # candidates separated in the executor's terms
    label_det = arm in TERM_ARMS                                         # learner 5.0: stage 2 under the policy's mean intent
    (out / 'manifest.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'steps': steps, 'workers': workers, 'git_head': head, 'smoke': smoke,
                                                   'model_beta': model.model_beta, 'model_start': model.model_start, 'ent_coef': ENT_COEF,
                                                   'k_candidates': candidates, 'collection': 'asynchronous', 'window': window,
                                                   'gradient_steps_per_transition': model.gradient_steps, 'entry_stages': entry_stages, 'sync_updates': SYNC_UPDATES, 'age_cap': AGE_CAP,
                                                   'pooled_calibration': bool(POOL_CALIBRATION and (candidates > 0 or pref)),
                                                   'urgent_deadline_s': list(urgent_deadline) if urgent_deadline is not None else [170.0, 210.0],
                                                   'guide': bool(guide), 'guide_p0': GUIDE_P0 if guide else 0.0, 'guide_until': GUIDE_UNTIL if guide else 0,
                                                   'obs_profile': obs_profile, 'pref': pref, 'acc': arm == 'e_acc', 'relative': arm == 'e_rel', 'rel_beta': REL_BETA if arm == 'e_rel' else 0.0,
                                                   'cand_min_sep_m_s': cand_min_sep, 'common_random_numbers': arm == 'e_rel',
                                                   'terminal_horizon': TERM_ARMS.get(arm, 0), 'term_beta': TERM_BETA if arm in TERM_ARMS else 0.0,
                                                   'label_policy': 'deterministic' if label_det else 'sample', 'feedback': FB_ARMS.get(arm),
                                                   'fb_radius': FB_RADIUS if arm in FB_ARMS else None, 'fb_ridge': FB_RIDGE if arm in FB_ARMS else None,
                                                   'fb_clip': FB_CLIP if arm in FB_ARMS else None, 'fb_window': FB_WINDOW if arm in FB_ARMS else None,
                                                   'fb_batch': FB_BATCH if arm in FB_ARMS else None, 'fb_min_spread': FB_MIN_SPREAD if arm in FB_ARMS else None,
                                                   'actor_view': OBS_DIM, 'critic_view': int(getattr(model.policy, 'critic_dim', OBS_DIM)),
                                                   'acc_dp': ACC_DP, 'acc_kappa': ACC_KAPPA, 'acc_lambda_q': ACC_LAMBDA_Q, 'acc_eta': ACC_ETA,
                                                   'acc_min_labels': ACC_MIN_LABELS, 'acc_min_acc': ACC_MIN_ACC,
                                                   'pref_lambda': PREF_LAMBDA if pref else 0.0, 'pref_delta': PREF_DELTA, 'pref_kappa': PREF_KAPPA,
                                                   'pref_margin_cap': PREF_MARGIN_CAP,
                                                   'env_version': T2_VERSION, 'learner_version': LEARNER_VERSION}, indent=1), encoding='utf-8')
    ctx = mp.get_context('spawn')
    pipes, procs = [], []
    for rank in range(workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, rank, seed, candidates, entry_stages, window, urgent_deadline, obs_profile, pref, cand_min_sep, label_det,
                                                       arm in FB_ARMS), daemon=True)
        p.start(); pipes.append(a); procs.append(p)
    marks = sorted(set([m for m in marks if m <= steps] + [steps]))
    model.save(out / 'checkpoints' / 'step_0')
    episodes, order, t0 = [], [], time.time()
    bias_by_worker = {}
    synced = {}                                   # worker -> update count of its policy copy
    rng = np.random.default_rng(seed)
    pool = CalibrationPool() if (POOL_CALIBRATION and (candidates > 0 or pref)) else None
    guided_by_worker = {}                         # T2.7: guided decisions of each worker's current episode
    cal_pending = {}                              # T2.11: each worker's executed-decision features of the current episode
    ret_true, ret_shaped = {}, {}                 # T2.12: each worker's current episode return (true task reward / shaped training reward)
    active = list(pipes)
    while active:
        for conn in wait(active):
            transition, obs, rows = conn.recv()
            if pool is not None:
                pool.add(pipes.index(conn), rows)
            if transition is not None and model.num_timesteps < steps:
                o, o1, act, r, term, trunc, info = transition
                done = term or trunc
                info = dict(info, **({'TimeLimit.truncated': True} if trunc and not term else {}))
                model.replay_buffer.add(o[None], o1[None], act[None], np.array([r]), np.array([done]), [info])
                model.num_timesteps += 1; order.append(pipes.index(conn))
                w_ = pipes.index(conn)
                ret_true[w_] = ret_true.get(w_, 0.0) + float(info.get('true_reward') or 0.0); ret_shaped[w_] = ret_shaped.get(w_, 0.0) + float(r)
                if info.get('guided'):
                    guided_by_worker[pipes.index(conn)] = guided_by_worker.get(pipes.index(conn), 0) + 1
                    c_ = model.replay_buffer.model_counts; c_['guided'] = c_.get('guided', 0) + 1
                b = (info['model']['diag'] or {}).get('entry_bias')
                if b is not None:                      # latest known outcome-model bias of each environment
                    bias_by_worker[pipes.index(conn)] = b
                    model.entry_bias_now = float(np.mean(list(bias_by_worker.values())))
                if model.acc and (info.get('pref') or {}).get('f'):
                    cal_pending.setdefault(pipes.index(conn), []).append(dec_features(info['pref']['f'][PREF_EXEC]))
                if done:
                    if model.acc and info.get('event') in ('success', 'timeout', 'resource_exhausted', 'unsafe'):
                        for fx in cal_pending.get(pipes.index(conn), []):     # predict-then-observe, then add
                            model.dec_cal.observe(fx, info.get('event') == 'success')
                    cal_pending.pop(pipes.index(conn), None)
                    episodes.append({'steps': int(model.num_timesteps), 'worker': pipes.index(conn), **{k: info.get(k) for k in ('event', 'time_s', 'delta_v', 'mission')},
                                     'entry_bias': (info['model']['diag'] or {}).get('entry_bias'), 'guided': guided_by_worker.pop(pipes.index(conn), 0),
                                     'return_true': ret_true.pop(pipes.index(conn), 0.0), 'return_shaped': ret_shaped.pop(pipes.index(conn), 0.0)})
                if model.num_timesteps >= model.learning_starts:
                    model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)
                while marks and model.num_timesteps >= marks[0]:
                    m = marks.pop(0); model.save(out / 'checkpoints' / f'step_{m}')
                    (out / 'heartbeat.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'wall_s': time.time() - t0, 'episodes': len(episodes),
                                                                    'updates': int(model._n_updates), 'replay': model.replay_buffer.model_counts,
                                                                    'calibration': None if model.dec_cal is None else {'n': len(model.dec_cal.y), 'accuracy': model.dec_cal.accuracy,
                                                                                                                       'trusted': model.dec_cal.trusted, 'w': model.dec_cal.w.tolist()}}), encoding='utf-8')
                    # logs at every checkpoint (a stopped run keeps them; T2.3 seed 1 lost its logs)
                    (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
                    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
            if model.num_timesteps >= steps:
                conn.send('stop'); active.remove(conn); continue
            if model.num_timesteps < model.learning_starts:
                action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                action = model.predict(obs, deterministic=False)[0]
            payload, w_ = None, pipes.index(conn)
            if candidates > 0 and entry_stages >= 2:     # refresh the environment's policy copy (stage 2 of the labels)
                if w_ not in synced:
                    payload = ('actor', copy.deepcopy(model.actor).cpu())
                elif model._n_updates - synced[w_] >= SYNC_UPDATES:
                    payload = ('state', {k: v.detach().clone() for k, v in model.actor.state_dict().items()})
                if payload is not None:
                    synced[w_] = model._n_updates
            a_det = model.predict(obs, deterministic=True)[0] if (model.acc or model.feedback) else None   # the policy's own intent (T2.11, learner 6.0)
            conn.send((action, payload, pool.pending(w_) if pool is not None else None,
                       guide_probability(model.num_timesteps, GUIDE_P0, GUIDE_UNTIL) if guide else 0.0, a_det))
    for p in procs:
        p.join(timeout=60)
    model.save(out / 'final')
    (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
    (out / 'finished.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'updates': int(model._n_updates), 'wall_s': time.time() - t0,
                                                   'worker_order': order, 'replay': model.replay_buffer.model_counts,
                                                   'calibration': None if model.dec_cal is None else {'n': len(model.dec_cal.y), 'accuracy': model.dec_cal.accuracy,
                                                                                                      'trusted': model.dec_cal.trusted, 'w': model.dec_cal.w.tolist()}}), encoding='utf-8')
