"""M1 training-time counterfactual: one alternative intent, one isolated MPC solve.

Design: "T1 SAC–MPC coupling design, upper hand-off final version, 2026-10-12",
section 5. Nothing here changes the real branch: the alternative is evaluated
in a *shadow* copy of the wrapper (its own ``T1Env``, MPC controller and G*
planner instances), synchronised from the real one at the start of each 2 s
decision (state, clock, episode bookkeeping, G* plan memory, the wrapper's
``_last_request``/``_last_kappa``, and the controller's warm start, drift,
step counter and held reference). The CVXPY problems are never shared or
deep-copied; CLARABEL keeps no solver-side warm start, so equal inputs give an
equal first QP (checked bitwise in the tests).

Per decision k (real action a_k, alternative a~_k):

* ``predict(a)``: G* decides for ``a`` (tau, kappa and, interface v4, the entry speed) in the shadow, one MPC solve at the
  current state, then the solve's nominal open-loop prediction over the first
  20 control steps (the controller's own ``_rollout`` of its nominal controls,
  read only). Predicted reward with the real metering
  ``-0.05 * 2 - 5 * sum 0.1/106 |clip(F_i)|``; predicted target from the
  MPC's own target model (20 x 0.1 s); the chaser reconstructed from the
  predicted relative state; the observation built by the *same*
  ``T1GStarEnv._observation`` of the shadow after its memory and clock have
  been advanced for ``a`` (so ``tau_request`` and ``kappa`` are those of
  ``a``, never of the real action).
* the real branch runs unchanged (20 MPC steps, RK45 truth);
* errors of the real action's own nominal prediction against what happened:
  ``e_r = |r^ - r|``, ``e_p = |p^' - p'|`` (relative ECI position),
  ``e_o = |z^' - z'|_2`` (normalised observation, all dims of the interface);
* the gate ``w`` in {0, 1} with the reasons it was closed.

This is a one-QP open-loop prediction, not 20 receding solves: it can be wrong
under saturation, mode switches or model error, which is what the gate and the
closed-loop check (``closed_loop``) are for.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc.prediction import relative_to_vector
from dynamics.lie import se3_exp
from dynamics.relative import relative_state
from env.t1_env import T1EnvConfig
from env.t1_gym import OBSERVATION_NAMES, T1GStarEnv, T1GymConfig

_ENV_FIELDS = (
    "time_s", "decision", "anchor", "_capture_streak", "_delta_v", "_fallbacks", "_control_steps",
    "_saturated_steps", "_max_range", "_min_margins", "_last_reference", "_reference_jump_max", "_last_intent",
)
_CONTROLLER_FIELDS = (
    "_nominal_controls", "_drift", "_exact_linearization", "_control_step", "_episode_plan_initial_state",
    "_episode_plan_start_time_s", "_held_external_reference", "_held_external_velocity",
)
_GENERATOR_FIELDS = (
    "kappa", "pure_approach", "tau_min_s", "descent_radius_m", "replan_tau_tolerance_s", "replan_deviation_m",
    "slips_s", "earliest_grid_s", "tau_s", "tau_effective_s", "plan_start_s", "last_status",
    "deadline_margin_s", "last_deadline", "entry_speed_m_s",
)
_TRACKING = ("optimal", "optimal_inaccurate", "kept")


@dataclass(frozen=True)
class CounterfactualConfig:
    """Alternative-intent rule and gate. Thresholds marked (B) are frozen from validation B."""

    shift_s: float = 15.0
    min_tau_difference_s: float = 4.0
    #: entry-speed counterfactual (interface v4): the alternative moves a[2] by this much
    #: (towards the other half of the range: +1 if a[2] < 0 else -1, i.e. 0.04 m/s), and the
    #: two accepted plans must differ in entry speed by at least ``min_entry_speed_difference``
    entry_speed_shift: float = 1.0
    min_entry_speed_difference_m_s: float = 0.02
    max_predicted_saturation: float = 0.5
    end_margin_s: float = 4.0
    max_solve_slack: float = 1.0e-3
    #: (B) bounds on the executed action's own nominal-prediction error. Proposed from
    #: validation B (evidence/T1_M1_validationB): ~p95 of the executed-branch errors over
    #: the 149 structurally open states (0.0585, 0.0158 m, 0.0708). Frozen only on approval.
    max_reward_error: float = 0.06
    max_position_error_m: float = 0.02
    max_observation_error: float = 0.07


@dataclass
class Prediction:
    action: np.ndarray
    tau_request_s: float
    kappa: float
    plan_status: str
    tau_effective_s: float
    first_wrench: np.ndarray
    solved: bool
    solve_slack: float
    reward: float
    saturation: float
    observation: np.ndarray
    relative_position_eci_m: np.ndarray
    end_descending: bool
    end_mouth_or_cone: bool
    end_violation: bool
    max_range_m: float
    #: the whole predicted 2 s window (steps 1..20), not only its end
    window_violation: bool
    window_violation_kinds: list[str]
    window_first_violation_step: int | None
    window_mode_switch: bool
    window_first_mode_switch_step: int | None
    wall_s: float
    extra: dict[str, Any] = field(default_factory=dict)


def mirror_solver_cache(src_problem, dst_problem) -> None:
    """CVXPY keeps the solver object between solves; the first solve after a
    fresh build differs from a cached one at ~1e-7 (see MPCController.reset).
    Mirror "fresh" vs "cached" so that the shadow's solve is the real one: a
    real problem without a cache (a just-reset controller, a planner built at
    reset) must be shadowed by one without a cache (found in validation B)."""

    if not getattr(src_problem, "_solver_cache", None):
        cache = getattr(dst_problem, "_solver_cache", None)
        if cache is not None:
            cache.clear()


def copy_controller_state(src, dst) -> None:
    mirror_solver_cache(src._problem, dst._problem)
    for name in _CONTROLLER_FIELDS:
        setattr(dst, name, deepcopy(getattr(src, name)))
    for a, b in ((src._x, dst._x), (src._u, dst._u)):
        b.value = None if a.value is None else np.array(a.value)
    if src._slack is not None and dst._slack is not None:
        dst._slack.value = None if src._slack.value is None else np.array(src._slack.value)


def copy_generator_state(src, dst) -> None:
    # (the G* planner solves from a fresh solver every time, v0.3.2: no cache to mirror)
    for name in _GENERATOR_FIELDS:
        setattr(dst, name, deepcopy(getattr(src, name)))
    dst.plan = deepcopy(src.plan)
    dst.hold_position = None if src.hold_position is None else np.array(src.hold_position)
    dst._descent = deepcopy(src._descent)
    dst.plan_wall_s = []


class CounterfactualPredictor:
    def __init__(self, gym_config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 config: CounterfactualConfig | None = None, shadow_factory=None) -> None:
        #: the shadow wrapper; ``shadow_factory`` builds one of the real wrapper's own class
        #: (e.g. the T1-L review wrapper), so the prediction runs the same interface code
        self.shadow = shadow_factory() if shadow_factory is not None else T1GStarEnv(gym_config, env_config)
        self.config = config or CounterfactualConfig()

    # ------------------------------------------------------------------ #
    def sync(self, real: T1GStarEnv, fresh_solvers: bool = False) -> None:
        """Copy the real wrapper's state into the shadow. ``fresh_solvers`` (T1-O branching)
        also drops the shadow MPC's solver cache: a cached CLARABEL object carries numerical
        state from its own solve history (~1e-8 after many solves), so a clone cannot be made
        bitwise equal to a long-running real branch; branches started from the same synced
        state with fresh solvers are bitwise equal to each other, and their difference to the
        real continuation is the measured numerical repeatability."""

        s, r = self.shadow, real
        if s.generator is None:
            s.reset(options={"opening": r.current_opening if r.current_opening is not None else 0})
        s.env.target_state = r.env.target_state.copy()
        s.env.chaser_state = r.env.chaser_state.copy()
        for name in _ENV_FIELDS:
            setattr(s.env, name, deepcopy(getattr(r.env, name)))
        copy_controller_state(r.env.controller, s.env.controller)
        if fresh_solvers:
            cache = getattr(s.env.controller._problem, "_solver_cache", None)
            if cache is not None:
                cache.clear()
        assert r.generator is not None and s.generator is not None
        copy_generator_state(r.generator, s.generator)
        s._last_request, s._last_kappa = r._last_request, r._last_kappa
        s.current_opening, s.episode = r.current_opening, r.episode
        if hasattr(r, "review_state"):  # T1-L review wrapper: clock, due flag, committed kappa / v_in
            s.set_review_state(r.review_state())

    def target_window(self) -> list:
        """Predicted target states over the next 2 s from the (synced) shadow's current target.

        Action-independent, so one propagation per decision serves both branches.
        """

        e = self.shadow.env
        model = e.controller.reference_model
        targets = [e.target_state]
        for i in range(e.config.decision_steps):
            targets.append(model.propagate_target(targets[-1], e.time_s + i * e.config.dt_s))
        return targets

    def predict(self, action: np.ndarray, targets: list | None = None) -> Prediction:
        """One G* decision and one MPC solve in the (already synced) shadow.

        ``targets``: the decision's ``target_window()`` (computed here when omitted).
        """

        started = perf_counter()
        sh = self.shadow
        e, g = sh.env, sh.generator
        assert g is not None
        c = e.config
        if targets is None:
            targets = self.target_window()
        if hasattr(sh, "review_state"):
            sh._continuing = False  # a prediction is always of a review decision
        tau, kappa, v_in = sh.apply_intent(action)
        g.decide(e, tau)
        sh._last_request, sh._last_kappa = tau, kappa
        m = e.metrics()
        waypoint, path = g(e, m)
        x0 = relative_to_vector(relative_state(e.target_state, e.chaser_state))
        controller = e.controller
        wrench, diagnostics = controller.command(
            x0, target_state=e.target_state, time_seconds=e.time_s,
            external_reference=waypoint, external_reference_path=path,
            terminal_latched=bool(m.mouth_region),
        )
        limits = np.array([*([c.max_torque_per_axis_nm] * 3), *([c.max_force_per_axis_n] * 3)])
        first_wrench = np.clip(np.asarray(wrench, dtype=np.float64), -limits, limits)
        steps = c.decision_steps
        controls = np.clip(np.asarray(controller._nominal_controls[:steps], dtype=np.float64), -limits, limits)
        states = controller._rollout(x0, controller._nominal_controls[:steps])
        force = controls[:, 3:]
        dv = float(np.sum(np.linalg.norm(force, axis=1))) * c.dt_s / e.chaser_parameters.mass
        reward = -c.reward.time_cost_per_s * c.dt_s * steps - c.reward.delta_v_cost_per_m_s * dv
        saturation = float(np.mean(np.max(np.abs(force), axis=1) >= 0.99 * c.max_force_per_axis_n))
        ranges = [float(np.linalg.norm(se3_exp(x[:6])[:3, 3])) for x in states]
        model = controller.reference_model
        target = targets[steps]
        window = check_predicted_window(e, g, targets, states)
        near = near_field_window(e, g, targets, states, m)
        chaser = model.reconstruct_chaser(target, states[steps])
        e.target_state, e.chaser_state = target, chaser
        e.time_s = round(e.time_s + steps * c.dt_s, 10)
        if hasattr(sh, "advance_review"):  # the review clock and due flag of the predicted successor
            sh.advance_review(True, kappa, v_in)
        observation = sh._observation()
        m_end = e.metrics()
        return Prediction(
            action=np.asarray(action, dtype=np.float64).copy(),
            tau_request_s=tau, kappa=kappa, plan_status=g.last_status, tau_effective_s=g.tau_effective_s,
            first_wrench=first_wrench, solved=not diagnostics.used_zero_fallback,
            solve_slack=float(diagnostics.maximum_slack), reward=float(reward), saturation=saturation,
            observation=observation, relative_position_eci_m=chaser.position - target.position,
            end_descending=bool(g._descending(e, m_end)),
            end_mouth_or_cone=bool(m_end.mouth_region or m_end.in_cone),
            end_violation=bool(m_end.hard_violation or m_end.range_m > c.task.max_distance_m),
            max_range_m=max(ranges), wall_s=perf_counter() - started, **window,
            extra={"entry_speed_plan_m_s": None if g.plan is None else float(g.plan.get("entry_speed_m_s", 0.06)),
                   "new_plan": g.last_status.startswith(("optimal", "recovery")),
                   # the 20 clipped nominal controls of the decision (for real-vs-alternative differences)
                   "controls": controls.copy(), **near},
        )

    def closed_loop(self, action: np.ndarray) -> dict[str, Any]:
        """The real 2 s closed loop of ``action`` in the (already synced) shadow (validation only)."""

        obs, reward, terminated, truncated, info = self.shadow.step(action)
        e = self.shadow.env
        return {
            "observation": obs, "reward": float(reward), "terminated": terminated, "truncated": truncated,
            "event": info["event"], "relative_position_eci_m": e.chaser_state.position - e.target_state.position,
            "execution": info["execution"],
        }

    # ------------------------------------------------------------------ #
    def alternative(self, real: T1GStarEnv, action: np.ndarray, real_prediction: Prediction) -> tuple[np.ndarray | None, str]:
        """The single alternative intent of the design's table (same kappa), or None with the reason."""

        g = real.generator
        assert g is not None
        if g._descending(real.env, real.env.metrics()):
            return None, "descent_mode"
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a.size < 3:
            a = np.concatenate([a, np.zeros(3 - a.size)])
        cfg, gym = self.config, real.config
        # interface v4: where the real action makes a new plan, every other decision (fixed
        # parity of episode + decision index) asks the single alternative to differ only in the
        # entry speed (same tau, kappa); everywhere else the tau alternative of the design table
        if real_prediction.extra.get("new_plan") and (real.episode + real.env.decision) % 2 == 0:
            alt = a.copy()
            alt[2] = a[2] + (cfg.entry_speed_shift if a[2] < 0.0 else -cfg.entry_speed_shift)
            return alt, "entry_speed"
        remaining = g.remaining_s(real.env)
        if remaining > 0.0:
            keeps = abs(real.shift_of(float(a[0]))) <= gym.keep_tolerance_s
            if not keeps:
                return np.array([0.0, a[1], a[2]]), "replan_vs_keep"
            sign = 1.0 if real.env.decision % 2 == 0 else -1.0
            if remaining - cfg.shift_s <= g.tau_min_s:
                sign = 1.0
            return np.array([real.action_for_shift(sign * cfg.shift_s), a[1], a[2]]), "keep_vs_shift"
        tau = gym.tau_max_s * 0.5 * (a[0] + 1.0)
        if tau > g.tau_min_s:
            return np.array([-1.0, a[1], a[2]]), "request_vs_earliest"
        later = real_prediction.tau_effective_s + cfg.shift_s
        a0 = 2.0 * later / gym.tau_max_s - 1.0
        if a0 > 1.0:
            return None, "not_representable"
        return np.array([a0, a[1], a[2]]), "earliest_vs_later"

    def gate(self, real: T1GStarEnv, p_real: Prediction, p_alt: Prediction, errors: dict[str, float],
             kind: str = "") -> tuple[int, list[str]]:
        cfg = self.config
        reasons = []
        if kind == "entry_speed":
            # physically distinct: both actions made a new plan, with entry speeds that differ
            v_real, v_alt = p_real.extra.get("entry_speed_plan_m_s"), p_alt.extra.get("entry_speed_plan_m_s")
            if (not p_alt.extra.get("new_plan") or v_real is None or v_alt is None
                    or abs(v_alt - v_real) < cfg.min_entry_speed_difference_m_s):
                reasons.append("same_entry_speed")
        elif abs(p_alt.tau_effective_s - p_real.tau_effective_s) < cfg.min_tau_difference_s:
            reasons.append("same_effective_tau")
        for name, p in (("alt", p_alt), ("real", p_real)):
            if p.plan_status not in _TRACKING:
                reasons.append(f"{name}_not_tracking:{p.plan_status.split(':')[0]}")
            if not p.solved:
                reasons.append(f"{name}_fallback")
            if p.solve_slack > cfg.max_solve_slack:
                reasons.append(f"{name}_slack")
            if p.window_mode_switch:
                reasons.append(f"{name}_mode_switch")
            if p.window_violation:
                reasons.append(f"{name}_predicted_violation:{'+'.join(p.window_violation_kinds)}")
        if p_alt.saturation > cfg.max_predicted_saturation:
            reasons.append("alt_saturation")
        # called after the real step: time_s is already the predicted end time t + 2 s
        if real.env.time_s > real.env.config.max_time_s - cfg.end_margin_s:
            reasons.append("near_horizon")
        if errors["e_r"] > cfg.max_reward_error:
            reasons.append("e_r")
        if errors["e_p"] > cfg.max_position_error_m:
            reasons.append("e_p")
        if errors["e_o"] > cfg.max_observation_error:
            reasons.append("e_o")
        return int(not reasons), reasons


def check_predicted_window(env, generator, targets: list, states: list) -> dict[str, Any]:
    """Truth rules on every predicted state of the 2 s window (design 2026-10-12 / P0-1).

    Each predicted relative state is turned into a chaser state against the
    predicted target of the same instant and checked with the truth metric
    (collision, sphere-or-cone union, FOV, radial / axial braking envelope,
    near-field total speed), the 30 m distance limit and finiteness. A step in
    the mouth region, in the cone, inside the sphere or in the descent region
    is a mode switch. Any violation or mode switch anywhere -> w = 0; a window
    that violates mid-way and recovers by its end is *not* accepted.
    """

    from env.t1_task import compute_t1_metrics

    task = env.config.task
    model = env.controller.reference_model
    kinds: list[str] = []
    first_violation = first_switch = None
    for i in range(1, len(states)):
        x = np.asarray(states[i], dtype=np.float64)
        step_kinds: list[str] = []
        if not np.all(np.isfinite(x)):
            step_kinds.append("non_finite")
        else:
            chaser = model.reconstruct_chaser(targets[i], x)
            m = compute_t1_metrics(targets[i], chaser, task)
            step_kinds.extend(sorted(m.violations))
            if m.range_m > task.max_distance_m:
                step_kinds.append("distance")
            if first_switch is None and (m.mouth_region or m.in_cone or generator._descending(env, m)):
                first_switch = i
        if step_kinds:
            first_violation = i if first_violation is None else first_violation
            kinds.extend(k for k in step_kinds if k not in kinds)
    return {
        "window_violation": first_violation is not None,
        "window_violation_kinds": kinds,
        "window_first_violation_step": first_violation,
        "window_mode_switch": first_switch is not None,
        "window_first_mode_switch_step": first_switch,
    }


def near_field_window(env, generator, targets: list, states: list, start_metrics) -> dict[str, Any]:
    """Mode changes of a predicted 2 s window that starts in G*'s descent mode (T1-O near-field
    kappa counterfactual). ``check_predicted_window`` counts every step in the cone or in the
    descent region as a switch, which is right for the far field but rejects every descent
    window; here a switch is a change against the window's start: entering or leaving the
    mouth region, or leaving the descent region. Read only."""

    from env.t1_task import compute_t1_metrics

    task = env.config.task
    model = env.controller.reference_model
    start_mouth = bool(start_metrics.mouth_region)
    first = None
    for i in range(1, len(states)):
        x = np.asarray(states[i], dtype=np.float64)
        if not np.all(np.isfinite(x)):
            first = i
            break
        m = compute_t1_metrics(targets[i], model.reconstruct_chaser(targets[i], x), task)
        if bool(m.mouth_region) != start_mouth or not generator._descending(env, m):
            first = i
            break
    return {"near_mode_switch": first is not None, "near_first_mode_switch_step": first}


def prediction_errors(p: Prediction, reward: float, observation: np.ndarray, relative_position_eci_m: np.ndarray) -> dict[str, float]:
    return {
        "e_r": abs(p.reward - float(reward)),
        "e_p": float(np.linalg.norm(p.relative_position_eci_m - relative_position_eci_m)),
        "e_o": float(np.linalg.norm(p.observation - observation)),
    }


class T1CounterfactualEnv(T1GStarEnv):
    """T1GStarEnv whose transitions are bitwise those of M0, plus ``info["counterfactual"]``."""

    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, counterfactual: CounterfactualConfig | None = None,
                 enabled: bool = True) -> None:
        super().__init__(config, env_config, openings)
        self.predictor = CounterfactualPredictor(config, env_config, counterfactual)
        self.enabled = enabled

    def step(self, action: np.ndarray):
        if not self.enabled:
            return super().step(action)
        started = perf_counter()
        assert self.generator is not None
        if self.generator._descending(self.env, self.env.metrics()):
            # the design disables the counterfactual in the descent mode (w = 0 whatever is
            # predicted), so nothing is predicted there
            obs, reward, terminated, truncated, info = super().step(action)
            info["counterfactual"] = {
                "kind": "descent_mode", "a_tilde": None, "r_hat": None, "z_hat": None, "w": 0,
                "reasons": ["descent_mode"], "errors_real": None, "real_first_wrench": None,
                "tau_effective_real_pred_s": None, "tau_effective_alt_s": None, "alt_status": None,
                "alt_saturation": None, "overhead_wall_s": perf_counter() - started,
            }
            return obs, reward, terminated, truncated, info
        pred = self.predictor
        pred.sync(self)
        targets = pred.target_window()
        p_real = pred.predict(action, targets)
        alt, kind = pred.alternative(self, action, p_real)
        p_alt = None
        if alt is not None:
            pred.sync(self)
            p_alt = pred.predict(alt, targets)
        overhead = perf_counter() - started
        obs, reward, terminated, truncated, info = super().step(action)
        rel = self.env.chaser_state.position - self.env.target_state.position
        errors = prediction_errors(p_real, reward, obs, rel)
        if p_alt is None or terminated or truncated:
            w, reasons = 0, [kind if p_alt is None else "real_terminated"]
        else:
            w, reasons = pred.gate(self, p_real, p_alt, errors, kind)
        info["counterfactual"] = {
            "kind": kind,
            "a_tilde": None if alt is None else alt.astype(np.float32),
            "r_hat": None if p_alt is None else p_alt.reward,
            "z_hat": None if p_alt is None else p_alt.observation,
            "w": w,
            "reasons": reasons,
            "errors_real": errors,
            "real_first_wrench": p_real.first_wrench,
            "tau_effective_real_pred_s": p_real.tau_effective_s,
            "tau_effective_alt_s": None if p_alt is None else p_alt.tau_effective_s,
            "alt_status": None if p_alt is None else p_alt.plan_status,
            "alt_saturation": None if p_alt is None else p_alt.saturation,
            "entry_speed_real_plan_m_s": p_real.extra.get("entry_speed_plan_m_s"),
            "entry_speed_alt_plan_m_s": None if p_alt is None else p_alt.extra.get("entry_speed_plan_m_s"),
            # how different the alternative's predicted execution is from the real action's
            "alt_first_wrench_diff": None if p_alt is None else float(np.linalg.norm(p_alt.first_wrench - p_real.first_wrench)),
            "alt_controls_diff_max": None if p_alt is None else float(np.max(np.linalg.norm(
                p_alt.extra["controls"] - p_real.extra["controls"], axis=1))),
            "alt_reward_diff": None if p_alt is None else float(p_alt.reward - p_real.reward),
            "alt_observation_diff": None if p_alt is None else float(np.linalg.norm(p_alt.observation - p_real.observation)),
            "overhead_wall_s": overhead,
        }
        return obs, reward, terminated, truncated, info


__all__ = [
    "CounterfactualConfig", "CounterfactualPredictor", "Prediction", "T1CounterfactualEnv",
    "check_predicted_window", "copy_controller_state", "near_field_window", "copy_generator_state", "mirror_solver_cache", "prediction_errors", "OBSERVATION_NAMES",
]
