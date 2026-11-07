"""T1-B V3 interface: the arrival time is a committed intent (task, D/B, G*, MPC and truth unchanged).

V1/V2 asked SAC again every 6 s while a plan was live, with the "candidates" [remaining - 15,
remaining, remaining + 15]: 12.7 of the 13.9 learning decisions per V2 episode were such relative
shifts. A late arrival then survives only if ~15 consecutive stochastic shifts agree, the critic
learns the local value of advance/keep/delay under the noisy policy, and its equilibrium is
"advance" (evidence/T1B_V3_PRECHECK). The candidate truth scan, by contrast, commits and keeps.

V3: a learning decision exists only when no plan is live (the initial review and after a plan is
lost: candidates [first, first + 20, first + 40] verified by G*); while a plan is live the only
admissible request is that plan's own arrival (keep; G* re-plans the same arrival after a
deviation exactly as before). The real SMDP interval of a decision therefore runs until the plan
is lost or the episode ends; when it ends at a terminal its target is the Monte Carlo return of the
committed intent. The paired MPC counterfactual is taken at the same decisions: both branches are
fresh G* plans, and the bootstrap at the alternative's predicted successor is the critic's value of
a committed state, learned from every real step of every committed trajectory.
"""
from time import perf_counter

import numpy as np

from env.t1_b_cf import NominalRK4Model
from env.t1_b_feasible_v1 import FeasiblePredictor
from env.t1_b_interval_cf import IntervalModelConfig
from env.t1_counterfactual import CounterfactualPredictor
from env.t1_b_feasible_v2 import FeasibleArrivalEnvV2
from env.t1_gstar import _plan_position, free_body_rotations

VERSION = 't1b-committed-intent-sac-v3'


class FeasibleArrivalEnvV3(FeasibleArrivalEnvV2):
    def __init__(self, *args, feedback=False, **kwargs):
        super().__init__(*args, feedback=False, **kwargs)
        self.feedback_enabled = feedback
        if feedback:
            self.predictor = FeasiblePredictorV3(self)

    def support(self):
        e, g = self.env, self.generator
        if g.remaining_s(e) <= 0:
            return super().support()
        if not self._due or g._descending(e, e.metrics()):
            return ()
        key = ('v3_keep', e.time_s, e._delta_v, e.chaser_state.position.tobytes())
        if self._support_cache and self._support_cache[0] == key:
            return self._support_cache[1]
        start = perf_counter(); tau = float(g.remaining_s(e))
        q = e.chaser_state.position - e.target_state.position
        v = e.chaser_state.rotation @ e.chaser_state.velocity - e.target_state.rotation @ e.target_state.velocity
        valid, audit = [], []
        if g.plan is not None and np.linalg.norm(_plan_position(g.plan, e.time_s - g.plan_start_s, g.planner.mass) - q) <= g.replan_deviation_m:
            valid.append(tau); audit.append({'tau': tau, 'status': 'kept_verified'})
        else:
            history = free_body_rotations(e.target_state.rotation, e.target_state.omega, e.target_parameters.inertia, tau + 1.)
            result = g.planner.solve(q, v, tau, history, None, entry_speed_m_s=g.entry_speed_m_s); self.support_solves += 1
            audit.append({'tau': tau, 'status': result['status']})
            if result['feasible']:
                valid.append(tau)
        self.support_wall += perf_counter() - start
        self._support_cache = (key, tuple(valid), audit)
        return tuple(valid)


class FeasiblePredictorV3(FeasiblePredictor):
    """The V1 predictor with a V3 shadow (keep semantics inside the prefix), nominal RK4 MPC model."""

    def __init__(self, real):
        CounterfactualPredictor.__init__(self, shadow_factory=lambda: FeasibleArrivalEnvV3(
            config=real.config, env_config=real.env.config, mission=real.mission_distribution,
            mission_override=real.mission_override, earliest_summary=real.earliest_summary, feedback=False))
        old = self.shadow.env.controller.reference_model
        self.shadow.env.controller.reference_model = NominalRK4Model(
            target_parameters=old.target_parameters, chaser_parameters=old.chaser_parameters, dt_s=old.dt_s,
            gravity_options=old.gravity_options, solver_settings=old.solver_settings, unwrap_attitude=old.unwrap_attitude)
        self.interval_config = IntervalModelConfig()
