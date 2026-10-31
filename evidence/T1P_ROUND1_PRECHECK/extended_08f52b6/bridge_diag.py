import json, sys
import numpy as np
import env.t1_p_stage as st
from env.t1_p_review import T1PReviewEnv, geometric_rule
rows = []
orig = st.T1PGenerator.try_bridge
def wrapped(self, env, m):
    q, _ = st.relative_inertial(env)
    los = q / np.linalg.norm(q)
    t = env.target_state
    hist = st.free_body_rotations(t.rotation, t.omega, env.target_parameters.inertia, 41.0)
    ang = [float(np.degrees(np.arccos(np.clip((st._rotation_at(hist, tau)[0] @ env.config.task.approach_axis) @ los, -1, 1)))) for tau in st.BRIDGE_TAUS_S]
    ok = orig(self, env, m)
    rows.append({"t": env.time_s, "range": float(m.range_m), "margin": float(m.radial_envelope_margin_m_s), "angles": ang,
                 "bridge_status": getattr(self.bridge, "last_status", None), "ok": ok})
    return ok
st.T1PGenerator.try_bridge = wrapped
k = int(sys.argv[1])
env = T1PReviewEnv(openings=[k]); env.reset(); done = False
while not done:
    a = geometric_rule(env) if env._due else np.zeros(3)
    _, _, term, trunc, info = env.step(a); done = term or trunc
json.dump({"event": info.get("event"), "rows": rows}, open(sys.argv[2], "w"))
