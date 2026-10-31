import json, sys
import numpy as np
import env.t1_p_stage as st
_mk = st.make_conditional_gstar
st.make_conditional_gstar = lambda env, **kw: _mk(env, deadline_margin_s=float(sys.argv[3]), **kw)
from env.t1_p_review import T1PReviewEnv, geometric_rule
k = int(sys.argv[1]); env = T1PReviewEnv(openings=[k]); env.reset(); done = False; modes = []; minr = 1e9
while not done:
    a = geometric_rule(env) if env._due else np.zeros(3)
    _, _, term, trunc, info = env.step(a); done = term or trunc
    modes.append((env.env.time_s, info["execution"]["mode"])); minr = min(minr, float(env.env.metrics().range_m))
json.dump({"opening": k, "margin": float(sys.argv[3]), "event": info.get("event"), "violations": info.get("episode_summary", {}).get("violations"),
           "first_bridge_s": next((t for t, m in modes if m == "bridge"), None), "first_descent_s": next((t for t, m in modes if m == "descent"), None),
           "min_range_m": minr, "time_s": env.env.time_s}, open(sys.argv[2], "w"))
