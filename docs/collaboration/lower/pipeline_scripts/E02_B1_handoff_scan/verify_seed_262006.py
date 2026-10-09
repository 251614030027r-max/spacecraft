"""Read-only fidelity gate from Stage B1 run order section 3."""

import json
from pathlib import Path


root = Path(r"D:\py\DRL2")
scan = json.loads((root / "eval/v3e/stage_b/262420/seed_262006.json").read_text())
pure = {r["seed"]: r for r in json.loads((root / "eval/v3e/pure_mpc.json").read_text())["records"]}[262006]
formal = {r["seed"]: r for r in json.loads((root / "eval/v3e/262420/learned_only.json").read_text())["records"]}[262006]
k0 = scan["handoffs"][0]
checks = {
    "learned fails at decision 26 (as formal)": scan["learned_full"]["decisions"] == 26 == formal["decisions"]
    and not scan["learned_full"]["completed"]
    and not formal["completed"],
    "learned survival equals formal": scan["learned_full"]["survival_s"] == formal["survival_s"],
    "prefix verification is [0, 13, 25]": scan["verified_prefix_ks"] == [0, 13, 25],
    "k=0 completed equals Pure MPC": k0["completed"] == pure["completed"],
    "k=0 decisions equal Pure MPC": k0["decisions"] == pure["decisions"],
    "k=0 survival equals Pure MPC": k0["survival_s"] == pure["survival_s"],
    "26 handoffs scanned": [h["k"] for h in scan["handoffs"]] == list(range(26)),
    "clean checkout": scan["code_dirty"] is False,
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
print("ALL_PASS" if all(checks.values()) else "STOP_AND_REPORT")
raise SystemExit(0 if all(checks.values()) else 3)
