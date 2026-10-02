"""Exploratory diagnostics after the stage C C_STOP verdict (2026-10-02).

Not a gate and not preregistered; it informs the discussion only. Reads the
stage C delivery (C1 records, B1/B2 scans) from the path set in C below.
"""
import json, numpy as np
from pathlib import Path
from experiments.v3_stage_c import (_scans, build_dataset, feature_index, fold_of, HandoffClassifier,
                                    labels_from_scans, first_trigger, N_FOLDS)
C = Path("/tmp/claude-0/-home-user-spacecraft/a72fce5d-f284-5874-bb9e-28e8a164f362/scratchpad/c")
models = ("262420", "262421", "262422")
rec = {m: dict(np.load(C/f"results/c1_{m}.npz")) for m in models}
meta = json.loads((C/"results/c1_262420.json").read_text())
feat = feature_index({n: slice(*b) for n, b in meta["observation_slices"].items()})
b1 = {m: _scans(str(C/f"inputs/stage_b/{m}")) for m in models}
b2 = {m: _scans(str(C/f"inputs/stage_b2/{m}")) for m in models}

# 1. label contamination: completed but not clean (violation somewhere in the episode)
for name, src in (("B1", b1), ("B2", b2)):
    tot = comp_viol = 0
    for m in models:
        for s, sc in src[m].items():
            if not sc["scanned"]: continue
            for h in sc["handoffs"]:
                tot += 1; comp_viol += int(h["completed"] and not h["zero_violation"])
    print(f"{name}: handoff states {tot}, completed-but-violated {comp_viol}")

# 2. label flips between consecutive decisions on B1 (stride 1)
flips = pairs = 0
for m in models:
    for s, sc in b1[m].items():
        if not sc["scanned"]: continue
        lab = [h["clean_completion"] for h in sc["handoffs"]]
        pairs += len(lab) - 1; flips += sum(a != b for a, b in zip(lab, lab[1:]))
print(f"B1 adjacent-decision label flips: {flips}/{pairs} = {flips/pairs:.1%}")

# 3. out-of-fold first-trigger behaviour across tau (exploratory)
data = build_dataset(rec, b1, b2, feat)
folds = np.array([fold_of(s) for s in data["seed"]])
fm = {f: HandoffClassifier.fit(data["x"][folds != f], data["y"][folds != f], data["w"][folds != f], feat) for f in range(N_FOLDS)}
# per-source precision at a few tau
oof = np.zeros(len(data["y"]))
for f in range(N_FOLDS): oof[folds == f] = fm[f].predict_features(data["x"][folds == f])
for src in ("B1", "B2"):
    sel = data["source"] == src
    print(src, "states", int(sel.sum()), "pos frac (w)", round(float((data["w"][sel]*data["y"][sel]).sum()/data["w"][sel].sum()), 3))
for tau in (0.8, 0.9, 0.95, 0.99):
    out = {"B1": {"rescued": 0, "premature_fail": 0, "never": 0}, "B2": {"kept_never": 0, "clean": 0, "destroy": 0, "off_grid": 0}}
    for m in models:
        r = rec[m]
        for s in sorted(set(r["seed"].tolist())):
            labels = labels_from_scans(b1[m].get(s), b2[m].get(s))
            if not labels: continue
            mask = r["seed"] == s
            p = fm[fold_of(s)].predict_features(r["obs"][mask][:, feat])
            hit = first_trigger(p, r["k"][mask], labels, tau)
            if b1[m][s]["scanned"]:
                key = "never" if hit["k"] is None else ("rescued" if hit["label"] else "premature_fail")
                out["B1"][key] += 1
            else:
                key = "kept_never" if hit["k"] is None else {True: "clean", False: "destroy", None: "off_grid"}[hit["label"]]
                out["B2"][key] += 1
    print("tau", tau, out)

# 4. where do first triggers land, and how good is p at k=0 alone?
tau = 0.95
ks = []
for m in models:
    r = rec[m]
    for s in sorted(set(r["seed"].tolist())):
        mask = r["seed"] == s
        p = fm[fold_of(s)].predict_features(r["obs"][mask][:, feat])
        hit = first_trigger(p, r["k"][mask], {}, tau)
        ks.append(hit["k"])
ks_n = [k for k in ks if k is not None]
print("first-trigger k at tau 0.95: k=0:", sum(k == 0 for k in ks_n), " 1-5:", sum(1 <= k <= 5 for k in ks_n), " >5:", sum(k > 5 for k in ks_n), " never:", ks.count(None))
k0 = data["k"] == 0
y0, p0 = data["y"][k0], oof[k0]
pos, neg = p0[y0 > .5], p0[y0 < .5]
auc0 = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
print(f"k=0 states: {int(k0.sum())}, Pure-from-reset clean {int(y0.sum())}, OOF AUC at k=0 {auc0:.2f}")

# 5. exploratory: states after the opening (k >= 1) only
from experiments.v3_stage_c import choose_tau
late = data["k"] >= 1
yl, pl, wl = data["y"][late], oof[late], data["w"][late]
pos, neg = pl[yl > .5], pl[yl < .5]
pw, nw = wl[yl > .5], wl[yl < .5]
auc_late = float((((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])) * pw[:, None] * nw[None, :]).sum() / (pw.sum() * nw.sum()))
tau_l, curve_l = choose_tau(pl, yl, wl)
print(f"k>=1 states {int(late.sum())}: weighted OOF AUC {auc_late:.2f}; precision at tau 0.9/0.99:",
      [round(c['weighted_precision'], 3) for c in curve_l if c['tau'] in (0.9, 0.99)], "tau>=0.95 found:", tau_l)
for tau in (0.9, 0.99):
    out = {"B1": {"rescued": 0, "premature_fail": 0, "never": 0}, "B2": {"kept_never": 0, "clean": 0, "destroy": 0, "off_grid": 0}}
    for m in models:
        r = rec[m]
        for s in sorted(set(r["seed"].tolist())):
            labels = labels_from_scans(b1[m].get(s), b2[m].get(s))
            mask = (r["seed"] == s) & (r["k"] >= 1)
            p = fm[fold_of(s)].predict_features(r["obs"][mask][:, feat])
            hit = first_trigger(p, r["k"][mask], labels, tau)
            if b1[m][s]["scanned"]:
                out["B1"]["never" if hit["k"] is None else ("rescued" if hit["label"] else "premature_fail")] += 1
            else:
                out["B2"]["kept_never" if hit["k"] is None else {True: "clean", False: "destroy", None: "off_grid"}[hit["label"]]] += 1
    print("start learned, hand off from k>=1, tau", tau, out)
