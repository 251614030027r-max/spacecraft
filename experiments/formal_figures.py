"""Render the pre-registered ECC figures and tables from saved experiment data."""
from __future__ import annotations

import argparse
import csv
import json
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.final import check_formal, registry, stage_of
from experiments.formal_readout import ECC, REFERENCE, opening_rows, read_json, success

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
STYLE = ("-", "--", "-.", ":")


def select_cases(ecc, sac, pure, nocert):
    triple = [o for o in set(ecc) & set(sac) & set(pure) if all(success(rows[o]) for rows in (ecc, sac, pure))]
    mechanism = [o for o in set(ecc) & set(nocert) if success(ecc[o]) and not success(nocert[o])]
    fallback = [o for o in set(ecc) & set(sac) if success(ecc[o]) and not success(sac[o])]

    def choice(primary, rule):
        if primary:
            return {"opening": min(primary), "rule": rule}
        return {"opening": min(fallback, default=None), "rule": "ECC success and SAC-MPC failure fallback"}

    return {"physical": choice(triple, "ECC, SAC-MPC and pure MPC all capture"),
            "mechanism": choice(mechanism, "ECC capture with certification only")}


def dump(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_plot(out, name, figure, payload, raster=False):
    for extension in (("png",) if raster else ("pdf", "svg")):
        figure.savefig(out / f"{name}.{extension}", dpi=300 if raster else None, bbox_inches="tight")
    dump(out / f"{name}.json", payload)
    plt.close(figure)


def write_table(out, name, headings, rows):
    with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        writer.writerows(rows)
    lines = ["| " + " | ".join(map(str, headings)) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(out / f"{name}.json", {"columns": headings, "rows": rows})


def show_missing(axis, detail):
    axis.text(.5, .5, detail, ha="center", va="center", transform=axis.transAxes)


def case_rows(root, name, opening):
    if opening is None:
        return None
    candidates = sorted((root / "TRAJ").glob(f"traj_{name}_{opening}.json"))
    return read_json(candidates[0]) if candidates else None


def main_figures(root, out, result, smoke=False):
    trained = registry()["trained"]
    formal = registry()["protocol"]["stages"]["formal"]
    names = {key: cfg["display"] for key, cfg in trained.items()}
    names.update(pure="Pure MPC", gstar="Direct entry planning")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    for i, (label, curve) in enumerate(result["curves"].items()):
        name = label.rpartition("_s")[0]
        if name not in formal["validation"]["trained"]:
            continue
        k = formal["validation"]["trained"].index(name)
        style = STYLE[i % len(STYLE)]
        if curve.get("steps"):
            axes[0].plot(curve["steps"], curve["true_return_smoothed"], label=label, color=COLORS[k], ls=style)
        val = sorted((int(step), row["capture_rate"]) for step, row in curve["validation"].items() if row["n"])
        if val:
            axes[1].plot(*zip(*val), marker=("o", "s", "^")[k], label=label, color=COLORS[k], ls=style)
    for ax, label in zip(axes, ("Training true return (50 episodes)", "Deterministic validation capture rate")):
        ax.set(xlabel="Decisions", ylabel=label)
        if ax.lines:
            ax.legend(fontsize=6)
        else:
            show_missing(ax, "No saved validation")
    write_plot(out, "F1", fig, result["curves"])

    seed = 10 if smoke else formal["seeds"][0]
    c = root / "C"
    ecc = opening_rows(sorted((c / f"{ECC}_s{seed}_urgent").glob("*.json")))
    sac = opening_rows(sorted((c / f"{REFERENCE}_s{seed}_urgent").glob("*.json")))
    pure = opening_rows(sorted((root / "S" / "pure_urgent").glob("*.json")))
    nocert = opening_rows(sorted((c / f"{ECC}_s{seed}_urgent_nocert").glob("*.json")))
    cases = select_cases(ecc, sac, pure, nocert)
    dump(out / "CASES.json", cases)
    opening = cases["physical"]["opening"]
    trace_only_smoke = False
    if smoke and opening is None:
        traces = sorted((root / "TRAJ").glob(f"traj_{ECC}_*.json"))
        if traces:
            opening = int(traces[0].stem.rsplit("_", 1)[1])
            trace_only_smoke = True
    trajectories = {m: case_rows(root, m, opening) for m in (ECC, REFERENCE, "pure")}
    fig = plt.figure(figsize=(5.5, 4))
    ax = fig.add_subplot(111, projection="3d")
    for i, (name, row) in enumerate(trajectories.items()):
        if not row:
            continue
        points = np.asarray([np.asarray(x["rot"]).T @ x["q"] for x in row["rows"]])
        ax.plot(*points.T, label=names[name], color=COLORS[i], ls=STYLE[i])
        if name == ECC:
            indexes = [j for j, x in enumerate(row["rows"]) if j == 0 or x["mode"] != row["rows"][j - 1]["mode"]]
            ax.scatter(*points[indexes].T, s=14, color=COLORS[i])
            axis = np.asarray(row["approach_axis"])
            ax.quiver(0, 0, 0, *axis, length=2, color="k")
    ax.set(xlabel="Target x (m)", ylabel="Target y (m)", zlabel="Target z (m)",
           title=f"Opening {opening}" + (" (rendering smoke)" if trace_only_smoke else ""))
    if ax.lines:
        ax.legend()
    trace_payload = {"selection": cases["physical"], "rendered_opening": opening,
                     "rendering_smoke_only": trace_only_smoke, "trajectories": trajectories}
    write_plot(out, "F2", fig, trace_payload, raster=True)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), layout="constrained")
    quantities = [
        ("range_m", "Range (m)"), ("port_los_deg", "Port angle (deg)"), ("speed_m_s", "Speed (m/s)"),
        ("attitude_error_rad", "Attitude error (rad)"), ("angular_velocity_error_rad_s", "Angular rate error (rad/s)"),
        ("union_margin_m", "Operation margin (m)"),
    ]
    for axis, (key, title) in zip(axes.flat, quantities):
        for i, (name, row) in enumerate(trajectories.items()):
            if row:
                xs = row["rows"]
                ys = [x.get(key, x.get("terminal", {}).get(key, x.get("margins", {}).get(key))) for x in xs]
                if any(y is not None for y in ys):
                    axis.plot([x["t"] for x in xs], ys, color=COLORS[i], ls=STYLE[i], label=names[name])
        if key.endswith("margin_m"):
            axis.axhline(0, color=".3", ls=":")
        axis.set(xlabel="Time (s)", ylabel=title)
    write_plot(out, "F3", fig, trace_payload)

    urgent = result["conditions"]["urgent"]
    economy = result["conditions"]["economy"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    methods = {m: v["pooled"] for m, v in urgent["learned"].items() if m in formal["trained"]}
    methods.update(urgent["fixed"])
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for i, event in enumerate(("success", "timeout", "resource_exhausted", "unsafe", "no_plan")):
        n = np.array([m["events"].get(event, 0) / max(1, m["n"]) for m in methods.values()])
        axes[0].bar(x, n, bottom=bottom, color=COLORS[i], label=event, hatch=("", "//", "..", "xx", "\\\\")[i])
        bottom += n
    axes[0].set_xticks(x, [f"{names[m]}\nn={v['n']}" for m, v in methods.items()], rotation=35, ha="right", fontsize=7)
    axes[0].legend(fontsize=6)
    axes[0].set_ylabel("Outcome fraction")
    comparisons = [(m, v["combined"]["return"]) for m, v in urgent["attribution_2x2"].items() if v and m != REFERENCE]
    for i, (m, v) in enumerate(comparisons):
        low, high = v["interval"]
        axes[1].plot([low, high], [i, i], color=COLORS[i])
        axes[1].scatter(v["mean"], i, color=COLORS[i])
    axes[1].set_yticks(range(len(comparisons)), [names[m] for m, _ in comparisons])
    axes[1].axvline(0, color=".3", ls=":")
    axes[1].set_xlabel("Urgent return difference vs SAC–MPC")
    eff = (economy["ecc_vs_sac"] or {}).get("combined", {})
    for i, (key, unit) in enumerate((("time", "s"), ("delta_v", "m/s"))):
        v = eff.get(key, {})
        if v.get("interval"):
            axes[2].text(.05, .7 - .3 * i, f"{key}: {v['mean']:.3g} {unit}\n95% CI {v['interval'][0]:.3g}, "
                         f"{v['interval'][1]:.3g}  n={v['n']}", transform=axes[2].transAxes)
    axes[2].set_axis_off()
    write_plot(out, "F4", fig, {"urgent": urgent, "economy": economy})

    selected = cases["mechanism"]["opening"]
    episode = ecc.get(selected)
    decision = next((d for d in episode.get("decisions", []) if d.get("cert", {}).get("intervened")), None) if episode else None
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    if decision:
        cert = decision["cert"]
        for point in cert["cands"]:
            axes[0].scatter(*point["b"], marker="o" if point["certified"] else "x", color=COLORS[0])
        axes[0].scatter(*cert["a"], marker="D", color=COLORS[1], label="Proposed")
        axes[0].scatter(*cert["a_exec"], marker="*", color=COLORS[2], label="Executed")
        axes[0].legend(fontsize=7)
        timeline = {"Decision": 0, "Period end": 16, "Predicted event": cert["alt_t"],
                    "Deadline": decision["remaining_s"]}
        if decision.get("true_mode_event_t") is not None:
            timeline["Truth event"] = decision["true_mode_event_t"] - decision["t_start"]
        for i, (label, time) in enumerate(timeline.items()):
            axes[1].scatter(time, i, color=COLORS[i])
        axes[1].set_yticks(range(len(timeline)), list(timeline), fontsize=7)
    else:
        show_missing(axes[0], "No saved intervention")
        show_missing(axes[1], "No saved intervention")
    axes[0].set(xlabel="Closing intent", ylabel="Sweep intent")
    axes[1].set_xlabel("Seconds from decision")
    strata = urgent["mechanism"]["strata"]
    if strata:
        axes[2].bar([f"{k}\nn={v['n']}" for k, v in strata.items()], [v["modification_rate"] for v in strata.values()])
    axes[2].set(ylabel="Modification fraction", xlabel="Opportunity margin η")
    write_plot(out, "F5", fig, {"selection": cases["mechanism"], "decision": decision, "mechanism": urgent["mechanism"]})

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), layout="constrained")
    attributions = urgent["attribution_2x2"]
    order = [REFERENCE, "SAC_CERT_LOOP", "EACH_C", ECC]
    for i, m in enumerate(order):
        if m in methods:
            axes[0, 0].bar(i, methods[m]["capture_rate"], color=COLORS[i])
    axes[0, 0].set_xticks(range(4), [f"{z}/{c}" for z, c in (("off", "off"), ("off", "on"), ("on", "off"), ("on", "on"))])
    axes[0, 0].set(xlabel="Feedback / certification", ylabel="Urgent capture fraction")
    switches = urgent["switches"]
    items = [(k, switches[k]) for k in ("ecc_cert", "each_cert", "sac_cert") if switches[k]]
    for i, (_, v) in enumerate(items):
        axes[0, 1].bar(i - .18, len(v["rescued"]), width=.35, color=COLORS[0])
        axes[0, 1].bar(i + .18, -len(v["lost"]), width=.35, color=COLORS[1])
    axes[0, 1].set_xticks(range(len(items)), [k for k, _ in items])
    axes[0, 1].set_ylabel("Rescued (+) / lost (−)")
    for axis, key, title in ((axes[1, 0], "rotation_guidance", "Rotation guidance"),
                             (axes[1, 1], "replanning", "Replanning")):
        v = switches.get(key)
        if v:
            axis.bar(["Rescued", "Lost"], [len(v["rescued"]), len(v["lost"])], color=COLORS[:2])
        else:
            show_missing(axis, "No paired outcomes")
        axis.set_title(title)
    write_plot(out, "F6", fig, {"attribution": attributions, "switches": switches, "absolute": methods})

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    resource = result["conditions"]["resource"]
    for i, source in enumerate((economy, resource)):
        for j, m in enumerate(order):
            if m in source["learned"]:
                v = source["learned"][m]["pooled"]
                axes[0].scatter(j, v["capture_rate"], marker=("o", "s")[i], color=COLORS[j])
                axes[0].scatter(j, v["events"].get("resource_exhausted", 0) / v["n"],
                                marker=("x", "+")[i], color=COLORS[j])
    axes[0].set_xticks(range(4), [names[m] for m in order], rotation=25, ha="right", fontsize=7)
    axes[0].set_ylabel("Capture / exhaustion fraction")
    spin = {}
    for name in (ECC, REFERENCE):
        spin[name] = []
        rows = [row for folder in sorted(c.glob(f"{name}_s*_urgent"))
                for row in opening_rows(sorted(folder.glob("*.json"))).values()]
        for low, high in ((2.1, 2.3), (2.3, 2.5), (2.5, 2.700001)):
            subset = [r for r in rows if r.get("initial") and
                      low <= np.degrees(np.linalg.norm(r["initial"]["omega"])) < high]
            spin[name].append({"n": len(subset), "capture_rate": sum(map(success, subset)) / len(subset) if subset else None})
        axes[1].plot([2.2, 2.4, 2.6], [x["capture_rate"] for x in spin[name]], marker="o", label=names[name])
    axes[1].set(xlabel="Initial spin (deg/s)", ylabel="Capture fraction")
    axes[1].legend(fontsize=7)
    write_plot(out, "F7", fig, {"economy": economy, "resource": resource, "spin_bins": spin})

    lam = np.linspace(0, pi, 150)
    phase = np.linspace(-pi, pi, 160)
    beta = pi / 3
    cosine = np.cos(lam[:, None]) * np.cos(beta) + np.sin(lam[:, None]) * np.sin(beta) * np.cos(phase)
    fig, axis = plt.subplots(figsize=(5, 3))
    axis.contour(np.degrees(phase), np.degrees(lam - beta), cosine, levels=[np.cos(np.radians(35))], colors=[COLORS[0]])
    axis.set(xlabel="Phase φ (deg)", ylabel="Latitude error λ−β (deg)")
    geoscan = list((root / "G").glob("geoscan_*.json"))
    geo = read_json(sorted(geoscan)[0]) if geoscan else None
    write_plot(out, "RWG_GEOMETRY", fig, {"beta_rad": beta, "lambda_rad": lam.tolist(),
               "phase_rad": phase.tolist(), "cosine": cosine.tolist(), "geoscan": geo})

    write_table(out, "T1", ["Method", "Actor", "Critic", "Certification", "Geometry", "Steps"],
                [[names[m], trained[m]["actor_view"], trained[m]["critic_view"], trained[m].get("certify", False),
                  trained[m]["geometry"], formal["steps"]] for m in formal["trained"]])
    write_table(out, "T2", ["Condition", "Method", "N", "Captures", "Unsafe", "Mean true return"],
                [[condition, names.get(m, m), v["n"], v["captures"], v["events"].get("unsafe", 0), v["return"].get("mean")]
                 for condition, block in result["conditions"].items()
                 for m, v in ({k: x["pooled"] for k, x in block["learned"].items()} | block["fixed"]).items()])
    negative = Path(__file__).resolve().parents[1] / "docs" / "DEAD_ENDS.md"
    lines = [line.strip() for line in negative.read_text(encoding="utf-8").splitlines()
             if line.startswith("|") and any(f"D{i}" in line for i in range(39, 46))]
    write_table(out, "T3", ["Registered development negative results"], [[line] for line in lines])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--readout", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if not a.smoke and not check_formal(a.root, stage_of("formal"))["complete"]:
        raise SystemExit("Formal figures require complete registered evidence")
    a.output.mkdir(parents=True, exist_ok=True)
    main_figures(a.root, a.output, read_json(a.readout), a.smoke)


if __name__ == "__main__":
    main()
