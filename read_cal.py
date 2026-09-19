import json, glob, os
rows=[]
for f in sorted(glob.glob("eval/cal2/*.json")):
    d=json.load(open(f, encoding="utf-8")); recs=d["records"]
    comp=[r for r in recs if r["completed"]]
    def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
    rows.append((os.path.basename(f), d.get("tumble_scale"),
        f'{len(comp)}/{len(recs)}',
        round(mean([r["equivalent_delta_v_m_s"] for r in comp]),3),
        round(mean([r["survival_s"] for r in comp]),1),
        round(min([r["minimum_truth_normalized_margin"] for r in recs if r["minimum_truth_normalized_margin"] is not None]),4)))
print(f'{"file":22}{"tumble":8}{"complete":10}{"Δv":8}{"time":8}{"worstMargin"}')
for r in rows: print(f'{r[0]:22}{str(r[1]):8}{r[2]:10}{str(r[3]):8}{str(r[4]):8}{r[5]}')