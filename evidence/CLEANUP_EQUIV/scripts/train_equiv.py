"""python -B train_equiv.py ROOT CONFIG OUTDIR: 40-decision single-worker training as the formal stage would run it."""
import inspect, json, os, sys
from pathlib import Path
def main():
    root = sys.argv[1]; sys.path.insert(0, root); os.chdir(root)
    from train.t2_mve_sac import train
    global CFG
    CFG = {'ECC_CERT_LOOP': ('e_priv', True, True), 'SAC_C': ('e_sac', True, False), 'EACH_C': ('e_priv', True, False),
           'SAC_CERT_LOOP': ('e_sac', True, True), 'ECC_CERT_LOOP_NOGEO': ('e_priv', False, True), 'TD3_C': ('e_td3', True, False)}
    arm, geometry, certify = CFG[sys.argv[2]]
    kw = dict(updates_per_transition=2.0, window=True, geometry=geometry, certify=certify,
              manifest_extra={'final_config': sys.argv[2], 'final_stage': 'formal'})
    params = inspect.signature(train).parameters
    if 'mission' in params:
        kw['mission'] = 'urgent_mix:0.75'
    else:
        kw.update(urgent_deadline='urgent_mix:0.75', obs_profile='cut_priv')
    train(Path(sys.argv[3]), arm, 10, 40, 1, [20], True, **kw)


if __name__ == "__main__":
    main()
