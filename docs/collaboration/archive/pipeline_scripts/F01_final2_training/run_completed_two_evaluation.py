"""User-authorized early evaluation of 262460/262462; defer 262461, no readout."""
import json, os, zipfile
import run_formal_after_training as f

def run():
    f.SEEDS=(262460,262462)
    f.STATE=f.REC/'FORMAL_TWO_EXECUTION.json'
    f.S={'phase':'preflight','workers':{},'training_untouched':True,
         'science_commit':f.COMMIT,'evaluated_models':list(f.SEEDS),
         'deferred_model':262461,'official_readout_authorized_here':False,
         'authorization':'2026-10-09 user: immediately evaluate completed two seeds; defer 262461',
         'boundary':'Partial preregistered data only. Cannot omit the deferred seed from official verdict.'}
    assert not f.STATE.exists(), 'Existing two-seed evaluation state; do not duplicate'
    assert not (f.REC/'FORMAL_EXECUTION.json').exists(), 'Other formal supervisor state exists'
    assert not f.E.exists(), 'Existing evidence directory; inspect rather than overwrite'
    f.frozen()
    for seed in f.SEEDS:
        m=f.read(f.MAIN/f'logs/final2_{seed}/manifest.json')
        assert m['status']=='completed' and m['actual_outer_decisions']==60000
        assert not f.alive(f.TRAIN_PIDS[seed]), 'Completed training process still active'
        model=f.MAIN/f'logs/final2_{seed}/final_model.zip'
        with zipfile.ZipFile(model) as z:assert z.testzip() is None
        f.S.setdefault('final_model_sha256',{})[str(seed)]=f.sha(model)
    f.phase('two_seed_evaluation_blinded')
    try:
        block='271000-271047,272000-272047'
        expected=list(range(271000,271048))+list(range(272000,272048))
        jobs=[]
        # Start completed model rows immediately; common baselines occupy remaining slots.
        for seed in f.SEEDS:
            for row in ('stopping','learned'):
                dest=f.E/row/str(seed)
                jobs.append((f'{row}_{seed}','experiments.v3_stopping',
                    ['evaluate','--run-dir',f'logs/final2_{seed}','--row',row,
                     '--seeds',block,'--output-dir',dest],dest,expected))
        for row in ('pure','nominal'):
            dest=f.E/row
            args=['evaluate','--row',row,'--seeds',block,'--output-dir',dest]
            if row=='pure':module='experiments.v3_stopping';args+=['--run-dir','logs/final2_262460']
            else:module='experiments.regime_screen';args+=['--regime','w2.36_r15']
            jobs.append((row,module,args,dest,expected))
        f.batch(jobs)
        f.phase('two_seed_evaluation_completed_await_deferred_seed_decision')
    except Exception as err:
        f.S['error']=repr(err);f.phase('fault_preserve_evidence');raise

if __name__=='__main__':run()
