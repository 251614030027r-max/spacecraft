"""Audit logs must store the terminal state, expose feedback, and stop numerical failures."""
import io,json
from types import SimpleNamespace
import numpy as np
import torch
from train.train_t1_b import Record

def callback(tmp_path,event=None,n=1):
    b=Record(tmp_path,[]);b.num_timesteps=n;b.handle=io.StringIO()
    b.model=SimpleNamespace(_last_obs=np.zeros((1,46),np.float32),policy=torch.nn.Linear(1,1),
        _n_updates=4,actor_updates=1,replay_buffer=SimpleNamespace(
            feedback_stats={'closed':4,'valid':3},interval_counts={'closed_by_review':4,'closed_by_terminal':0,'closed_by_truncation':0}),
        delta_calibration={'lambda':.2},delta_stats={'applied_share':[.05]})
    b.locals={'infos':[{'event':event,'terminal_observation':np.full(46,3.,np.float32)}],
        'actions':np.zeros((1,1),np.float32),'rewards':np.zeros(1),'dones':np.array([bool(event)]),
        'new_obs':np.full((1,46),99.,np.float32)}
    return b

def test_terminal_record_contains_terminal_not_auto_reset_state(tmp_path):
    b=callback(tmp_path,'success');assert b._on_step()
    line=json.loads(b.handle.getvalue())
    assert line['observation_before']==[0.]*46 and line['observation_after']==[3.]*46

def test_numerical_failure_stops_and_preserves_explicit_reason(tmp_path):
    b=callback(tmp_path,'numerical_failure');assert not b._on_step()
    assert json.loads((tmp_path/'ABORTED.json').read_text())['reason']=='numerical_failure'

def test_live_heartbeat_reports_actual_feedback_and_update_counts(tmp_path):
    b=callback(tmp_path,None,20);assert b._on_step()
    h=json.loads((tmp_path/'heartbeat.json').read_text())
    assert h['updates']==4 and h['actor_updates']==1 and h['finite']
    assert h['valid_model_intervals']==3 and h['real_review_intervals_closed']==4
    assert h['auxiliary_lambda_frozen']==.2 and h['last_auxiliary_gradient_share']==.05
