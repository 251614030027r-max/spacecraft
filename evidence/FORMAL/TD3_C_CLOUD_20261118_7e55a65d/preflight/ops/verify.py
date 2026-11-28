"""TD3-only delivery checks. Never claims the full multi-method matrix is complete."""
import csv
import hashlib
import json
from pathlib import Path

SHA = '7e55a65d373666bdb901357e07f5c53713647a4c'


def hashes(root, target):
    root, target = Path(root), Path(target)
    lines = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p != target and p.suffix != '.tmp':
            h = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
            lines.append(f'{h}  {p.relative_to(root).as_posix()}\n')
    target.write_text(''.join(lines))
    return len(lines)


def verify_train(root, seed):
    from experiments.final import verify_manifest
    run = Path(root) / 'B' / f'TRAIN_TD3_C_s{seed}'
    m = json.loads((run / 'manifest.json').read_text())
    verify_manifest(m, 'TD3_C')
    for key, want in {'git_head': SHA, 'steps': 5000, 'seed': seed, 'workers': 2,
                      'final_config': 'TD3_C', 'final_stage': 'formal', 'smoke': False}.items():
        assert m[key] == want, (key, m.get(key), want)
    finished = json.loads((run / 'finished.json').read_text())
    assert finished['steps'] == 5000
    assert len(finished['worker_order']) == 5000
    assert set(finished['worker_order']) == {0, 1}
    assert finished['replay']['transitions'] == 5000
    episodes = json.loads((run / 'episodes.json').read_text())
    live = [json.loads(x) for x in (run / 'episodes.live.jsonl').read_text().splitlines()]
    assert episodes == live, 'native and rolling episode records differ'
    progress = [json.loads(x) for x in (run / 'progress.jsonl').read_text().splitlines()]
    assert [x['steps'] for x in progress] == list(range(1, 5001))
    models = [run / 'final.zip', *(run / 'checkpoints' / f'step_{k}.zip' for k in [0,1000,2000,3000,4000,5000])]
    assert all(p.is_file() and p.stat().st_size > 0 for p in models)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(run / 'tensorboard'), size_guidance={'scalars': 0}).Reload()
    assert accumulator.Scalars('progress/decisions')[-1].step == 5000
    count = hashes(run, run / 'SHA256SUMS')
    result = {'seed': seed, 'status': 'verified_train', 'steps': 5000, 'episodes': len(episodes),
              'git_head': SHA, 'workers': 2, 'hashed_files': count, 'finished': finished}
    # worker_order is preserved in finished.json, not repeated in the compact report.
    result['finished'] = {k:v for k,v in finished.items() if k != 'worker_order'}
    (Path(root) / f'VERIFIED_TRAIN_s{seed}.json').write_text(json.dumps(result, indent=2))
    return result


def verify_evaluations(root, seed):
    root = Path(root)
    reports, rows = [], []
    specs = [('urgent', None, root / 'C' / f'TD3_C_s{seed}_urgent', set(range(300000,300096)))]
    specs += [('validation', k, root / 'V' / f'TD3_C_s{seed}_step{k}', set(range(295200,295224)))
              for k in (1000,2000,3000,4000,5000)]
    for condition, step, directory, expected in specs:
        results = [json.loads(p.read_text()) for p in directory.glob('*.json')]
        assert not any(r.get('event') == 'error' for r in results), str(directory)
        episodes = [r for r in results if 'opening' in r]
        openings = [r['opening'] for r in episodes]
        assert len(openings) == len(expected) and set(openings) == expected, (directory, len(openings))
        for r in episodes:
            assert r.get('condition') == condition, (directory, r.get('condition'))
            assert float(r['mission']['budget_m_s']) == 3.0, r['mission']
            assert 'terminal' in r and 'min_margins' in r
            rows.append({'config':'TD3_C','seed':seed,'condition':condition,'checkpoint_step':step or 5000,
                         'opening':r['opening'],'event':r['event'],'return_true':r['return'],
                         'time_s':r['time_s'],'delta_v':r['delta_v'],'source':str(directory.relative_to(root))})
        reports.append({'condition':condition,'checkpoint_step':step or 5000,'episodes':len(episodes),
                        'numerical_failure':sum(r['event']=='numerical_failure' for r in episodes)})
    with (root / f'TD3_C_s{seed}_EPISODES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'seed':seed,'status':'verified_authorized_TD3_scope','science_sha':SHA,'evaluations':reports,
            'full_formal_matrix_complete':False}
    (root/f'VERIFIED_s{seed}.json').write_text(json.dumps(result,indent=2))
    return result
