"""Read-only Python 3.12 monitoring; no repository files or scientific objects are changed."""
import os
import sys

if os.environ.get('TD3_CLOUD_OBSERVE') == '1' and 'experiments.final' in sys.orig_argv and 'train' in sys.orig_argv:
    import atexit
    import json
    import time
    from pathlib import Path

    TOOL = 5
    TARGET = '/workspace/spacecraft-td3/train/t2_mve_sac.py'
    state = {'step': 0, 'episodes': 0, 'writer': None}
    sys.monitoring.use_tool_id(TOOL, 'td3-read-only-telemetry')

    def close_writer():
        if state['writer'] is not None:
            state['writer'].flush()
            state['writer'].close()

    atexit.register(close_writer)

    def observe(code, line):
        # The loop reaches this line after accepting the transition, appending any
        # completed episode, and completing the prescribed learner updates.
        if line != state['line']:
            return
        frame = sys._getframe(1)
        if frame.f_code is not code:
            raise RuntimeError('monitor frame mismatch')
        loc = frame.f_locals
        model = loc['model']
        step = int(model.num_timesteps)
        if step <= state['step']:
            return
        out = loc['out']
        if state['writer'] is None:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            state.update(writer=EventFileWriter(str(out / 'tensorboard'), flush_secs=10), Event=Event, Summary=Summary)
        wall = time.time() - loc['t0']
        row = {'time_utc_unix': time.time(), 'pid': os.getpid(), 'steps': step,
               'budget_steps': int(loc['steps']), 'updates': int(model._n_updates),
               'episodes': len(loc['episodes']), 'wall_s': wall,
               'decisions_per_second': step / wall if wall > 0 else None,
               'worker': int(loc['w_']), 'source': 'read_only_sys.monitoring_train_locals'}
        with (out / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        tmp = out / 'progress.tmp'
        tmp.write_text(json.dumps(row, indent=2))
        tmp.replace(out / 'progress.json')
        new_episodes = loc['episodes'][state['episodes']:]
        for episode in new_episodes:
            with (out / 'episodes.live.jsonl').open('a') as f:
                f.write(json.dumps(episode, default=str) + '\n')
        values = {'progress/decisions': step, 'progress/updates': row['updates'],
                  'progress/episodes': row['episodes'], 'progress/decisions_per_second': row['decisions_per_second']}
        if model.model_stats:
            values.update({'learner/' + k: v for k, v in model.model_stats[-1].items()
                           if k not in ('steps', 'update') and isinstance(v, (int, float))})
        if new_episodes:
            values.update({'episode/' + k: v for k, v in new_episodes[-1].items()
                           if k in ('return_true', 'return_shaped', 'time_s', 'delta_v') and isinstance(v, (int, float))})
        if step % 10 == 0 or new_episodes or step == 1:
            Summary = state['Summary']
            summary = Summary(value=[Summary.Value(tag=k, simple_value=float(v)) for k, v in values.items() if v is not None])
            state['writer'].add_event(state['Event'](wall_time=time.time(), step=step, summary=summary))
            state['writer'].flush()
        if step % 20 == 0 or new_episodes or step == 1:
            print(json.dumps({'telemetry': row}), flush=True)
        state.update(step=step, episodes=len(loc['episodes']))

    def entered(code, offset):
        if code.co_filename == TARGET and code.co_name == 'train':
            lines = Path(TARGET).read_text().splitlines()
            state['line'] = next(i for i, line in enumerate(lines, 1)
                                 if line.strip() == 'while marks and model.num_timesteps >= marks[0]:')
            sys.monitoring.set_events(TOOL, 0)
            sys.monitoring.set_local_events(TOOL, code, sys.monitoring.events.LINE)

    sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, entered)
    sys.monitoring.register_callback(TOOL, sys.monitoring.events.LINE, observe)
    sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)
