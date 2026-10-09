"""Synthetic end-to-end package/publication QA; never reads or moves live assets."""
import importlib.util, json, tempfile
from pathlib import Path

path=Path(__file__).with_name('B2助手.py')
spec=importlib.util.spec_from_file_location('b2qa',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
real_records=m.REC
with tempfile.TemporaryDirectory(dir=real_records,prefix='delivery_qa_') as tmp:
    root=Path(tmp).resolve(); m.ROOT=root; m.TOPIC=root/'过程文件/阶段B2'; m.REC=m.TOPIC/'记录'
    m.MAIN=root/'fake_engineering'; m.OUT=m.REC/'readout_b1b2.json'; m.BUILD=m.TOPIC/'交付展开'
    m.PACKAGE=root/'上层交付/待发布/STAGE_B2_LITE_262000_BLOCK.zip'; m.ANCHOR=m.REC/'B1_FIXED_DELIVERY_ANCHOR.json'
    latest=root/'上层交付/最新'; latest.mkdir(parents=True)
    m.REC.mkdir(parents=True); (m.TOPIC/'脚本').mkdir()
    fixture={'verdict':'PROCEED','fidelity_problems':[], 'b2':{'valid':True,'fidelity_problems':[]}}
    for model in m.MODELS: fixture['b2'][model]={'summary':{'successes':38,'with_destroying_handoff':2}}
    m.write(m.OUT,fixture)
    receipt=m.REC/'B1_FROZEN_RECEIPT.json'; b1=m.REC/'B1_FROZEN_READOUT.json'
    m.write(receipt,{'zip_sha256':'fixture-b1-fixed-sha'}); m.write(b1,{'verdict':'PROCEED'})
    m.write(latest/'STAGE_B1_DELIVERY_RECEIPT_20261001.json',m.read(receipt))
    (latest/'old-scientific-evidence.zip').write_bytes(b'fixture historical evidence')
    (latest/'REPORT.md').write_text('fixture old report')
    m.write(m.ANCHOR,{'receipt':str(receipt),'readout':str(b1),'files':[{'path':str(p),'sha256':m.sha(p)} for p in (receipt,b1)]})
    for model in m.MODELS:
        for phase in ('stage_b','stage_b2'):
            for seed in m.SEEDS: m.write(m.MAIN/f'eval/v3e/{phase}/{model}/seed_{seed}.json',{'seed':seed,'fixture':True})
        for name in ('learned_only','m2_a','m2_b','m2_c','m2_d'): m.write(m.MAIN/f'eval/v3e/{model}/{name}.json',{})
        m.write(m.MAIN/f'logs/v3e_{model}/manifest.json',{})
    m.write(m.MAIN/'eval/v3e/pure_mpc.json',{})
    for name in ('docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md','docs/STAGE_B_RUN_ORDER_LOWER_20260930.md','experiments/v3_handoff_readout.py'):
        p=m.MAIN/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fixture frozen code')
    m.status=lambda:{'state':'READY'}; m.check_versions=lambda:None; m.official=lambda:fixture
    m.finalize(); m.verify_package()
    before=m.sha(latest/'old-scientific-evidence.zip')
    m.publish()
    archived=root/'上层交付/历史/STAGE_B1_262000_BLOCK_20261001/old-scientific-evidence.zip'
    assert m.sha(archived)==before
    assert (latest/'REPORT.md').read_text(encoding='utf-8').startswith('# Stage B2')
    assert not m.PACKAGE.exists() and (latest/m.PACKAGE.name).exists()
    m.verify_package(); m.publish(); m.finalize()
    assert len(list((root/'上层交付/历史').glob('STAGE_B1_*')))==1
print('END_TO_END_QA_OK: 独立打包、CRC/全部SHA、归档哈希、发布及重复执行保护通过；仅合成数据。')
