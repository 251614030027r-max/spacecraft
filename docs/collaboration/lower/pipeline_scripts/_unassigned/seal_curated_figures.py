from pathlib import Path
import hashlib,json,shutil
top=Path(__file__).resolve().parents[1];root=top/'reports'/'ppt_20261009'
for p in (top/'脚本'/'curate_demonstration.py',root/'curate_demonstration.py',root/'TRAINING_AND_FIGURE_REVIEW.md'):
    s=p.read_text(encoding='utf-8').replace('237个交接回合里140个完成','237个交接回合里137个完成');p.write_text(s,encoding='utf-8')
shutil.copy2(top/'脚本'/'audit_training_interpretation.py',root/'audit_training_interpretation.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'}
(root/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(root/name)==digest for name,digest in files.items())
assert len(list(root.glob('fig_*.png')))==3
print('CURATED_DELIVERY_VERIFIED',len(files))
