from pathlib import Path
import hashlib,json
from PIL import Image

root=Path(__file__).resolve().parents[1]/'reports'/'ppt_20261009'
report=root/'FIGURE_REPORT.md'
text=report.read_text(encoding='utf-8')
append='\n\n## 科学支撑审核\n\n完整判断见[FIGURE_AUDIT.md](FIGURE_AUDIT.md)。推荐汇报顺序：三种子训练曲线（展示分化，不宣称收敛）、fig_structural_evidence（三模型交接结构）、fig_handoff_value（实际阈值触发）、fig_trajectory_3d与fig_constraint_response（固定案例闭环与约束）。补充图全部复用已有官方结构检查及已核验重放，不新增仿真。此案例协调方案比Pure MPC耗时增加34.5%、Delta-v增加36.0%，不能用于宣称性能优势。\n'
if '## 科学支撑审核' not in text: report.write_text(text+append,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'}
(root/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(sha(root/rel)==digest for rel,digest in files.items())
for p in sorted(root.glob('fig_*.png')):
    with Image.open(p) as im:
        assert im.info.get('dpi',(0,0))[0]>=300
        assert (root/(p.stem+'.pdf')).is_file() and (root/(p.stem+'.svg')).is_file()
        print(p.name,im.size,im.info['dpi'])
print('ALL_DELIVERY_HASHES_VERIFIED',len(files))
