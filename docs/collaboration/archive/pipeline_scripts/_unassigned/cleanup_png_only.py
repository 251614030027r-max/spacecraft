from pathlib import Path
import hashlib, json, re

BASE = Path(r'C:\Users\35884\Documents\Spacecraft').resolve()
TOP = BASE / '过程文件/阶段基准图_20261009'
REPO = BASE / '过程文件/协作/Git工作树'
LOWER = REPO / 'docs/collaboration/lower/handoffs'
roots = [TOP / 'reports/figures_interim_20261009',
         BASE / '过程文件/汇报图表_20261009/reports',
         LOWER / 'figures_interim_20261009', LOWER / 'ppt_temporary_20261009']
removed = []
for root in roots:
    root = root.resolve()
    assert root.is_relative_to(BASE) and root != BASE
    if not root.exists():
        continue
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.pdf', '.svg'}:
            assert p.resolve().is_relative_to(root)
            removed.append({'path': str(p), 'bytes': p.stat().st_size,
                            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
            p.unlink()
    for p in root.rglob('*.md'):
        if p.name == 'RUN_ORDER.md':
            continue  # Preserve the archived upper instruction verbatim.
        s = p.read_text(encoding='utf-8-sig')
        t = re.sub(r'\s*·\s*\[(?:PDF|SVG)\]\([^\n)]*\)', '', s)
        for a in ['PNG/PDF/SVG', 'PNG、PDF、SVG', 'PNG + PDF + SVG']:
            t = t.replace(a, 'PNG')
        if t != s:
            p.write_text(t, encoding='utf-8')

# Future plotting uses only PNG; do not execute plotting or simulations here.
for p in [TOP / '脚本/plot_interim.py', *roots[0].rglob('plot_interim.py'),
          *roots[2].rglob('plot_interim.py')]:
    s = p.read_text(encoding='utf-8-sig')
    s = s.replace("('png','pdf','svg')", "('png',)")
    p.write_text(s, encoding='utf-8')
for p in [TOP / '脚本/finish_delivery.py', *roots[0].rglob('finish_delivery.py'),
          *roots[2].rglob('finish_delivery.py')]:
    s = p.read_text(encoding='utf-8-sig')
    s = s.replace("assert (OUT/(name+'.pdf')).is_file() and (OUT/(name+'.svg')).is_file()",
                  "assert (OUT/(name+'.png')).is_file()")
    s = s.replace(' · [PDF]({name}.pdf) · [SVG]({name}.svg)', '')
    p.write_text(s, encoding='utf-8')

note = '\n\n2026-10-09格式更新：按用户最新指令仅保留高清PNG，移除PDF/SVG；科学数据、脚本及审核证据保留。归档执行单中的矢量格式要求由本次用户指令取代。\n'
for p in [roots[0] / 'README.md', roots[2] / 'README.md', roots[3] / 'README.md',
          LOWER / 'figures_interim_20261009.md', LOWER / 'ppt_temporary_20261009.md']:
    if p.exists():
        s = p.read_text(encoding='utf-8-sig').replace('PNG/PDF/SVG', 'PNG')
        p.write_text(s.rstrip() + note, encoding='utf-8')

verified = {}
for root in roots:
    if not root.exists():
        continue
    for manifest in root.rglob('FILES_SHA256.json'):
        folder = manifest.parent
        files = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(folder.rglob('*')) if p.is_file() and p != manifest}
        manifest.write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding='utf-8')
        assert all(hashlib.sha256((folder / n).read_bytes()).hexdigest() == h
                   for n, h in files.items())
        verified[str(folder)] = len(files)
    assert not any(p.is_file() and p.suffix.lower() in {'.pdf', '.svg'} for p in root.rglob('*'))
receipt = {'removed_files': removed, 'removed_count': len(removed),
           'freed_bytes': sum(x['bytes'] for x in removed), 'verified_manifests': verified,
           'boundary': 'Figure packages only; no training, scientific data, or Git history changed.'}
(TOP / '记录/PNG_ONLY_CLEANUP.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k != 'removed_files'}, ensure_ascii=False))
