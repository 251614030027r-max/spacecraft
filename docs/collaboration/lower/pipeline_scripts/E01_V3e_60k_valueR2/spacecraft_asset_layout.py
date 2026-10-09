"""Workspace asset locations and publication; independent of scientific evaluation."""
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
PUBLIC = BASE / '上层交付'
PROCESS = BASE / '过程文件' / 'V3e60k'
SCRIPTS = PROCESS / '脚本'
RECORDS = PROCESS / '记录'
STATE = BASE / 'V3E_60K_EXECUTION_20260929.json'
if not STATE.exists():
    STATE = PROCESS / '运行归档' / STATE.name

def guarded_move(source, destination):
    source, destination = Path(source), Path(destination)
    root = BASE.resolve()
    for path in (source, destination):
        absolute = path.resolve()
        if absolute == root or not absolute.is_relative_to(root):
            raise ValueError(f'Move outside workspace: {absolute}')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))

def public_bundle(name):
    for path in (PUBLIC / '最新' / name, PUBLIC / '历史' / name,
                 PUBLIC / '历史' / name / name):
        if path.is_dir():
            return path
    raise FileNotFoundError(name)

def publish(bundle, archive, receipt):
    """Publish only an already verified bundle; preserve the prior latest release."""
    latest = PUBLIC / '最新'
    latest.mkdir(parents=True, exist_ok=True)
    old_items = list(latest.iterdir())
    old_zips = [p for p in old_items if p.suffix == '.zip']
    if old_items:
        tag = old_zips[0].stem if len(old_zips) == 1 else datetime.now().strftime('旧交付_%Y%m%d_%H%M%S')
        history = PUBLIC / '历史' / tag
        if history.exists():
            history = history.with_name(tag + '_' + datetime.now().strftime('%H%M%S'))
        history.mkdir(parents=True)
        for path in old_items:
            guarded_move(path, history / path.name)
    for path in (Path(bundle), Path(archive), Path(receipt)):
        guarded_move(path, latest / path.name)
    final_bundle, final_archive, final_receipt = (latest / Path(p).name for p in (bundle, archive, receipt))
    (latest / 'README.md').write_text(
        '# 当前唯一最新交付\n\n交给上层只需发送本目录中的一个ZIP：\n\n'
        f'[{final_archive.name}]({final_archive.as_posix()})\n\n'
        f'[阅读报告]({(final_bundle / "REPORT.md").as_posix()})。同名目录是便于浏览的证据副本，回执记录核验结果。\n', encoding='utf-8')
    if 'R1_SUPPLEMENT' in final_bundle.name:
        with (latest / 'README.md').open('a', encoding='utf-8') as handle:
            handle.write('\n本包仅为第一轮价值补充结果，第二轮价值主评估仍在运行。\n')
    digest = hashlib.sha256(final_archive.read_bytes()).hexdigest()
    (latest / 'ZIP_SHA256.txt').write_text(f'{digest}  {final_archive.name}\n', encoding='utf-8')
    return final_bundle, final_archive, final_receipt
