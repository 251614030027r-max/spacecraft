from pathlib import Path
import json,shutil,time
import runtime as a
R=a.R;src=Path(a.read(R/'记录/MIDTERM_BUILD.json')['snapshot']);latest=a.SPACE/'上层交付/最新'
hist=a.SPACE/'上层交付/历史'/('T26W1_before_T27_midterm_'+time.strftime('%Y%m%d_%H%M%S'))
assert src.resolve().is_relative_to(R.resolve()) and latest.resolve().is_relative_to(a.SPACE.resolve()) and hist.resolve().is_relative_to(a.SPACE.resolve()) and not hist.exists()
for line in (src/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
 h,p=line.split('  ',1);assert a.sha(src/p)==h
if latest.exists() and any(latest.iterdir()):
 before={p.relative_to(latest).as_posix():a.sha(p) for p in latest.rglob('*') if p.is_file()};shutil.move(str(latest),str(hist));assert before=={p.relative_to(hist).as_posix():a.sha(p) for p in hist.rglob('*') if p.is_file()}
latest.mkdir(parents=True,exist_ok=True);shutil.copytree(src,latest,dirs_exist_ok=True)
(latest/'README.md').write_text('# T2.7 中期交付｜2026-11-01｜训练进行中\n\n当前报告：[RESULT_MIDTERM.md](RESULT_MIDTERM.md)。配套原始快照、MIDTERM_DATA.json、历史必要对照、MIDTERM_CURVES.png与SHA256SUMS随附。尚无本轮部署评估或正式判定；不把历史结果当成本轮结果。\n\n上一轮已关闭T2.6W1交付保全归档：`'+str(hist)+'`。当前训练不受交付整理影响。\n',encoding='utf-8')
handoff=a.COLLAB/'docs/collaboration/lower/handoffs/t2_7_guide_20261031';dest=handoff/'midterm_20261101'
assert not dest.exists();shutil.copytree(src,dest)
(handoff/'README.md').write_text('# T2.7 本机交付入口\n\n- 2026-11-01：当前为[中期报告](midterm_20261101/RESULT_MIDTERM.md)，原始快照和哈希随附；训练进行中，未作部署判定。\n- 千步机制早报见RESULT_1000.md。\n- RESULT_FINAL.md尚未生成，最终以原生t27判定和四种配对完整结果为准。\n',encoding='utf-8')
a.git('fetch','origin','collab/spacecraft',repo=a.COLLAB);a.git('merge','--ff-only','origin/collab/spacecraft',repo=a.COLLAB)
assert not a.git('diff','--cached','--name-only',repo=a.COLLAB)
a.git('add','--',str(dest.relative_to(a.COLLAB)),str((handoff/'README.md').relative_to(a.COLLAB)),repo=a.COLLAB)
a.git('commit','-m','evidence: T2.7 midterm training snapshot and bounded historical comparison',repo=a.COLLAB);report_commit=a.git('rev-parse','HEAD',repo=a.COLLAB)
shutil.copy2(a.CURRENT,R/'记录/CURRENT_BEFORE_MIDTERM.json');cur=a.read(a.CURRENT)
cur.update(updated_at=a.now(),collab_delivery_commit=report_commit,local_report=str(latest/'RESULT_MIDTERM.md'),current_delivery_report=str((dest/'RESULT_MIDTERM.md').relative_to(a.COLLAB)).replace('\\','/'),current_delivery_phase='midterm_training_in_progress',midterm_snapshot=str(src),midterm_report_SHA256=a.sha(src/'RESULT_MIDTERM.md'),previous_delivery_history=str(hist));a.dump(a.CURRENT,cur)
a.git('add','--','docs/collaboration/CURRENT.json',repo=a.COLLAB);a.git('commit','-m','state: T2.7 midterm delivery pointer; training continues',repo=a.COLLAB);head=a.git('rev-parse','HEAD',repo=a.COLLAB)
a.git('push','origin','HEAD:refs/heads/collab/spacecraft',repo=a.COLLAB);remote=a.git('ls-remote','origin','refs/heads/collab/spacecraft',repo=a.COLLAB).split()[0];assert remote==head
for line in (src/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
 h,p=line.split('  ',1);assert a.sha(latest/p)==h and a.sha(dest/p)==h
a.dump(R/'记录/MIDTERM_PUSH_RECEIPT.json',{'jst':a.now(),'status':'pushed_verified','report_commit':report_commit,'current_state_commit':head,'local_delivery':str(latest),'report_SHA256':a.sha(src/'RESULT_MIDTERM.md'),'preserved_prior_delivery':str(hist)})
shutil.copy2(R/'记录/MIDTERM_PUSH_RECEIPT.json',latest/'MIDTERM_PUSH_RECEIPT.json')
print(json.dumps(a.read(R/'记录/MIDTERM_PUSH_RECEIPT.json'),ensure_ascii=True))
