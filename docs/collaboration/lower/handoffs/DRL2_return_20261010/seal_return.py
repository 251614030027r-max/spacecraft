from pathlib import Path
import json,shutil,hashlib,subprocess
B=Path(r'C:\Users\35884\Documents\Spacecraft');T=B/'过程文件/整理_20261010';R=T/'记录/RETURN_DRL2'
C=B/'过程文件/协作/Git工作树/docs/collaboration';D=C/'lower/handoffs/DRL2_return_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.write_text(s,encoding='utf-8')
test=json.loads((R/'TEST_VERIFICATION.json').read_text(encoding='utf-8'))
assert test['retry_exit_code']==0 and '13 passed' in (R/'pytest_setup_retry.log').read_text(encoding='utf-8')
test.update(final_total_passed=354,final_xfailed=3,final_failed=0,final_errors=0,validation='Full suite: 341 passed + 3 xfailed, 13 setup errors due missing basetemp parent; after fixing test directory only the 13 were rerun and passed')
write(R/'TEST_VERIFICATION.json',json.dumps(test,ensure_ascii=False,indent=2))
before=json.loads((R/'PREPARED.json').read_text(encoding='utf-8'));main=Path(r'D:\py\DRL2')
assert all(sha(main/p)==h for p,h in before['environment_sha256'].items())
assert all(sha(main/f'logs/final2_{s}/final_model.zip')==h for s,h in before['model_sha256'].items())
assert not subprocess.check_output(['git','-C',str(main),'status','--porcelain','--untracked-files=all'])
for n in ('TEST_VERIFICATION.json','pytest_setup_retry.log'):shutil.copy2(R/n,D/n)
shutil.copy2(T/'脚本/seal_return.py',D/'seal_return.py')
report='''# 最新工程已回到DRL2：整理完成与边界

用户授权直接整理本地DRL2及Spacecraft，并从Git找回已有代码/文档/证据位置，不再补剩余评估。最新科学工程为D:/py/DRL2，科学工作分支claude/sac-mpc-coupling-design-ns7g6i，4795910102cf9fbf1dfec74b77e9449623a7b51e，Git工作树干净。临时D:/py/spacecraft不替代这个工程入口，原.venv、PyCharm配置及三个final2模型SHA均未改变。

## 实际执行

- 归档前4608个logs/eval/local_artifacts文件、3,999,974,773字节复制到D:/spacecraft_archive/return_drl2_20261010/pre_clean_source并逐文件SHA核验，旧未跟踪/忽略输出在residual_outputs。Git正常切换科学分支，无reset/force/历史重写，无删除原始证据；三个有效模型目录仍在logs/final2_262460–262462，eval只留代码。
- 62个保留科学Python文件的Git blob与F01固定f2f8169相同；新目录下docs/current、HISTORY、DEAD_ENDS、evidence、docs/archive由已整理科学分支恢复。ASSET_LOCATION_INDEX.csv填入4608个原资产的当前有效路径、备份位置及SHA；GIT_ASSET_ENTRIES.csv给出核验过的历史标签证据路径与索引。旧JSON里的历史绝对路径不改造冒充当时路径，从外部索引找资产。
- 代码测试合计354通过、3预期失败、0最终错误。初次完整运行341通过、3预期失败，13项因未创建basetemp父目录在setup报错；修正目录后仅重跑这13项，13通过。科学代码未改，日志和命令附TEST_VERIFICATION及pytest_setup_retry.log。不能说一次整套运行无报错。
- 本机Spacecraft项目入口、最新交付和主题README已更新。旧两模型阶段交付移入上层交付/历史/F01_STAGE_20261009，迁移前后SHA一致；本轮完整三模型读数和现有650个结果在F01_closed_records_20261010，不含模型或嵌套ZIP。

## 收口而不是补造

271000七个正式行336回合完整（另nominal48仅描述），冻结f2f8169官方三模型METHOD_DOES_NOT_HOLD，problems=[]。现存全部评估650/768，272000缺118，重放12/0/11未齐。Git中没有本机未执行的结果，缺失保留编号，不能用其他开局/实验替代；不统计不完整块排行榜或完整误差指标。

用户已取消剩余记录仿真，本次不补旧六回合行为复现，不能把源码一致或单元测试当成该复现通过。新方法/奖励/接口/阈值没有改，新研究执行仍等上层与用户明确。所有已知旧评估和监督进程已消失，保全中断现场，不重启、不清锁。

## 保留的事项

main未合并，科学/协作分支不互混；7个历史归档标签及6个旧分支处理此前已完成，没有重复改远端引用。旧DRL2_v3e、历史工作树、ORPHAN及769源文件归属歧义保留，没以“无价值”为由删除。新归档与旧C段备份有部分重复，按索引可查，未来确认后再去重。外置离线备份未提供目的地，尚未建立；同盘归档不是离线备份。
'''
write(D/'README.md',report)
write(D/'FILES_SHA256.json',json.dumps({p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'},ensure_ascii=False,indent=2))
write(B/'上层交付/最新/ENGINEERING_RETURN.md',report)
latest=B/'上层交付/最新';write(latest/'FILES_SHA256.json',json.dumps({p.relative_to(latest).as_posix():sha(p) for p in latest.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'},ensure_ascii=False,indent=2))
p=B/'项目说明.md';s=p.read_text(encoding='utf-8').replace('完整pytest核验结果见整理交付，出现错误会如实记录，不冒充全部通过。','完整pytest核验合计354通过、3预期失败；初次13项临时父目录setup错误修复后仅重跑这13项全部通过，详见整理交付。');write(p,s)
p=C/'CURRENT.json';state=json.loads(p.read_text(encoding='utf-8'))
state['local_cleanup'].update(verification='354 passed and 3 expected failures across full-suite run plus 13 setup-error retries; no new formal replay',return_handoff='lower/handoffs/DRL2_return_20261010/README.md',git_clean=True)
state['cleanup_b'].update(step5_condition='User cancelled remaining experiments; tests done, 6 formal replay checks not performed',verification_target='4795910102cf9fbf1dfec74b77e9449623a7b51e',step5_status='CODE_TESTS_DONE_BEHAVIOR_RERUN_NOT_PERFORMED')
write(p,json.dumps(state,ensure_ascii=False,indent=2))
p=C/'lower/README.md';s=p.read_text(encoding='utf-8');s=s.replace('三个种子均已完成60k，F01已判失败，现有评估仅补记录；不据历史执行单自动启动任务，新工作等待用户指令。','三个种子均已完成60k，F01三模型官方METHOD_DOES_NOT_HOLD，现有650个评估已收口归档、缺118不补跑；最新工程已回到DRL2，整体审查待新计划。');s+='\n- [最新工程DRL2整理回执与资产索引](handoffs/DRL2_return_20261010/README.md)：4795910科学分支Git干净，环境与模型SHA不变，354通过/3预期失败；无新增正式仿真。\n';write(p,s)
print('FINAL_SEAL_GIT_ENV_MODEL_HASH_OK')
