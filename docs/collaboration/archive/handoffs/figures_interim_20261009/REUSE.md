# 正式完成后复用（现在不执行）

项目Python解释器：D:/py/DRL2/.venv/Scripts/python.exe。脚本不会启停训练、执行仿真或修改科学工作区。

输入结构：root/pure/seed_<开局>.json，root/learned/<模型>/seed_<开局>.json，root/stopping/<模型>/seed_<开局>.json；基线模式再读root/nominal/seed_<开局>.json。每行要求48开局、seed/row吻合、code_dirty=false。应先做当轮正式的commit、模型SHA、工况与完整性核验，再绘图；绘图不替代官方判读。

```powershell
& 'D:/py/DRL2/.venv/Scripts/python.exe' -B './plot_interim.py' --generic-root 'D:/py/DRL2/eval/final2/formal' --block 271000 --models 262460 262461 262462 --caption '271000–271047，w2.36_r15，60k正式模型，结论按当轮官方判读'
& 'D:/py/DRL2/.venv/Scripts/python.exe' -B './plot_interim.py' --generic-root '<该块完整基线目录>' --block 272000 --baseline-only --caption '272000–272047，w2.36_r15，Pure与nominal，同块基线'
```

复用时输出TOP为脚本父目录的父目录，请将脚本先复制到下一主题的脚本/目录，防止覆盖本临时资产。画失败回合的真实结束时间，无成功筛选；正式方法箱线图同时输出共同成功配对，不用自己的成功样本组成掩盖效率比较。

论文轨迹编号固定规则：每个模型在271000–271047中分别找Pure失败/stopping成功、Pure成功/stopping失败、两者均成功的开局，各取编号最小的一个；这里成功指干净完成。类别为空则如实报告，不换块、不手挑。只有正式结果出来且重放授权到位后才能导出新物理轨迹。
