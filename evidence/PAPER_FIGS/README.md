# 第一轮 ECC 论文案例数据

科学检出 868fe5e60bc8328c6c3fff1c59af4bce7ba2e3e5，执行书协作提交 e664bb9af40ae7b8743477debe7443f97faee788。
认证案例 X = 300005。它是执行书列出的首个开局，ECC 种子 20 的原始审计记录包含 3 次认证介入。
两种方法都使用第一轮种子 20 的 5000 步检查点，原评估和检查点均已核对 SHA-256。

## 重放命令

    D:\py\DRL2\.venv\Scripts\python.exe -B -m experiments.final traj --entry ECC=C:\Users\35884\Documents\Spacecraft\过程文件\FORMAL\B\TRAIN_ECC_CERT_LOOP_s20 --entry SAC=C:\Users\35884\Documents\Spacecraft\过程文件\FORMAL\B\TRAIN_SAC_C_s20 --openings 300000,300005 --output C:\Users\35884\Documents\Spacecraft\过程文件\PAPER_FIGS --condition urgent

## 复现检查

四条轨迹的结局与原评估一致，回报差均在 1e-9 的绝对容差内。

| 文件 | 结局 | 原评估回报 | 重放回报 | 轨迹记录数 | 决策数 |
|---|---|---:|---:|---:|---:|
| traj_ECC_300000.json | success | 64.8948615433783 | 64.8948615433783 | 86 | 6 |
| traj_ECC_300005.json | success | 66.7358719956275 | 66.7358719956275 | 80 | 6 |
| traj_SAC_300000.json | success | 66.3191736509522 | 66.3191736509522 | 83 | 4 |
| traj_SAC_300005.json | timeout | -129.876468362892 | -129.876468362892 | 81 | 8 |
