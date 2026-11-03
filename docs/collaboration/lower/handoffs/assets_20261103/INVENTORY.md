# 本机资产盘点｜2026-11-03

科学只读检出 `19278093c3b4bfb5a954280becf68ae15f1b3aec`；不训练、不评估、不运行测试、不修改科学代码、不删除文件。

|目录|文件数|GiB|SB3模型ZIP|评估JSON（含保全副本）|ERROR文件|
|---|---:|---:|---:|---:|---:|
|T2_6W_WIDE_140_200|4688|0.187|53|3996|0|
|T2_6W1_CUTOFF_20261029|4708|0.069|17|4008|0|
|T2_7_GUIDE_20261031|7914|0.150|34|7467|0|
|T2_8_KEYSCORE_20261101|不存在（可选）|—|—|—|—|
|T2_8_NIGHT_20261101|2197|0.101|29|1920|0|
|T2_8_P4_INFO_20261102|325|0.009|0|0|0|
|T2_11_EACCA_20261102|1922|0.101|29|1728|0|

评估JSON口径：单文件含opening、event、return三字段；统计全部保全副本，不能当独立试验样本数。各原始C/Q/E_guided/发布快照数量分列在JSON；仅事件/过程错误回执不冒充ERROR_*科学回合错误。ZIP按SB3 data与policy.pth识别模型；所有ZIP均登记哈希。

## 核验

T2.11科学MODEL_SHA256逐项核对 24 项；P4夜间16检查点及P3旧32检查点共48项；不一致数 0。扫描前后大小/修改时间变化 0。全部模型SHA及训练manifest关键字段见INVENTORY.json。

## 保留与待确认候选

指定实验资产全部原位保留。T2.11、T2.8夜间全部checkpoint/final、P3/P4旧checkpoint、P4 info/opp2原件必须保留。TensorBoard与30秒日志本单保持原件，不另压缩。

未来可另行确认的清理候选仅为旧行政故障记录副本、已结束发布的临时Git index、重复中期发布快照；本单未认定任何科学原始资产可删除，未删除上述候选，未执行压缩去重或数据迁移。候选仍保留其审计用途。

具体候选（仅登记，保留原件）：

- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_6W1_CUTOFF_20261029\记录\science_publication.index`，1488637 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_7_GUIDE_20261031\记录\interim_doc_scope.index`，1651019 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_7_GUIDE_20261031\记录\interim_publication_20261101_155147.index`，1651019 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_7_GUIDE_20261031\记录\science_publication.index`，1908647 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_8_NIGHT_20261101\记录\science_publication_1793579168021674400.index`，2053110 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_8_NIGHT_20261101\记录\science_publication_1793581371329230400.index`，2053110 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_8_P4_INFO_20261102\记录\science_publication_1793616520385907600.index`，2058474 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_8_P4_INFO_20261102\记录\science_publication_1793617175229939100.index`，2062968 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。
- `C:\Users\35884\Documents\Spacecraft\过程文件\T2_11_EACCA_20261102\记录\science_publication_1793667137793042800.index`，2187044 bytes：已结束发布的临时Git index，仍保留审计用途；须另行批准后处理。

P4 info/opp2原件80份已另登记SHA256，保持原位。
