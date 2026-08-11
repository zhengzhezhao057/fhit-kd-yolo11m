# FHiT-KD v2 全流程证据与论文素材计划

> 适用数据版本：`scene811_v3_grouped_clean_r10` 及其后续冻结版本
> 适用学生模型：YOLO11m
> 目标：让每个实验都能回答“用的是什么数据、代码、初始权重、参数和服务器，训练是否真正发生，为什么选中该模型，能否复现”
> 工具入口：`python -m src.experiment_ledger`

---

## 1. 为什么训练开始前就要设计论文证据

本项目此前已经出现过以下风险：

- 四个权重 SHA 不同，但评估配置过粗，结果完全相同；
- KD 分支缓存未命中、hook 未捕获或梯度为零时可能静默退化为基线；
- 旧数据随机切分使同源场景进入不同集合，验证指标不能代表未知场景；
- 只保留 `best.pt` 和最终 mAP，无法还原训练过程、损失冲突与失败原因；
- 训练目录自动出现 `-2/-3`，最终不清楚论文表格对应哪个 checkpoint；
- 只报告 mAP，不能回答赛方真正排名使用的召回、虚警和时效性问题。

因此，论文素材不是训练结束后的“补日志”，而是正式实验的一部分。没有完整证据的运行只能叫调试，不得进入主表、消融结论或答辩材料。

---

## 2. 赛方指标必须按七个排名维度记录

评分细则对应七个独立排名维度：

1. 舰船 Recall；
2. 舰船 FDR（虚警率）；
3. 飞机 Recall；
4. 飞机 FDR；
5. 车辆 Recall；
6. 车辆 FDR；
7. 总时效性。

其中车辆匹配 IoU 为 `0.35`，舰船和飞机为 `0.50`。正式竞争评估必须启用粗类感知的一对一匹配，不能用 `class_aware_matching=False` 的结果替代。

时效性计时边界必须固定为：

```text
开始：所有待测图像读取完成之后
计入：预处理、模型推理、NMS、滑窗合并（如适用）、结果生成与输出
结束：检测结果完成输出之后
排除：从磁盘读取图像的时间
```

不能把 Ultralytics 日志中的纯网络 `inference ms` 当作赛方总时效性。论文同时报告：

- 测试图像数；
- 总秒数；
- 平均毫秒/图；
- P50/P95 毫秒/图（若逐图计时可得）；
- batch、输入尺寸、GPU、warm-up 次数、重复次数；
- 是否包含滑窗与合并；
- 明确写出 `excludes_image_read=true`。

推荐至少 warm-up 10 次、独立重复 3 次，并使用中位总时长作为主结果。所有候选必须在同一台服务器、相同功耗模式、相同 batch 和相同测试清单上比较。

---

## 3. 不可变运行证据结构

每次运行目录固定为：

```text
runs/scene811_v3_grouped_clean_r10/<experiment>_seed<seed>/
├── evidence/
│   ├── run_manifest.json
│   ├── epoch_events.jsonl
│   ├── resume_events.jsonl
│   ├── completion.json
│   └── snapshots/
│       ├── results_csv.csv
│       ├── kd_health.jsonl
│       ├── native.json
│       ├── competition.json
│       ├── diagnostics.json
│       └── timing.json
├── weights/
│   ├── best.pt
│   ├── last.pt
│   └── best_deploy.pt
├── results.csv
└── kd_health.jsonl                 # KD 运行需要
```

全局注册表：

```text
runs/scene811_v3_grouped_clean_r10/experiment_registry.jsonl
```

`run_manifest.json` 和 `completion.json` 是一次写入的哈希封装。JSONL 中每条事件都包含上一条事件 SHA-256，形成哈希链。工具在追加、续训、完成和汇总前都会重新验证。不得手动编辑这些文件；内容写错时创建一个新运行，不得“修表”。

### 3.1 初始化时冻结的内容

- 数据集 fingerprint、冻结报告路径和 SHA-256；
- Git commit、branch、remote、dirty 状态与 status 摘要 SHA；
- 配置文件 SHA、解析后的完整 resolved config 及其 SHA；
- 完整训练命令和工作目录；
- seed；
- 官方初始化权重 SHA；
- fresh/resume 身份、父运行和续训 checkpoint SHA；
- Python、PyTorch、CUDA、cuDNN、Ultralytics 和关键依赖版本；
- GPU 名称、显存、驱动、主机名和操作系统；
- 开始时磁盘总量、已用量和剩余量。

### 3.2 训练过程中记录的内容

- 每轮 box/cls/DFL、总检测损失、学习率；
- P、R、mAP50、mAP50-95；
- GPU 显存、累计耗时、磁盘剩余；
- KD 原始损失、加权损失和分支 schedule；
- cache request/hit/miss；
- 有效 RoI、教师候选及保留数；
- feature/prototype/localization 分支非零 batch；
- 投影头和学生共享层梯度事件；
- 辅助梯度/检测梯度比例与余弦冲突率；
- NaN/Inf、无梯度、优化器遗漏等健康 Gate。

Ultralytics 的 `results.csv` 是逐轮主来源；ledger 的 `record-epoch` 用于额外实时快照和关键健康事件。正式完成时两者都会被登记。

### 3.3 完成时冻结的内容

- `best.pt`、`last.pt`、`best_deploy.pt` 的绝对路径、大小和 SHA；
- `results.csv` 的完整逐轮数据；
- KD health 的原始 JSONL 和聚合健康判断；
- 原生 25 类 P/R/mAP50/mAP50-95 和逐类 AP；
- 赛方 operating point、总 TP/FP/FN、三粗类 Recall/FDR/F1；
- small/medium/large、crowded、edge 的检出率、IoU50/IoU75 和错误类型；
- FP 的 duplicate/wrong-group/localization/background 分类；
- 不含读图的赛方时效性；
- 训练耗时、结束磁盘状态和全部 resume lineage。

---

## 4. CLI 使用规范

以下示例以正式基线 `B0_seed42` 为例。命令文件应提前写好并纳入 Git；比在 shell 历史中临时拼参数更可靠。

### 4.1 训练开始前初始化

```bash
cd /root/fhit-kd-yolo11m

python -m src.experiment_ledger init \
  --run-dir runs/scene811_v3_grouped_clean_r10/B0_seed42 \
  --experiment B0 \
  --dataset-report artifacts/scene811_v3_grouped_clean_r10/dataset_fingerprint.json \
  --config configs/scene811_v3/B0_seed42.yaml \
  --seed 42 \
  --initial-checkpoint /root/weights/yolo11m.pt \
  --command-file configs/scene811_v3/commands/B0_seed42.sh \
  --registry runs/scene811_v3_grouped_clean_r10/experiment_registry.jsonl
```

看到 `initialized ... manifest_sha256=...` 后才允许启动 GPU 训练。若 Git dirty 为 `true`，该运行只能用于调试；`--paper-ready` 会拒绝把它登记为正式论文运行。正式配置、训练代码和命令文件必须先提交，使初始化时 Git clean。

### 4.2 记录关键 epoch

训练回调可直接调用 Python API；手工记录示例：

```bash
python -m src.experiment_ledger record-epoch \
  --run-dir runs/scene811_v3_grouped_clean_r10/FK_seed42 \
  --epoch 10 \
  --metrics-file reports/live/FK_seed42_epoch10.json \
  --kd-health-file runs/scene811_v3_grouped_clean_r10/FK_seed42/kd_health.jsonl \
  --elapsed-seconds 1320.4 \
  --gpu-memory-gib 19.7
```

相同 epoch 不允许重复登记，epoch 也不能倒退。若训练日志在 resume 后重复行，应先确认 Ultralytics 的 `results.csv` 是否正确恢复，不要覆盖 ledger。

### 4.3 中断后续训

精确续训前先登记实际使用的 `last.pt`：

```bash
python -m src.experiment_ledger resume \
  --run-dir runs/scene811_v3_grouped_clean_r10/FK_seed42 \
  --checkpoint runs/scene811_v3_grouped_clean_r10/FK_seed42/weights/last.pt \
  --command "python -m src.train_ablation --config configs/scene811_v3/FK_seed42.yaml --resume runs/scene811_v3_grouped_clean_r10/FK_seed42/weights/last.pt"
```

该操作重新记录服务器环境、续训命令和 checkpoint SHA。只有优化器、scheduler、EMA、scaler、epoch 和随机状态均来自 `last.pt`，才可以称为“完全拟合续训”。只加载模型权重重新创建优化器必须登记为新运行。

### 4.4 正式完成

基线使用：

```bash
python -m src.experiment_ledger complete \
  --run-dir runs/scene811_v3_grouped_clean_r10/B0_seed42 \
  --status completed \
  --best-checkpoint runs/scene811_v3_grouped_clean_r10/B0_seed42/weights/best.pt \
  --last-checkpoint runs/scene811_v3_grouped_clean_r10/B0_seed42/weights/last.pt \
  --deploy-checkpoint runs/scene811_v3_grouped_clean_r10/B0_seed42/weights/best_deploy.pt \
  --results-csv runs/scene811_v3_grouped_clean_r10/B0_seed42/results.csv \
  --native reports/scene811_v3/B0_seed42_native.json \
  --competition reports/scene811_v3/B0_seed42_competition.json \
  --diagnostics reports/scene811_v3/B0_seed42_diagnostics/summary.json \
  --timing reports/scene811_v3/B0_seed42_timing.json \
  --model-key B0 \
  --operating-point best_f1 \
  --paper-ready
```

KD 候选额外加入：

```bash
  --kd-health runs/scene811_v3_grouped_clean_r10/FK_seed42/kd_health.jsonl \
  --require-kd-health
```

`--paper-ready` 会拒绝 Git dirty、`class_aware_matching` 未开启或缺少 results/native/competition/diagnostics/timing 的运行；`--require-kd-health` 会再拒绝缺少/未通过 KD 健康证据的运行。调试运行可以不加此 Gate，但不得进入论文主表。

时效性 JSON 最小格式：

```json
{
  "protocol": "competition_no_image_io_v1",
  "excludes_image_read": true,
  "interval_start": "after_all_image_reads_complete",
  "interval_end": "after_result_output_complete",
  "included_stages": ["preprocess", "inference", "postprocess", "result_output"],
  "image_count": 455,
  "total_seconds": 9.1,
  "p50_ms_per_image": 19.5,
  "p95_ms_per_image": 23.8,
  "warmup_iterations": 10,
  "repetitions": 3,
  "batch": 8,
  "image_size": 640,
  "device": "NVIDIA GeForce RTX 4090 D"
}
```

### 4.5 自动生成论文对比表

```bash
python -m src.experiment_ledger summarize \
  --registry runs/scene811_v3_grouped_clean_r10/experiment_registry.jsonl \
  --out-dir reports/scene811_v3/experiment_ledger
```

生成：

```text
experiment_summary.json   # 完整嵌套证据
experiment_summary.csv    # 统计与绘图数据源
experiment_summary.md     # 快速审阅表
```

---

## 5. 数据集论文证据

### 5.1 冻结前必须保存

- 原始 ZIP 名称、字节数和 SHA-256；
- 官方/新增/隔离样本数；
- 25 细类、3 粗类的图像数与实例数；
- 每类 small/medium/large、crowded、edge 分布；
- scene group 规则、分组数量和最大组大小；
- 每个 split 的来源、场景和类别分布；
- 跨 split 完全重复、近重复、同场景泄漏检查；
- 空标签和异常标签人工复核记录；
- 所有排除、修复、补标操作的 patch manifest；
- 唯一 dataset fingerprint。

### 5.2 论文应回答的问题

1. 为什么不能采用 ZIP 自带随机划分？
2. 为什么新增数据只进入训练集？
3. 场景隔离后验证指标为什么可能下降但可信度反而提高？
4. 新增数据对船、车、飞机的有效覆盖分别是多少？
5. TRAIN/VAL/TEST 的目标尺寸和拥挤度是否可比？
6. 车辆人工也难辨认时，如何避免把漏标真目标当成背景？

### 5.3 推荐图表

- 图 1：官方与新增数据来源、scene group 和 split 的 Sankey/流程图；
- 图 2：随机划分与场景隔离划分的泄漏示意；
- 图 3：三集合的粗类/尺寸/拥挤度分布；
- 表 1：数据来源和清洗前后数量；
- 表 2：泄漏 Gate 与数据 fingerprint；
- 附录表：所有人工复核样本编号、结论和复核人。

---

## 6. 预注册实验问题与表格

不要先看到测试结果再改变“成功”的定义。正式训练前冻结以下问题和 Gate。

### Q1：新增数据是否真正改善未知官方场景？

| 实验 | 官方训练 | 经审新增训练 | KD | 目的 |
|---|---:|---:|---:|---|
| B-official | 是 | 否 | 否 | 纯官方证据基线 |
| B-mix | 是 | 是 | 否 | 测量新增数据本身收益 |

两者必须使用相同官方 VAL/TEST、初始化、epoch、增强、batch 和 3 个种子。若 B-mix 只改善训练/新增域而降低官方场景，则新增数据不能直接成为主训练集，需要调整采样或隔离来源。

### Q2：DINOv3 的哪类知识有效？

| 实验 | 全局相关特征 G | 场景隔离原型 P | 定位 L | 背景 B |
|---|---:|---:|---:|---:|
| C0 | 0 | 0 | 0 | 0 |
| G | 1 | 0 | 0 | 0 |
| P | 0 | 1 | 0 | 0 |
| G+P | 1 | 1 | 0 | 0 |
| L | 0 | 0 | 1 | 0 |
| G+P+L | 1 | 1 | 1 | 0 |
| G+P+L+B | 1 | 1 | 1 | 1 |

每个模块先做单变量短筛。组合实验只有在组成模块分别通过 Gate 后才能启动，从而区分“模块无效”和“损失打架”。

### Q3：failure-conditioned routing 是否比固定线性加权可靠？

固定总 KD 预算，比较：

- 静态线性加权；
- 仅 epoch schedule；
- size-aware routing；
- failure-conditioned routing；
- failure-conditioned routing + 负梯度投影/跳过。

除最终指标外，必须报告辅助/检测共享梯度比、负余弦率、被跳过比例和每类路由样本数。若路由没有改变样本分配或梯度，不能声称其有效。

### Q4：人工确认车辆背景是否减少真实虚警？

比较同一学生起点：

- 无背景样本；
- 未经人工复核的模型挖掘背景（只作风险对照，不能进入最终模型）；
- 人工确认 hard background；
- 随机背景数量匹配对照。

主证据不是总体 mAP，而是 vehicle background FP、vehicle FDR、vehicle Recall 和被误伤真目标数。只有 FDR 下降且 Recall 不发生不可接受下降，才能判为有效。

### Q5：纯 YOLO 部署是否保留训练收益？

对 `best.pt` 与剥离教师/adapter 后的 `best_deploy.pt` 做预测逐框一致性、参数量、GFLOPs、显存和七项竞赛指标对比。部署模型不应携带 DINOv3，也不应增加 YOLO11m 推理结构。

---

## 7. 统计与晋级规则

### 7.1 正式三种子

推荐配对种子：

```text
42, 3407, 20260810
```

每个候选和 C0 使用相同的三个种子。报告均值、标准差和逐 seed 配对差值，不能只挑最好 seed。

### 7.2 场景级置信区间

检测实例并非独立样本，同一场景裁片高度相关。因此 bootstrap 单位必须是 `scene_group`，不是单个框。建议 2000 次场景级重采样，报告：

- mAP50-95 差值 95% CI；
- 三粗类 Recall/FDR/F1 差值 95% CI；
- small/crowded/edge 子集差值 95% CI；
- 车辆背景 FP 差值 95% CI。

### 7.3 建议晋级 Gate

短筛进入长训至少满足：

- KD health 全通过，cache miss 为 0；
- 与同 seed C0 相比 mAP50-95 不显著恶化；
- 船或车至少一个目标指标出现预注册方向的改善；
- 飞机 F1 下降不超过 0.2 个百分点；
- 纯 YOLO deploy parity 通过。

最终候选至少满足：

- 3 个 seed 中至少 2 个优于配对 C0；
- mAP50-95 平均提升建议不少于 0.3 个百分点，或七排名指标出现更有比赛价值的稳定改善；
- 舰船或车辆 F1 平均提升建议不少于 1 个百分点；
- vehicle FDR 下降不能靠大幅牺牲 Recall 获得；
- 场景级 CI、失败案例和时效性均完整；
- TEST 只在配置冻结后使用一次，不回到 TEST 调参。

这些数值是项目决策 Gate，不应伪装成统计显著性阈值。样本不足时如实报告不确定性。

---

## 8. 论文主表、消融表和图的预先规划

### 8.1 主结果表

必须同时包含：

| Model | Params | GFLOPs | mAP50 | mAP50-95 | Ship R/FDR | Aircraft R/FDR | Vehicle R/FDR | Time* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

脚注：`Time*` 排除读图，包含预处理至结果输出。车辆 IoU=0.35，其余 IoU=0.50。

### 8.2 消融表

- 数据：official / mixed / source-aware mixed；
- 蒸馏：G、P、L、B；
- routing：static / scheduled / failure-conditioned；
- prototype：随机划分原型 / scene-held-out OOF 原型；
- background：none / random / mined-unverified / human-verified；
- resolution：640 / 768 / final 1024；
- large inference：direct / overlap sliding window / edge-weighted merge。

每张消融表只改变一个主要因素。若同时修改数据、输入尺寸、增强和损失，就不能把收益归因于蒸馏。

### 8.3 关键曲线

- 训练/验证 detection loss 和 mAP50-95；
- 各 KD 分支 raw/weighted loss；
- KD/检测共享梯度比；
- det-feature、det-prototype、feature-prototype 余弦；
- 三粗类 PR/FDR-Recall 曲线；
- confidence-NMS 二维 operating point；
- size/crowded/edge 的相对增益；
- 场景级 bootstrap 差值分布；
- 不含读图的延迟箱线图。

### 8.4 定性图

每种模型使用同一批、同一阈值、同一版式：

- 密集相邻舰船由漏检变为正确分离；
- 小车辆由低置信变为正确检出；
- 典型车辆背景误检被抑制；
- 飞机精度保持；
- edge/部分目标；
- 失败案例：定位错、背景误检、极小目标无候选。

不能只展示成功图。主文至少放一组仍失败的例子，并在讨论中解释限制。

---

## 9. 真正可以主张的创新与边界

### 9.1 不得使用的表述

截至 2026 年，DisDop 已提出 RemoteCLIP 与 DINOv3 的多层视觉、文本和上下文蒸馏。因此本项目不得宣称：

- “首个将 DINOv3 用于遥感检测蒸馏”；
- “首个 DINOv3 全局—局部蒸馏”；
- “首次进行多层语义蒸馏”。

参考：[DisDop, arXiv:2605.24639](https://arxiv.org/abs/2605.24639)。引用它并清晰说明区别，反而比回避相关工作更可信。

### 9.2 有潜力形成项目辨识度的组合

以下内容必须由消融和审计证据支撑，当前只能称为“候选贡献”：

1. **Scene-held-out prototype distillation**
   原型只由训练场景或 OOF 预测建立，禁止使用当前样本自身教师答案和 VAL/TEST 场景；重点证明它在场景迁移下比随机切分原型可靠。

2. **Failure-conditioned distillation routing**
   根据 OOF 错误类型、目标尺寸、拥挤、边缘和教师可靠性决定 G/P/L/B 分支，而不是对全部样本固定线性加权；重点证据是路由覆盖、梯度预算和对应失败类型的改善。

3. **Human-verified vehicle background bank**
   模型挖掘只生成候选，人工排除漏标真目标后才作为困难背景；重点解决赛方车辆 FDR，而不是盲目添加大量“负样本”。

4. **Competition-seven-objective evidence alignment**
   训练选择同时面向三类 Recall/FDR 和排除读图的总时效性，避免只优化通用 mAP；这更适合作为工程方法与评估协议贡献，不应包装为全新的理论损失。

5. **Teacher-free deployment**
   DINOv3、prototype bank 和路由器仅参与训练，最终仍是标准 YOLO11m；通过 deploy parity 和时效性证明训练创新不增加部署负担。

组合的吸引力不来自模块数量，而来自一个连贯问题链：

```text
场景泄漏让旧结论不可信
    → scene-held-out 数据与原型保证证据边界
    → OOF 暴露船车的具体失败类型
    → failure-conditioned routing 把有限 KD 预算投向可受益样本
    → 人工背景确认抑制车辆虚警且不误伤真目标
    → 蒸馏部件全部剥离，保留纯 YOLO11m 时效性
```

只有每个箭头都被实验证明，这才是一项完整作品；若某模块没有稳定收益，应从最终模型移除，但可以作为负结果写入方法分析。

---

## 10. 重大意义如何写得可信

可以强调：

- 小样本、长尾和同源裁片是遥感比赛的真实难点；
- 场景级隔离比随机图像划分更接近卫星对未知地区的部署；
- 车辆 IoU 虽较低，但背景误检仍会直接伤害 FDR，不能只提高召回；
- DINOv3 的价值应通过“什么场景、什么目标、什么错误受益”解释，而不是只报一个总体 mAP；
- 纯 YOLO11m 部署使高成本教师只发生在离线训练，适合受限算力推理。

避免：

- 用单次 seed 的 0.1 个百分点变化宣称显著突破；
- 把 VAL 调参结果写成未知 TEST 泛化结论；
- 把相关性写成因果；
- 把 640 小图 pseudo-large 拼接结果等同真实 10000×10000 图；
- 把人工复核过的 VAL/TEST 失败案例重新放回训练；
- 把未通过消融的模块留在最终模型只为“创新点更多”。

好的答辩不是说“我们用了很多先进技术”，而是能展示：问题如何被可靠测量、每个设计如何对应失败类型、提升是否跨种子和场景成立、部署代价是否保持。

---

## 11. 每阶段结束时必须归档的清单

### D0 数据冻结

- 数据 fingerprint；
- split/source/patch manifest；
- 泄漏与近重复报告；
- 数据统计 CSV/JSON 和复核记录；
- 数据构建命令、Git SHA 和原始 ZIP SHA。

### B0/B-mix 基线

- 三 seed immutable manifest；
- 逐轮 CSV、best/last/deploy SHA；
- native、competition、diagnostics、timing；
- 场景级差值和新增数据收益结论。

### Teacher/Prototype

- DINOv3 权重 SHA；
- teacher/prototype 构建数据清单和场景排除证明；
- train/val teacher reliability、置信度和熵；
- prototype 每类支持数、OOF fold、特征层和归一化方式；
- cache key 中的数据/教师/代码/尺寸/变换 SHA。

### KD 短筛

- wiring health；
- 非零 loss/gradient/cache 命中；
- 共享梯度预算和冲突；
- 与同 seed C0 的配对差；
- 晋级或淘汰的书面理由。

### 长训与最终模型

- 3 seed 全量 evidence；
- 预注册 Gate 结果；
- deploy parity；
- 七排名指标与时效性；
- 阈值冻结记录；
- 大图滑窗配置和合并策略；
- 最终 checkpoint SHA 和只读归档。

---

## 12. 相关方法定位

- [DINOv3](https://arxiv.org/abs/2508.10104)：高质量自监督视觉表征；本项目将其限定为训练期语义教师。
- [PKD](https://proceedings.neurips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html)：异构检测器的 Pearson correlation 蒸馏依据。
- [FGD](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)：前景、背景和全局关系的检测蒸馏。
- [ScaleKD](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html)：尺度感知小目标蒸馏。
- [Localization Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)：定位分布蒸馏。
- [CrossKD](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)：减少检测蒸馏目标冲突的 cross-head 思路。
- [PCGrad](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)：冲突梯度处理；若采用必须报告触发频率和收益。
- [SAHI](https://arxiv.org/abs/2202.06934)：大图切片辅助推理依据。
- [DisDop](https://arxiv.org/abs/2605.24639)：已有 RemoteCLIP+DINOv3 多层视觉/文本/上下文遥感蒸馏，是创新声明必须对照的近邻工作。

引用方法不等于方法自动有效。本项目最终保留哪些模块，只由同一数据 fingerprint、同一预算、配对种子下的证据决定。

---

## 13. 完成判据

当且仅当以下条件同时满足，才能把模型称为“正式候选作品”：

- D0 数据 Gate 通过，VAL/TEST 无新增数据和场景泄漏；
- 每个正式运行有可验证 manifest、逐轮记录和 completion；
- KD 模型无静默失效，deploy parity 通过；
- 至少三种子、场景级置信区间和失败案例完整；
- 三粗类 Recall/FDR 与总时效性七项证据齐全；
- 创新声明与 DisDop 等已有工作边界清楚；
- 最终模型仍是纯 YOLO11m，且大图流程单独验证；
- 所有表格可由 registry 自动重建，而不是手工抄写。

该证据框架不能保证获奖，也不能把无效蒸馏变成有效结果。它的作用是把“可能有效的想法”转化为经得住评审追问、可复现、可归因且不过度宣传的作品证据。
