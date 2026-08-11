# FHiT-KD：DINOv3-SAT → YOLO11m 遥感检测复现工程

本仓库交付的是一条可审计、可中断续训、能证伪自身假设的竞赛训练流程。部署模型始终是纯 YOLO11m；DINOv3、投影器和原型头只在训练期使用。项目不承诺获奖，也不预设蒸馏一定超过新基线；它承诺数据划分、模型差异和结论可以复查。

完整执行手册见 [`docs/SERVER_RUNBOOK_SCENE811_V3.md`](docs/SERVER_RUNBOOK_SCENE811_V3.md)，数据划分报告见 [`docs/SCENE811_V3_SPLIT_REPORT.md`](docs/SCENE811_V3_SPLIT_REPORT.md)，论文证据字段见 [`docs/PAPER_EVIDENCE_PLAN.md`](docs/PAPER_EVIDENCE_PLAN.md)。旧 `scene811_v2` 文档与配置只用于历史结果解释，不再是主流程。

## 当前可交付状态（2026-08-11）

| 状态 | 内容 | 能否据此声称提升 |
|---|---|---|
| 已实测可跑 | V3 构建、D0 审计、服务器二次校验、双数据配方配置生成、旧 F/K/FK 健康保护、比赛规则评估、证据账本 | 只能声称工程与数据 Gate 通过 |
| 已实现待单卡 GPU 验证 | 数据指纹命名空间、P0 教师/缓存/续训隔离、G 全局相关性与 P 留一场景原型的核心计算 | 健康检查、短筛和部署等价性全过后才可进入效果比较 |
| 研究候选 | L 定位教师、真实大图跨窗一致性、人工复核车辆背景原型、分辨率 768/1024 | 未做单变量消融前不能写成已实现创新 |

## 冻结数据版本

`scene811_v3_grouped_clean_r10` 已在本地完成全量哈希校验：

- 输入 ZIP SHA-256：`f66212d1693baa92c6342ddac003775671a9c99e38fb6d26eee2cacd28d63bc5`
- 数据指纹：`b4367981f59e0d04cf7925587582acb0d3f25a2e9b145dfef662a8da8f0797b9`
- 选中 8,211 张：TRAIN 6,870、VAL 674、密封 TEST 667
- 官方 4,481 张：TRAIN 3,140、VAL 674、TEST 667
- 新增 3,730 张：全部只进入 TRAIN
- 场景跨集合、聚类跨集合、完全重复跨集合、未解决强近重复：均为 0
- 1 张已知漏标新增图被排除；1 行完全重复标签仅在派生视图中去重；源 ZIP 不改写
- 官方目标的小目标、密集和边缘比例跨 split 最大差分别为 `0.00158/0.00702/0.00371`

为什么不做普通随机划分：遥感裁片经常来自同一原始产品或相邻地理位置，随机拆分会让训练和验证共享背景纹理。V3 先把卫星产品、L1A 产品、FSC 位置、数字序列、AU/RU 序列、P 标识和强近重复合成不可拆场景组，再在组级联合平衡 25 类、三大组、尺寸、密集、边缘与来源族。新增数据只用于训练，从而让 VAL/TEST 仍代表赛方官方分布。

## 服务器配置

| 项目 | 最低 | 建议 |
|---|---:|---:|
| GPU | RTX 3090/4090/4090D 24GB | RTX 4090/4090D |
| CPU | 8 核 | 16 核以上 |
| 内存 | 32GB | 64GB |
| 可用磁盘 | 80GB（紧凑模式） | 120GB 以上 |
| 系统 | Ubuntu 22.04 | Ubuntu 22.04/24.04 |
| Python | 3.10/3.11 | 3.11 |

50GB 只适合不保留多候选和教师缓存的临时短筛。完整三种子基线、DINO 缓存、GPU 评估和大图验收建议至少 120GB。

## GitHub 不包含、必须手动上传的文件

```text
scene811_v3_grouped_clean_r10.tar
yolo11m.pt
dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

已生成的数据归档：

```text
文件：scene811_v3_grouped_clean_r10.tar
大小：2,189,470,720 bytes
SHA-256：3f98d5458a5e20eef4beee24f7f644189c3a2704f3862675328e5e5c0257b15f
```

数据、标签、比赛 PDF、DINO/YOLO 权重、训练 checkpoint、教师缓存、逐图预测和带文件名的清单均不得提交到公开仓库。若数据许可不允许再分发，归档只在授权的本地与服务器之间传输。

## 从零运行

### 1. 克隆与环境检查

```bash
git clone --recurse-submodules https://github.com/zhengzhezhao057/fhit-kd-yolo11m.git
cd fhit-kd-yolo11m
bash scripts/00_server_bootstrap.sh
export PYTHON="$(command -v python)"
export REPO="$(pwd -P)"
```

已有可用 Conda 环境时：

```bash
FHIT_USE_CURRENT_PYTHON=1 bash scripts/00_server_bootstrap.sh
```

### 2. 校验并解包上传数据

以下只是路径示例，仓库不依赖固定安装目录：

```bash
export INPUT_DIR=/data/fhit-input
export DATASET_ROOT=/data/dataset_scene811_v3_grouped_clean_r10

sha256sum "$INPUT_DIR/scene811_v3_grouped_clean_r10.tar"
# 必须得到 3f98d5458a5e20eef4beee24f7f644189c3a2704f3862675328e5e5c0257b15f

mkdir -p "$(dirname "$DATASET_ROOT")"
tar -xf "$INPUT_DIR/scene811_v3_grouped_clean_r10.tar" -C "$(dirname "$DATASET_ROOT")"

EXPECTED_FINGERPRINT=b4367981f59e0d04cf7925587582acb0d3f25a2e9b145dfef662a8da8f0797b9 \
DATASET_ROOT="$DATASET_ROOT" bash scripts/11_verify_scene811_v3.sh
```

校验必须显示 `passed=true`、`hash_mismatches=0`、`scene_leaks=0`、`cluster_leaks=0`。只做 `--no-image-hash` 的快速检查不能替代正式训练前全量校验。

### 3. 生成双配方、三种子基线配置

```bash
export YOLO_WEIGHTS="$INPUT_DIR/yolo11m.pt"
DATASET_ROOT="$DATASET_ROOT" \
SEEDS=42,3407,20260809 \
BASELINE_EPOCHS=120 \
IMAGE_SIZE=640 \
BASELINE_BATCH=16 \
bash scripts/15_prepare_scene811_v3_configs.sh
```

生成两个严格配方：

- `B-official`：只用 3,140 张官方 TRAIN，回答“新划分本身”带来什么；
- `B-mix`：使用 3,140 官方 + 3,730 新增 TRAIN，回答“新增数据”带来什么。

两者共享同一官方 VAL/TEST、初始化、超参数和种子。主开发初始权重固定使用 `B-mix seed42`，不能挑最好 seed 再做 KD；其余种子用于方差和复现。

### 4. 训练六个基线

```bash
DATASET_ROOT="$DATASET_ROOT" \
YOLO_WEIGHTS="$YOLO_WEIGHTS" \
RECIPES=official,mix \
SEEDS=42,3407,20260809 \
BASELINE_EPOCHS=120 \
bash scripts/20_train_baselines.sh 2>&1 | tee logs/scene811_v3_baselines.log
```

脚本的续训语义：同一 run 只有 `last.pt` 存在才使用 `--resume`；会恢复 optimizer、EMA、scaler、scheduler 与 epoch。目录存在却没有可续训 `last.pt` 时立即停止，绝不生成 `-2/-3` 目录，也不允许用 `best.pt` 冒充精确续训。

### 5. 生成 P0 隔离的 KD 配置

```bash
export BASELINE_WEIGHTS="$REPO/runs/scene811_v3_grouped_clean_r10/b_mix_s42/weights/best.pt"
export DINO_WEIGHTS="$INPUT_DIR/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

DATASET_ROOT="$DATASET_ROOT" \
BASELINE_WEIGHTS="$BASELINE_WEIGHTS" \
DINO_WEIGHTS="$DINO_WEIGHTS" \
bash scripts/15_prepare_scene811_v3_configs.sh

$PYTHON -m src.server_doctor --full --require-config \
  --config configs/generated/scene811_v3/experiment_v3.yaml \
  --min-free-gb 80
```

P0 强制所有 V3 中间物按数据指纹隔离：

```text
runs/scene811_v3_grouped_clean_r10__b4367981f59e/...
cache/teacher_signals/scene811_v3_grouped_clean_r10__b4367981f59e/train/...
cache/prototype_banks/scene811_v3_grouped_clean_r10__b4367981f59e/...
```

旧 V2 教师、缓存、原型库和 checkpoint 会因指纹/清单/权重不匹配而被拒绝。

### 6. 教师、缓存、原型库和接线健康检查

```bash
DATASET_ROOT="$DATASET_ROOT" \
CONFIG="$REPO/configs/generated/scene811_v3/experiment_v3.yaml" \
bash scripts/30_teacher_cache_health.sh 2>&1 | tee logs/scene811_v3_teacher_health.log
```

G/P/GP 只有同时满足下列条件才算“接线有效”：缓存请求全命中、选中分支 raw/weighted loss 非零、训练专用参数有有限非零梯度、路由对象数大于零、健康检查不更新 optimizer/BN、无 NaN/Inf、原型库与数据/教师/缓存指纹一致。任何一项失败都停训修代码，不能把该结果解释为“蒸馏无效”。

### 7. 8 轮单变量短筛

```bash
DATASET_ROOT="$DATASET_ROOT" \
SHORT_EPOCHS=8 \
bash scripts/40_core_short_screen.sh 2>&1 | tee logs/scene811_v3_short_screen.log
```

固定比较：`C0 / G / P / GP`。G 只在 P4/P5 做全局相关性传递；P 使用从同一场景扣除后的 DINO RoI 原型；GP 按组别、尺度和 OOF 失效类型把一个目标路由到一个主分支，避免简单线性相加。此处仍是待 GPU 证实的方法假设。

### 8. 官方 VAL 评估与 Gate

```bash
DATASET_ROOT="$DATASET_ROOT" \
bash scripts/50_evaluate_and_gate.sh 2>&1 | tee logs/scene811_v3_eval_gate.log
```

必须同时记录：25 类 mAP50-95、三大组 Recall/FDR/F1、置信度曲线、class-aware 一对一匹配、车辆 IoU 0.35、船/飞机 IoU 0.50、重复框 FP，以及排除读取后的推理时间。赛方最终比较的是船/飞机/车辆各自 Recall 与 FDR 六项加总耗时一项，共七项排名，不能只优化总 mAP。

### 9. 证据账本

每个正式 run 在训练前初始化账本，续训前追加 checkpoint SHA，完成评估后才不可逆地 `complete --paper-ready`。具体命令见运行手册。汇总：

```bash
REGISTRY=reports/scene811_v3/ledger/experiment_registry.jsonl \
OUT=reports/scene811_v3/ledger_summary \
bash scripts/60_summarize_ledger.sh
```

不要在尚未完成 native、competition、diagnostics、timing 的情况下做 paper-ready completion。

## 结果决策

| 结果 | 下一步 |
|---|---|
| D0/环境/接线/部署等价性失败 | 结果作废，修复后用新 run-name 重跑 |
| B-mix 三 seed 不优于 B-official | 不把新增数据作为主训练配方；先审查标签/来源偏移 |
| G 或 P 只在 seed42 提升 | 复跑另外两 seed，不进入长训 |
| GP 低于最佳单分支 | 保留 G 或 P；不能为了“创新更多”强行联合 |
| 船车改善、飞机非劣、总 mAP 微变 | 进入七项 Pareto 比较，而非只按 mAP 淘汰 |
| 数据课程提升但 KD 不提升 | 收益归因数据，不归因 DINOv3 |
| 三 seed 方向一致且配对置信区间支持 | 只长训最佳 KD 与匹配 C0 |

建议短筛 Gate：overall Recall ≥0.85、FDR ≤0.20；相对 C0 的 mAP50-95 下降不超过 0.002、飞机 F1 下降不超过 0.005；船或车辆至少一组 F1 提升 0.005。它们是开发门槛，不是赛方唯一计分公式。

## 预计耗时（单张 4090/4090D）

| 阶段 | 粗略时间 |
|---|---:|
| 环境、全量数据哈希 | 20–60 分钟 |
| 120 轮单个 640 基线 | 2–4 小时 |
| 六个基线 | 12–24 小时 |
| DINO 教师 30 轮 | 1–2 小时 |
| TRAIN 教师缓存与原型库 | 1–3 小时 |
| G/P/GP 健康检查 | 15–45 分钟 |
| C0/G/P/GP 各 8 轮 | 4–10 小时 |
| 三 seed 正式复核 | 8–24 小时 |
| 10000×10000 大图速度/滑窗验收 | 1–3 小时 |

具体时间受 batch、磁盘吞吐和缓存命中影响，运行日志中的实测 wall time 才可写入论文。

## 文献定位与创新边界

组不重叠并尽量保持类别比例与 `StratifiedGroupKFold` 的目标一致；多标签分层可参考 Sechidis 等人的迭代分层思想。检测蒸馏参考 GID、G-DetKD、DeFeat、FGD、ScaleKD 和梯度引导 KD，但本项目不会把它们的贡献改名后声称首创。2026 年 DisDop 已把 DINOv3/RemoteCLIP 的多层领域先验用于航空开放词汇检测，因此本项目的可检验差异应落在：无泄漏场景证据、留一场景原型、失效条件路由、比赛七项对齐及纯 YOLO 部署，而不是“首次把 DINOv3 用于遥感蒸馏”。文献链接和可证伪假设见最终方案文档。
