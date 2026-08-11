# Scene811 V3-R10 服务器执行手册

本手册对应数据指纹 `b4367981f59e0d04cf7925587582acb0d3f25a2e9b145dfef662a8da8f0797b9`。任何指纹不一致的数据、教师缓存或 checkpoint 都禁止混用。

## 1. 服务器最低配置

- Ubuntu 22.04/24.04，Python 3.11；
- RTX 3090/4090/4090D 24GB，CPU 8 核、内存 32GB；
- 紧凑运行至少 80GB 可用磁盘，建议 120GB；
- 推荐 4090/4090D、16 核、64GB。

## 2. 克隆与环境门禁

```bash
cd /root
git clone --recurse-submodules https://github.com/zhengzhezhao057/fhit-kd-yolo11m.git
cd fhit-kd-yolo11m
bash scripts/00_server_bootstrap.sh
export REPO="$(pwd -P)"
export PYTHON="${PYTHON:-/root/miniconda3/envs/fhit-kd/bin/python}"
```

手动上传到 `/root/rsdet/input/`：

```text
scene811_v3_grouped_clean_r10.tar
yolo11m.pt
dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

公开仓库不包含数据、权重、缓存或训练 checkpoint。

## 3. 解包与逐文件复核

```bash
mkdir -p /root/rsdet/data
tar -xf /root/rsdet/input/scene811_v3_grouped_clean_r10.tar -C /root/rsdet/data
export DATASET_ROOT=/root/rsdet/data/dataset_scene811_v3_grouped_clean_r10

DATASET_ROOT="$DATASET_ROOT" \
EXPECTED_FINGERPRINT=b4367981f59e0d04cf7925587582acb0d3f25a2e9b145dfef662a8da8f0797b9 \
bash scripts/11_verify_scene811_v3.sh
```

必须看到 `passed=true`、`hash_mismatches=0`、`scene_leaks=0`、`cluster_leaks=0`。

## 4. 配置和基线

```bash
export YOLO_WEIGHTS=/root/rsdet/input/yolo11m.pt
DATASET_ROOT="$DATASET_ROOT" YOLO_WEIGHTS="$YOLO_WEIGHTS" \
SEEDS=42,3407,20260809 BASELINE_EPOCHS=120 \
bash scripts/15_prepare_scene811_v3_configs.sh

DATASET_ROOT="$DATASET_ROOT" YOLO_WEIGHTS="$YOLO_WEIGHTS" \
RECIPES=official,mix SEEDS=42,3407,20260809 BASELINE_EPOCHS=120 \
bash scripts/20_train_baselines.sh 2>&1 | tee logs/v3_r10_baselines.log
```

先比较 B-official 与 B-mix，确认新增数据的净收益。后续 KD 固定从 `B-mix seed42` 初始化，不能挑最好 seed。

## 5. 教师、缓存和 KD 健康检查

```bash
export BASELINE_WEIGHTS="$REPO/runs/scene811_v3_grouped_clean_r10/b_mix_s42/weights/best.pt"
export DINO_WEIGHTS=/root/rsdet/input/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth

DATASET_ROOT="$DATASET_ROOT" BASELINE_WEIGHTS="$BASELINE_WEIGHTS" \
DINO_WEIGHTS="$DINO_WEIGHTS" bash scripts/15_prepare_scene811_v3_configs.sh

DATASET_ROOT="$DATASET_ROOT" \
CONFIG="$REPO/configs/generated/scene811_v3/experiment_v3.yaml" \
bash scripts/30_teacher_cache_health.sh 2>&1 | tee logs/v3_r10_teacher_health.log
```

缓存命中、G/P 分支非零损失、非零有限梯度、路由计数和部署等价性任一失败，都必须停训修代码。

## 6. 单变量短筛与决策

```bash
DATASET_ROOT="$DATASET_ROOT" SHORT_EPOCHS=8 \
bash scripts/40_core_short_screen.sh 2>&1 | tee logs/v3_r10_short.log

DATASET_ROOT="$DATASET_ROOT" \
bash scripts/50_evaluate_and_gate.sh 2>&1 | tee logs/v3_r10_eval.log
```

固定比较 `C0/G/P/GP`。只有相对 C0 同时满足以下条件才进入多种子长训：

- mAP50-95 下降不超过 0.002；
- 飞机 F1 下降不超过 0.005；
- 船或车辆至少一组 F1 提升 0.005；
- Recall≥0.85、FDR≤0.20；
- KD 健康与纯 YOLO 部署等价性通过。

GP 不优于最佳单分支时，正式模型采用 G 或 P，不为“联合创新”牺牲指标。

## 7. 正式评估

验证集扫描 confidence×NMS，冻结 operating point；测试集只评一次并复用冻结参数。比赛匹配为 class-aware 三粗类一对一：车辆 IoU=0.35，船/飞机 IoU=0.50，重复框计 FP。记录三类 Recall/FDR 六项以及排除读图后的总推理时长，共七项排名证据。

中断续训只能使用同一 run 的 `last.pt`；代码会核对数据指纹、缓存指纹、目标函数和 optimizer 状态。`best.pt` 只用于评估，不能冒充精确续训。

## 8. 结果分支

- G/P/GP 均未通过：保留 B-mix 或 C0，结论为数据收益、蒸馏假设未被支持；
- 单分支通过：追加两个固定 seed，报告均值、方差和场景 bootstrap 区间；
- 船车提升但飞机下降：降低弱组路由预算，不进入长训；
- 小图冻结后再做 10000×10000 滑窗、边缘融合和 20 秒时限验收，不能用伪大图代替正式证据。

论文记录结构和必须保存的字段见 `docs/PAPER_EVIDENCE_PLAN.md`。
