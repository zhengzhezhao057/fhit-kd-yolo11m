# FHiT-KD：DINOv3-SAT → YOLO11m遥感检测交付工程

这是一个面向服务器复现和竞赛交付的独立项目。部署模型始终是纯YOLO11m；DINOv3、投影层和RoI头只在训练阶段存在。

项目解决四个已确认问题：

1. 6699张数据中官方4481张与新增2218张来源不同，不能随机混入验证/测试；
2. 同源裁片和近重复场景必须作为不可拆分簇，防止数据泄漏；
3. 旧全局F/K/FK虽然接线有效，但没有稳定超过匹配C0；
4. 飞机接近饱和，船和车辆主要受低置信度、密集目标和背景误报影响。

新主线是“失效感知、25→3层级蒸馏”（FAH-KD）：先用场景分组OOF模型识别每个训练目标的失效类型，再按失效类型选择特征或语义知识，并让粗粒度船/飞机/车辆关系与细粒度25类关系分别蒸馏。

完整甲方交付说明见 [`docs/CLIENT_DELIVERY.md`](docs/CLIENT_DELIVERY.md)。

## 服务器最低配置

| 项目 | 最低 | 建议 |
|---|---:|---:|
| GPU | RTX 3090/4090/4090D 24GB | RTX 4090/4090D |
| CPU | 8核 | 16核 |
| RAM | 32GB | 64GB |
| 可用磁盘 | 80GB | 120GB以上 |
| 系统 | Ubuntu 22.04 | Ubuntu 22.04/24.04 |
| Python | 3.10/3.11 | 3.11 |

50GB只适合不保留多轮权重的紧凑短筛。完整OOF、教师缓存和大图阶段建议120GB以上。

## 必须手动上传的文件

```text
/root/rsdet/input/dataset_6699_scene811.tar.gz
/root/rsdet/input/official_4481_images.txt
/root/rsdet/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
/root/rsdet/weights/yolo11m.pt
```

其中`official_4481_images.txt`每行一个官方图片文件名。它是数据来源审计的刚性输入，不能根据模型结果反推或伪造。`yolo11m.pt`可由Ultralytics自动下载，但手动上传更可复现。

最终大图阶段可额外上传：

```text
/root/rsdet/large/652DDE13070AAD6D16C5A5D19A4192FD.jpg
```

不需要上传旧教师缓存、旧OOF模型或旧F/K/FK中间权重。

## 从GitHub克隆

```bash
cd /root
git clone --recurse-submodules https://github.com/zhengzhezhao057/fhit-kd-yolo11m.git
cd /root/fhit-kd-yolo11m
bash scripts/00_server_bootstrap.sh
```

以后所有命令均在仓库根目录运行，并固定使用：

```bash
PY=/root/miniconda3/envs/fhit-kd/bin/python
```

## 快速执行顺序

### 1. 解压输入数据

```bash
mkdir -p /root/rsdet/input/dataset_6699_scene811
tar -xzf /root/rsdet/input/dataset_6699_scene811.tar.gz \
  -C /root/rsdet/input/dataset_6699_scene811 --strip-components=1
```

确认存在：

```text
images/train、images/val、images/test
labels/train、labels/val、labels/test
dataset.yaml
```

### 2. 生成来源清单和无泄漏新划分

```bash
bash scripts/10_freeze_dataset.sh
```

通过条件：

- 总图像6699；
- official=4481，added=2218；
- VAL/TEST只含official；
- 同一`cluster_id`不跨集合；
- 不存在跨集合完全相同图像SHA256。

脚本通过硬链接建立新数据视图，不重复占用一份图片空间。

### 3. 生成基线配置

```bash
cp configs/baseline.example.yaml configs/baseline.yaml
```

若仓库路径保持默认，只需确认：

```yaml
data: /root/fhit-kd-yolo11m/artifacts/scene811_v2/split/dataset.yaml
```

### 4. 训练双种子新基线

```bash
bash scripts/20_train_baselines.sh
```

输出：

```text
runs/scene811_v2/baseline_seed0/weights/best.pt
runs/scene811_v2/baseline_seed1/weights/best.pt
```

中断后重新执行同一脚本，会从`last.pt`恢复optimizer、EMA、scaler、scheduler和epoch。

### 5. 生成KD配置

```bash
$PY -m src.prepare_config \
  --template configs/fhit.example.yaml \
  --data-yaml artifacts/scene811_v2/split/dataset.yaml \
  --baseline-weights runs/scene811_v2/baseline_seed0/weights/best.pt \
  --dino-weights /root/rsdet/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --out configs/experiment.yaml

$PY -m src.build_kd_configs \
  --base configs/experiment.yaml --mode global --out configs/global_kd.yaml

$PY -m src.server_doctor --full --require-config \
  --config configs/global_kd.yaml --min-free-gb 80
```

### 6. 教师、缓存和KD健康检查

```bash
bash scripts/30_teacher_cache_health.sh
```

必须看到：

- cache miss=0；
- F的feature loss和projector梯度非零；
- K的RoI、KL和梯度非零；
- FK两分支同时非零；
- health check不执行optimizer step；
- 不出现NaN/Inf。

### 7. 第一轮短筛：C0与传统Global-KD

```bash
bash scripts/40_core_short_screen.sh
bash scripts/50_evaluate_and_gate.sh
```

第一轮只回答“普通全局蒸馏是否值得继续”。不先加入OOF、背景头或大图模块。

### 8. OOF失效图谱

```bash
$PY -m src.build_scene811_oof_folds \
  --dataset-root artifacts/scene811_v2/split \
  --manifest artifacts/scene811_v2/split/split_manifest.csv \
  --out artifacts/scene811_v2/oof3 --folds 3 --seed 20260808

$PY -m src.run_oof_training \
  --config configs/global_kd.yaml \
  --folds-dir artifacts/scene811_v2/oof3 \
  --initial-weights /root/rsdet/weights/yolo11m.pt \
  --epochs 30 --batch 16 --workers 4 --device 0

$PY -m src.mine_oof_hard_examples \
  --config configs/global_kd.yaml \
  --folds-dir artifacts/scene811_v2/oof3 \
  --out reports/scene811_v2/oof_mining \
  --confidence-floor 0.01 --positive-confidence 0.50 \
  --negative-confidence 0.35 --batch 16
```

每张TRAIN图片只能由没有训练过其场景簇的fold模型诊断。

### 9. 生成FAH-KD配置并短筛

```bash
$PY -m src.build_kd_configs \
  --base configs/experiment.yaml --mode fah \
  --hard-manifest reports/scene811_v2/oof_mining/hard_examples_oof.json \
  --out configs/fah_kd.yaml

bash scripts/40_core_short_screen.sh
bash scripts/50_evaluate_and_gate.sh
```

FAH-KD配置会启用：

- OOF失效类型对应的分支权重；
- 船/车强化、已正确飞机弱蒸馏；
- 25细类与3粗类分层KL；
- 教师粗类正确性和置信度门控；
- 固定共享梯度预算。

### 10. 按Gate决定下一步

结果写入：

```text
reports/scene811_v2/core/gate_decision.json
```

| 结果 | 下一步 |
|---|---|
| KD健康失败或部署不等价 | 停止，修代码，不解释精度 |
| Recall<0.85或FDR>0.20 | 淘汰该模型 |
| mAP50-95下降超过0.002 | 淘汰，优先检查定位损失冲突 |
| 飞机F1下降超过0.005 | 淘汰或降低飞机KD预算 |
| 船/车F1均无至少0.005提升 | 不长训，重新设计失效路由 |
| seed0通过 | 使用seed1重复 |
| 两个seed方向一致 | 才进入40～100轮正式训练 |
| replay提升但KD不提升 | 收益归因于数据课程，不归因于DINOv3 |

### 11. 最终训练

只延长通过Gate的一个KD候选和一个匹配C0，不能同时长训所有模块。修改对应配置的：

```yaml
student:
  epochs: 40
  save_period: -1
```

使用新run-name启动；中断后加`--resume`：

```bash
$PY -m src.train_ablation --config configs/fah_kd.yaml --exp fk --run-name finalist_fah_seed0
$PY -m src.train_ablation --config configs/fah_kd.yaml --exp fk --run-name finalist_fah_seed0 --resume
```

### 12. 大图推理

小图最终模型冻结后才开始筛选滑窗参数：

```bash
$PY -m src.large_inference \
  --config configs/fah_kd.yaml \
  --model runs/finalist_fah_seed0/weights/best_deploy.pt \
  --source /root/rsdet/large \
  --tile-size 800 --tile-stride 480 --batch 16 --imgsz 640 \
  --conf 0.01 --tile-nms-iou 0.50 --global-nms-iou 0.65 \
  --merge-mode coarse --out reports/scene811_v2/large/predictions.json
```

只筛选`800/480`、`1024/640`和必要时`1280/800`三组，不做无界搜索。最终必须满足10000×10000图像推理不超过20秒。

## 输出定义

最终甲方接收：

```text
best_deploy.pt              纯YOLO11m部署权重
frozen_inference.json       固定conf/NMS/tile/stride
dataset_fingerprint.json    数据谱系
gate_decision.json          模型晋级证据
native.json                 25类mAP
*_competition.json          比赛Recall/FDR/船车飞机指标
kd_health.jsonl             蒸馏真实生效证据
predictions.json            大图/比赛输出
```

## 当前边界

- 本项目不能保证某个KD模型必然超过新基线；它保证每个结论可归因、失败会停止、成功可以复现。
- `official_4481_images.txt`缺失时不能建立可信官方VAL/TEST，流程应当阻断。
- 新划分后旧教师缓存和旧OOF结果均禁止复用。
- TEST冻结前不得用于阈值或模型选择。
