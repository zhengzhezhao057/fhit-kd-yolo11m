# Scene811 V3 最终训练方案

> 方案名称：**FHiT-KD v2：失效路由的全局—局部尺度感知 DINOv3 蒸馏**
> 最终部署模型：**纯 YOLO11m**
> 数据版本：`scene811_v3_grouped_clean_r10`
> 文档版本：2026-08-10
> 状态：**训练设计已冻结；数据冻结器与 v2 蒸馏模块完成实现、测试并推送后，才允许正式长训。**

---

## 1. 最终决策

本轮不直接使用最新 ZIP 自带的 `train/val/test` 开始长训，也不直接复跑旧版 F/K/FK。

最终路线固定为：

1. 从最新 ZIP 重新构建无场景泄漏的数据集；
2. `VAL/TEST` 只使用赛方官方 4481 张中的场景组，新增数据只进入 `TRAIN`；
3. 从官方 `yolo11m.pt` 重新训练匹配基线，旧 `best.pt` 只作历史参考，不作新证据模型初始化；
4. DINOv3 负责语义蒸馏，但不再把 stride-16 token 简单上采样后当成真实 P3；
5. 采用“全局语义 + 局部目标原型 + 可靠性/失效路由”，重点处理小车辆、密集船和背景误报；
6. 定位蒸馏是有资格门槛的可选分支，只在高分辨率检测教师确实优于 YOLO11m 时启用；
7. 所有模块先单变量短筛，再组合；未通过 Gate 的模块不得进入长训；
8. 最后只长训一个匹配 C0 和一个晋级 KD 候选，使用 3 个配对种子给出统计结论；
9. 小图模型冻结后，再单独优化 10000×10000 大图滑窗与合并。

这一方案保持“DINOv3 蒸馏 YOLO11m”的基础方向，但解决旧方案已暴露的三个根本问题：

- 单尺度 DINO token 无法通过插值补回真正的小目标细节；
- 只在 GT RoI 上做 25 类 KL，无法直接学习“车辆 vs 背景”；
- DINOv3 是语义教师，不能单独承担高 IoU 框定位监督。

---

## 2. 最新数据包审计结论

### 2.1 权威输入

```text
/authorized/input/场景区分+811划分(赛方+新增数据).zip
```

文件信息：

```text
size   = 2,174,241,917 bytes
sha256 = F66212D1693BAA92C6342DDAC003775671A9C99E38FB6D26EEE2CACD28D63BC5
```

任何服务器训练都必须先核对该 SHA-256。哈希不一致时不得沿用本文的统计和划分清单。

### 2.2 活跃数据统计

| Split | 图片 | 标注框 | 空标签 |
|---|---:|---:|---:|
| Train | 6977 | 24715 | 4 |
| Val | 780 | 2781 | 0 |
| Test | 455 | 2148 | 1 |
| 合计 | 8212 | 29644 | 5 |

来源构成：

| Split | 官方 | 新增 |
|---|---:|---:|
| Train | 3581 | 3396 |
| Val | 445 | 335 |
| Test | 455 | 0 |
| 合计 | 4481 | 3731 |

已确认：

- 图片与标签一一对应；
- 非法 YOLO 框为 0；
- 活跃图片字节级完全重复为 0；
- 最新清洗将 903 对低可信或非发射车 AU/RU 样本移入隔离区；
- 车辆框由旧版 8923 个降至 4984 个，清洗方向正确；
- `data.yaml` 注释仍写“4481 张、8:1:1”，已失效，不得作为数据证明。

### 2.3 训练前必须修复的问题

最新 ZIP 的标签质量比旧版好，但自带划分仍不满足最终证据要求：

- 335 张新增数据进入了 `VAL`；
- AU/RU 的 9 个序列族中有 8 个跨 `TRAIN/VAL`；
- 数字命名船图按前三段归组后有 77 个组跨 `TRAIN/VAL`；
- `P*.bmp` 归一化编号后有 5 个同源组跨 `TRAIN/VAL`；
- 官方坐标数据有 11 个相同 L1A 产品跨集合；
- 官方 FSC 按物理地点归一化后有 6 个地点跨集合；
- `labels/train/fsc_TG-N25.20-E121.42-lv20-Google_crop0001.txt` 有 1 行完全重复标签；
- `labels/train/4_8_96_10345.txt` 是新增空标签，但此前视觉复核发现图中存在明显船只，不能当负样本使用。

因此，**最新 ZIP 是新的原始数据源，不是可直接长训的最终划分。**

### 2.4 当前自带划分与 V3 场景划分的影响

两种数据集使用的是同一批主要原始数据，区别主要在于场景隔离、数据来源和训练采样，而不是简单增加或删除大量图片。

影响需要分成三个层面判断：

| 影响对象 | 影响程度 | 原因 |
|---|---|---|
| 训练时间和数据规模 | 较小 | 当前 Train 为 6977 张，V3 预计仍约 6800–7000 张 |
| 模型最终泛化能力 | 中等到较大 | V3 会改变官方/新增样本的有效采样比例，并消除同源场景捷径 |
| 验证结果与结论可信度 | 很大 | 当前 Val 的船车指标被新增数据和跨集合序列明显影响 |

当前 `VAL` 的具体构成说明了问题的严重性：

```text
vehicle: 311 个框，其中新增数据 290 个，约占 93%
ship:    624 个框，其中新增数据 380 个，约占 61%
```

因此，当前车辆验证结果主要反映新增 AU/RU 数据，而不是赛方官方 FSC 的未知场景泛化。船舶验证也受到新增同源序列影响。飞机没有对应新增数据，三粗类的验证来源并不一致。

重新划分后可能出现：

```text
新 VAL 指标低于当前 VAL，但真实未知场景泛化更好
```

这是消除泄漏后验证集变难造成的正常现象，不能直接解释为模型退步。新旧验证指标也不能横向放在同一张主表中比较；只有在相同 V3 数据指纹、相同初始化、相同训练预算下得到的模型才能用于正式消融。

V3 并不会大规模丢弃可信数据：

- 官方 4481 张全部保留，并按不可拆分场景组重新分配；
- 3730 张可信新增数据全部进入 `TRAIN`；
- 只排除未补标的 `4_8_96_10345.jpg`；
- 只机械去除 1 条重复标签；
- 已隔离的 903 对 AU/RU 样本继续留在 quarantine，不自动恢复；
- 通过来源感知采样控制约 `70% official / 30% added`，避免新增序列在梯度中占主导。

使用边界固定为：

| 数据版本 | 允许用途 | 禁止用途 |
|---|---|---|
| 最新 ZIP 自带划分 | 解压检查、环境检查、代码接线、1 epoch KD 健康测试 | 120 轮长训、模型选择、阈值选择、论文/答辩结论 |
| V3 场景划分 | B-official/B-mix、正式 C0、KD 消融、阈值冻结、最终报告 | 在 TEST 上反复调参 |

结论：重新划分的首要价值不是承诺立刻提高若干百分点，而是保证最终提升来自可泛化能力，而不是同源场景泄漏。正式 GPU 预算必须投入 V3，不应在当前自带划分上重复长训。

---

## 3. V3 数据冻结规范

### 3.1 数据取舍

主实验数据只使用：

- 官方 4481 张：全部保留；
- 最新 ZIP 活跃区内的高可信新增数据；
- 903 对 `curation_quarantine` 隔离样本默认全部不参与主实验；
- `4_8_96_10345.jpg` 在完成补标前排除；
- 重复标签行只在派生数据视图中去重，原 ZIP 保持只读；
- 4 张已经人工确认的官方空标签图可作为真实背景，并在审计清单中记录复核人和结论。

`balance_holdout` 中视觉上可能有效的发射车，暂不自动恢复。只有在主实验完成后，经过独立复核并作为单独数据变量时，才允许加入。

### 3.2 来源身份

必须生成并冻结：

```text
official_4481_manifest.csv
```

至少包含：

```text
image_name,image_sha256,label_sha256,source_family,scene_group
```

不能通过模型预测结果反推“官方/新增”身份。服务器需上传该清单，或从已审核的本地清单生成后随代码版本固定。

### 3.3 场景组规则

同一场景组是不可拆分单元：

| 来源 | 分组规则 |
|---|---|
| PAN/CCD 产品图 | 相同 L1A 产品号和同源裁片归为一组 |
| MAR20 | 文件序列规则 + 强近重复图像连通分量 |
| 坐标命名图 | 相同原始产品/经纬度场景归组 |
| 官方 FSC | 归一化经纬度，忽略 Google/Bing/Yandex 提供商差异 |
| 数字新增船图 | 正则取前三段，如 `a_b_c_*` |
| `P*.bmp` | 去除 `(2)` 等后缀后归组，再做感知近邻聚类 |
| AU/RU 发射车 | 使用 `AUAU01`、`RURU01` 等前 6 字符序列族 |

在文件名分组之外，再执行：

- 图像 SHA-256 完全重复检查；
- dHash/pHash 强近重复检查；
- DINOv3 embedding 近邻复核；
- 标签几何和类别边界检查。

### 3.4 划分策略

正式划分规则：

- 官方数据按场景组执行约 `70/15/15` 的 `TRAIN/VAL/TEST`；
- 新增数据全部只进入 `TRAIN`；
- `TEST` 一旦生成立即锁定；
- 划分算法只使用标签、来源和场景属性，不使用任何模型结果；
- 固定随机种子 `20260810`；
- 输出唯一数据指纹。

官方场景组联合平衡以下属性：

- 25 个细类；
- 船/飞机/车辆 3 个粗组；
- small/medium/large；
- crowded；
- edge/截断目标；
- 空背景；
- 官方来源家族。

不能只按图片数量随机分配。旧 V1 的 small 比例约为 `TRAIN 1.43%`、`VAL/TEST 约 4%`，crowded 比例约为 `TRAIN 75.8%`、`VAL/TEST 约 89%`；V3 必须在场景隔离前提下缩小这种分布差异。

### 3.5 数据 Gate D0

以下条件全部通过才允许训练：

```text
official images                    = 4481
added images in VAL/TEST           = 0
missing image/label pairs          = 0
invalid YOLO rows                  = 0
scene groups crossing splits       = 0
exact duplicates crossing splits  = 0
strong near-duplicate leaks        = 0 after review
unreviewed empty added labels      = 0
dataset fingerprint                = generated and immutable
```

分布软目标：

- small 比例在各集合间绝对差尽量不超过 1.5 个百分点；超过 3 个百分点必须重划或书面解释；
- crowded/edge 比例绝对差尽量不超过 5 个百分点；超过 8 个百分点必须重划或解释；
- 稀有细类在 `VAL/TEST` 中尽可能保证可评价实例数；无法满足时必须报告置信区间，不能用单个百分比作强结论。

V3 冻结输出：

```text
artifacts/scene811_v3_grouped_clean_r10/split/
├── images/{train,val,test}
├── labels/{train,val,test}
├── dataset.yaml
├── split_manifest.csv
├── source_manifest.csv
├── patch_manifest.csv
└── dataset_fingerprint.json
```

---

## 4. 为什么旧权重和旧缓存不能直接复用

旧基线、旧 OOF 模型和旧 KD 缓存可能见过 V3 新 `VAL/TEST` 中的图片或同源场景，因此：

- 旧 `best.pt` 可继续做本地演示和历史参考；
- 旧 `best.pt` 不得作为 V3 证据模型的初始化；
- V3 学生统一从官方 `yolo11m.pt` 初始化；
- DINOv3 SAT-493M 通用预训练权重可复用；
- 旧教师 RoI 头、教师缓存、OOF 结果、hard manifest 全部失效；
- 新缓存必须绑定数据指纹、教师 SHA、分辨率、变换和代码提交 SHA。

这样可以避免“新验证集实际已经被旧模型训练过”的隐性泄漏。

---

## 5. 新模型方案：FHiT-KD v2

### 5.1 总体结构

```text
                         ┌───────────────────────────────┐
原始/弱增强图像 ────────>│ 冻结 DINOv3 ViT-L/16 SAT-493M │
                         └──────────────┬────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
             全局中间层 token                        高分辨率局部目标裁片
                    │                                       │
        P4/P5 correlation distill              OOF prototype / background contrast
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │ 失效、尺寸、可靠性路由
强增强图像 ────────────────────────────>│
                                ┌───────▼────────┐
                                │    YOLO11m     │
                                │ P3 / P4 / P5  │
                                └───────┬────────┘
                                        │
                                best_deploy.pt
                                  纯 YOLO11m
```

DINOv3、adapter、prototype bank 和可选定位教师只在训练阶段使用，部署参数量和推理结构仍是纯 YOLO11m。

### 5.2 分支 G：全局相关性蒸馏

旧方案将 DINO 多层 token 一次融合，再通过插值/池化生成 P3/P4/P5，并做归一化 MSE。问题是 DINOv3 ViT-L/16 的空间步长仍为 16，插值不能创造真实 stride-8 细节。

新方案：

- 保留 DINO 中间层层级信息，不先全部压成一个融合图；
- 全局分支只监督 YOLO 的 P4/P5 或对应中高层语义；
- 采用 Pearson/correlation loss，减弱异构网络特征幅值差异；
- 使用 foreground/context/hard-background mask；
- 中大目标、复杂场景上下文和低置信目标优先；
- 不再声称上采样 DINO token 是“真实 P3 教师”。

该分支主要保持飞机优势，并改善中大船语义和复杂背景稳定性。

### 5.3 分支 P：局部目标与背景原型蒸馏

这是 V2 的核心创新，也是解决小车辆和相邻船的主分支。

流程：

1. 从原始分辨率图像裁出 GT 目标及上下文；
2. small/medium 目标裁片放大后单独送入 DINOv3；
3. 按 `细类 + 粗类 + 尺寸` 建立 prototype bank；
4. prototype 使用 3-fold leave-one-scene-out 构建：当前场景的目标不能参与其教师原型；
5. 学生 P3/P4 RoI 学习与正确原型接近、与混淆原型远离；
6. 车辆额外使用人工确认的背景 prototype，显式区分码头、建筑、道路和车辆；
7. 只对教师 margin 足够、弱视图一致、类别与 GT 兼容的目标启用。

这比过度自信的原始 25 类 KL 更稳健，因为教师输出从“单次高置信分类”改为“跨场景原型关系”。

**禁止把未标注预测框自动当作背景。** 车辆背景 prototype 必须来自 OOF 假阳性，并由人工确认该区域确实没有目标。

### 5.4 分支 L：可选定位蒸馏

DINOv3 不是框分布教师。要提升 mAP50-95，定位信号必须单独处理。

候选定位教师：

```text
YOLO11l/x，imgsz=1024，仅训练阶段使用
```

只有满足以下条件才启用：

- 教师在官方 `VAL` 的高 IoU AP 明显优于 YOLO11m；
- 船和飞机的匹配框 IoU 分布更好；
- small/crowded 子集没有明显退化；
- 教师框与 GT 足够一致；
- 单独 L 短筛通过 Gate。

启用时只蒸馏可靠正样本的 DFL/框分布，不蒸馏教师背景噪声。若教师不具备增量定位知识，该分支直接取消，不因“模型更大”而强行加入。

### 5.5 失效路由

每个辅助信号不是对所有目标同时施加：

| 目标状态 | 主要路由 |
|---|---|
| 已稳定检测的飞机 | 0 或极弱 G，防止破坏饱和精度 |
| 中大船、上下文复杂 | G |
| small vehicle | P |
| 密集/相邻船车 | P，必要时 L |
| low confidence / no candidate | P 或 G，按尺寸选择 |
| localization error | L；没有合格定位教师时只用检测损失 |
| background vehicle FP | 人工确认背景 prototype / hard-negative 数据变量 |
| teacher 错误或不一致 | 跳过 KD |

OOF 失效图谱只在 `TRAIN` 内通过场景分组折模型生成，不能用同一训练模型诊断自己的训练图。

---

## 6. 损失、冲突控制和训练日程

### 6.1 总损失

\[
L = L_{YOLO}
  + \alpha(t)L_{global\_corr}
  + \beta(t)L_{local\_prototype}
  + \gamma(t)L_{localization}
\]

这不是固定权重的永久线性相加：

- `L_YOLO` 每个 batch 都计算；
- G/P/L 按目标失效类型和教师可靠性稀疏启用；
- 分支权重由共享梯度预算约束；
- 辅助梯度与检测梯度 cosine `< -0.1` 时执行投影或跳过；
- KD 数值使用 FP32，整体训练可 AMP；
- KD 模式固定 `compile=False`。

建议最大共享梯度预算：

| 分支 | 相对检测共享梯度上限 |
|---|---:|
| G 全局相关性 | 3% |
| P 局部原型 | 1% |
| L 定位 | 5% |

这些值是上限，不是必须用满的固定权重。

### 6.2 120 轮正式训练日程

| Epoch | 分辨率 | KD 日程 |
|---:|---:|---|
| 1–5 | 768 | 仅检测，稳定学生 |
| 6–20 | 768 | G/P/L 线性渐入 |
| 21–90 | 768 | 通过路由和可靠性门控启用 |
| 91–110 | 768 | KD 逐步衰减 |
| 111–120 | 1024 | 仅检测，高分辨率定位收尾 |

1024 收尾必须同时用于 C0 和 KD 候选，不能只给 KD 使用。

### 6.3 初始训练参数

```yaml
student: YOLO11m
initial_weights: yolo11m.pt
epochs: 120
imgsz_main: 768
imgsz_final: 1024
optimizer: AdamW
student_lr0: 0.001
adapter_lr0: 0.0001
weight_decay: 0.0005
amp: true
compile: false
kd_math: fp32
effective_batch: 48
seeds: [42, 3407, 20260809]
```

24 GB GPU 从 `micro_batch=8, accumulate=6` 起步。通过显存健康测试后可增大 micro-batch，但所有对照组必须保持相同 effective batch 和学习率缩放。

### 6.4 数据增强

检测分支：

- D4 旋转/翻转，优先精确 90° 旋转；
- translate、scale；
- 适度亮度/对比度/色彩扰动；
- Mosaic 只在前 60% 训练使用；
- 最后 20% 关闭 Mosaic；
- MixUp 默认关闭；
- 增加滑窗边缘截断和平移增强。

KD 分支使用可追踪的弱增强或独立弱视图。不能为了缓存对齐，像旧 `train_ablation.py` 那样把 C0/F/K/FK 的所有增强全部关闭。

### 6.5 来源感知采样

每个 batch 的目标比例：

```text
official ≈ 70%
added    ≈ 30%
```

并满足：

- 同一 AU/RU 或同源船舶序列每个 batch 最多 1 张；
- 船/飞机/车辆按 inverse-sqrt 或 effective-number 限幅采样；
- 不通过简单复制让 FSC 序列支配训练；
- C0 和所有 KD 对照使用完全相同的采样清单。

---

## 7. 实验顺序与 Gate

### 7.1 阶段 A：数据收益归因

先跑两个完全匹配的纯 YOLO11m：

| 实验 | 训练数据 | 目的 |
|---|---|---|
| B-official | 官方 TRAIN | 官方数据基准 |
| B-mix | 官方 TRAIN + 新增 TRAIN | 判断新增数据是否真正有益 |

初始化、增强、轮数、seed、采样预算必须一致。

选择规则：

- B-mix 在纯官方 `VAL` 上稳定提高船/车或总分，才作为新主数据配方；
- 若 B-mix 只提高训练拟合，却增加官方车辆 FP，则降低新增占比或按序列限额；
- 不允许使用 `TEST` 选择数据配比。

### 7.2 阶段 B：新 C0 与 OOF 错误图谱

在选定数据配方上训练新的 C0，然后用 3-fold 场景隔离 OOF 输出：

```text
low_confidence
no_candidate
localization
nms_suppressed
wrong_group
background_fp
size / crowded / edge
```

错误类型决定下一模块，而不是先把所有损失堆在一起。

### 7.3 阶段 C：教师资格审查

DINOv3 语义教师必须报告：

- scene-grouped OOF 可靠性；
- ship/vehicle/small/crowded/edge 子集；
- target prototype margin；
- 弱视图一致性；
- 与 C0 错误的互补性。

定位教师必须报告高 IoU AP 和匹配框 IoU 分布。教师在目标子集没有增量信息时，对应分支不得训练。

### 7.4 阶段 D：接线与续训健康检查

正式短筛前必须通过：

```text
cache miss = 0
enabled branch loss > 0
corresponding shared gradient > 0
adapter/prototype head gradient > 0
NaN/Inf = 0
health check optimizer steps = 0
best_deploy detector SHA == full checkpoint detector SHA
continuous run == interrupted/resumed run within tolerance
```

checkpoint 必须保存：

- optimizer；
- EMA；
- scheduler；
- GradScaler；
- epoch/global step；
- RNG；
- sampler state；
- KD 权重 EMA/计数；
- prototype/cache fingerprint；
- 数据指纹和代码 SHA。

### 7.5 阶段 E：单变量短筛

从同一学生 checkpoint 开始，每组先跑 12–15 轮：

| 实验 | 启用模块 |
|---|---|
| C0-ft | 无 KD，匹配微调对照 |
| G | 仅全局 correlation |
| P | 仅局部 prototype |
| GP | 仅当 G/P 至少一个通过时运行 |
| L | 仅合格定位教师 |
| GPL | 仅当 GP 和 L 单独有效时运行 |
| C0+HN | 仅人工确认困难背景 |
| GP+HN 或 GPL+HN | 分离 KD 收益与数据收益 |

MGD 只保留为论文/答辩对照，不作为默认主线。

短筛 Gate：

- mAP50-95 相对匹配 C0 下降不得超过 0.2 个百分点；
- aircraft F1 下降不得超过 0.2 个百分点；
- ship 或 vehicle 至少一个提升 0.5 个百分点；
- 精确赛方评分必须提高；
- 所有 KD 健康与部署等价检查必须通过；
- seed 42 通过后才跑 seed 3407。

### 7.6 阶段 F：正式长训

只保留：

```text
1 个匹配 C0
1 个晋级 KD 候选
```

使用配对种子：

```text
42, 3407, 20260809
```

正式晋级条件：

- 3 seed 平均 mAP50-95 不低于 C0，目标提升至少 0.3 个百分点；
- 至少 2/3 seed 优于对应 C0；
- 精确赛方评分提高；
- aircraft F1 平均下降不超过 0.2 个百分点；
- ship 或 vehicle F1 至少提升 1 个百分点，或二者都提升至少 0.5 个百分点；
- paired bootstrap 置信区间支持结论；
- 不以单 seed 的最好 epoch 宣称蒸馏有效。

---

## 8. 比赛评估规范

必须同时报告：

1. 25 类 native `Precision / Recall / mAP50 / mAP50-95`；
2. 赛方船/飞机/车辆三粗类指标与总评分。

赛方评估固定：

```text
class-aware coarse matching = true
ship IoU                   = 0.50
aircraft IoU               = 0.50
vehicle IoU                = 0.35
one-to-one matching        = true
prediction cache floor     = 0.01
max_det                    >= 3000
```

不能固定 `confidence=0.6` 后下结论。应在 `VAL` 上扫描阈值，并按比赛规则冻结全局或分组阈值；若比赛不允许分组阈值，则只冻结一个全局阈值。

最低安全门：

```text
overall Recall >= 0.85
overall FDR    <= 0.20
```

同时增加船/车非劣约束，防止数量占优的飞机掩盖弱组退化。

`TEST` 只能在数据配方、模型、epoch、阈值和 NMS 全部冻结后使用一次。

---

## 9. 10000×10000 大图方案

大图阶段不改变学生模型，只改变推理管线，因此不会牺牲普通小图的直接推理能力。

小图：

```text
直接 YOLO11m 推理
```

10000×10000 大图：

```text
候选 A：tile=1024, stride=768   （25% overlap）
候选 B：tile=1280, stride=960   （25% overlap）
必要时：tile=1024, stride=640
```

合并要求：

- 保存 tile 来源和全局坐标；
- 对滑窗人工边缘使用低权重，而不是删除真实图像边缘目标；
- 只合并跨重叠 tile 的同一目标；
- 防止 NMS/WBF 将相邻船车错误合并；
- 对船、飞机、车辆分别验证合并阈值；
- 记录读图、切窗、推理、合并和写结果的完整时间；
- 在赛方指定硬件上满足 10000×10000 单图时间限制。

pseudo-large 只能验证坐标与合并代码，不能证明真实大图精度。最终应使用有标注真实大图，或由可追溯裁片坐标重建的场景组做验证。

---

## 10. 服务器、上传文件与预计耗时

### 10.1 服务器配置

| 项目 | 最低可运行 | 建议完整实验 |
|---|---:|---:|
| GPU | 24 GB RTX 3090/4090/4090D | RTX 4090/4090D 24 GB |
| CPU | 8 核 | 16 核以上 |
| RAM | 32 GB | 64 GB |
| 可用磁盘 | 80 GB，需逐阶段清理 | 150–200 GB |
| Python | 3.10/3.11 | 3.11 |
| 系统 | Ubuntu 22.04 | Ubuntu 22.04/24.04 |

100 GB 总盘可以做紧凑短筛，但必须只保留 `best/last/deploy` 和必要缓存。完整 3-fold OOF、DINO 缓存、多种子长训与大图报告建议至少 150 GB 可用空间。

### 10.2 必须手动上传

```text
/root/rsdet/input/scene811_latest_20260810.zip
/root/rsdet/input/official_4481_manifest.csv
/root/rsdet/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
/root/rsdet/weights/yolo11m.pt
```

可选：

```text
/root/rsdet/large/*.jpg
```

不上传旧教师缓存、旧 OOF、旧 hard manifest 和旧 KD 中间权重。

### 10.3 克隆和环境检查（当前仓库已支持）

```bash
cd /root
git clone --recurse-submodules https://github.com/zhengzhezhao057/fhit-kd-yolo11m.git
cd /root/fhit-kd-yolo11m
bash scripts/00_server_bootstrap.sh

PY=/root/miniconda3/envs/fhit-kd/bin/python
```

校验输入 ZIP：

```bash
sha256sum /root/rsdet/input/scene811_latest_20260810.zip
# 必须得到：
# f66212d1693baa92c6342ddac003775671a9c99e38fb6d26eee2cacd28d63bc5
```

### 10.4 目标执行接口（尚未合入当前 `main`，不可提前执行）

数据冻结器合入后：

```bash
$PY -m src.freeze_scene811_v3 \
  --archive /root/rsdet/input/scene811_latest_20260810.zip \
  --official-manifest /root/rsdet/input/official_4481_manifest.csv \
  --out artifacts/scene811_v3_grouped_clean_r10/split \
  --seed 20260810 \
  --added-policy train-only

$PY -m src.audit_dataset \
  --dataset-root artifacts/scene811_v3_grouped_clean_r10/split \
  --strict
```

v2 训练模块合入后，目标入口统一为：

```bash
# 数据配方基线
$PY -m src.run_v2_stage --config configs/v2/b_official.yaml
$PY -m src.run_v2_stage --config configs/v2/b_mix.yaml

# 接线健康检查
$PY -m src.kd_v2_health --config configs/v2/gp.yaml --batches 10

# 单变量短筛
$PY -m src.run_v2_matrix --matrix configs/v2/short_screen.yaml

# 正式长训（同一命令重新执行应精确续训）
$PY -m src.run_v2_stage --config configs/v2/final_c0_seed42.yaml
$PY -m src.run_v2_stage --config configs/v2/final_gp_seed42.yaml

# 汇总与Gate
$PY -m src.evaluate_v2 --matrix configs/v2/final_matrix.yaml
$PY -m src.gate_v2 --report reports/scene811_v3_grouped_clean_r10/final/summary.json
```

这些命令是下一批代码交付的接口合同，不代表当前 GitHub `main` 已经实现。当前仓库仍硬编码旧 `scene811_v2/6699` 的部分路径和数量；在这些接口完成、测试和推送前，不应启动最新数据的正式训练。

### 10.5 4090D 粗略耗时

以下为单卡估算，最终以 50 batch 基准测试校准：

| 阶段 | 预计耗时 |
|---|---:|
| 数据构建与审计 | 0.5–2 小时，主要为 CPU/磁盘 |
| 单个 30 轮数据配方筛选 | 1–2.5 小时 |
| 3-fold OOF，每 fold 30 轮 | 4–8 小时总计 |
| DINO 全局/局部信号构建 | 2–5 小时 |
| 单个 12–15 轮 KD 短筛 | 1–3 小时 |
| 单个 120 轮 C0 | 4–8 小时 |
| 单个 120 轮 KD | 8–16 小时 |
| 三种子 C0 + KD 正式对比 | 36–72 GPU 小时 |

第一轮不要一次租满 72 小时。应按 D0、健康检查、短筛 Gate 逐段续租，失败模块立即停止。

---

## 11. 当前代码能力与必须开发的部分

### 11.1 当前 GitHub 已有

- 服务器环境检查；
- 基础数据审计、场景分组、OOF；
- 旧 Global F/K/FK；
- FAH 失效权重与 25→3 层级 KL；
- 梯度预算、KD 健康检查、部署权重导出；
- optimizer/EMA/scaler/scheduler 续训；
- 赛方评估和滑窗推理。

### 11.2 V2 必须实现并测试

- 最新 ZIP 的 V3 数据冻结与补丁清单；
- 真正的来源感知 batch sampler；
- DINO 中间层独立 adapter 与 P4/P5 correlation loss；
- 高分辨率局部目标裁片；
- leave-one-scene-out prototype bank；
- 人工确认背景 prototype；
- margin/弱视图一致性门控；
- 可选定位教师与 DFL/框分布蒸馏；
- 冲突梯度投影/跳过；
- 强检测视图与弱 KD 视图的双流训练；
- 768→1024 的匹配两阶段调度；
- paired bootstrap 与最终 Gate；
- `competition_eval` 的 `batch/max_det` 固化；
- 完整断点续训一致性测试。

每个模块必须有单元测试和最小健康训练，不能把研究设想直接写入配置后开始长训。

---

## 12. 结果出来后的决策树

| 结果 | 下一步 |
|---|---|
| D0 泄漏或标签检查失败 | 停止训练，修数据清单 |
| B-mix 不如 B-official | 降低新增采样或退回 official 配方 |
| G 有效、P 无效 | 保留 G，检查局部 crop/prototype 质量 |
| P 有效、G 无效 | 以 P 为主；G 只留作消融 |
| G/P 都无效 | 不组合 GP，先查教师互补性和路由 |
| L 教师高 IoU 不优 | 删除定位分支，用匹配高分辨率收尾 |
| replay 有效但 KD 无效 | 收益归因于数据课程，不归因于 DINOv3 |
| vehicle recall 高但 FDR 高 | 优先人工背景 prototype/阈值，不盲目加 KL |
| ship low-confidence 多 | 优先局部 P 与高分辨率 |
| ship localization 多 | 评估 L 或定位收尾 |
| nms_suppressed 多 | 先调推理 NMS/合并，不修改训练损失 |
| aircraft 下降 | 降低/关闭已稳定飞机 KD 路由 |
| 单 seed 提升 | 只算候选，必须补 2 个配对 seed |
| 三 seed 通过正式 Gate | 冻结模型、阈值和大图参数 |

---

## 13. 最终交付物

```text
best_deploy.pt
frozen_inference.json
dataset_fingerprint.json
source_manifest.csv
split_manifest.csv
patch_manifest.csv
training_config.resolved.yaml
checkpoint_provenance.json
kd_health.jsonl
native_metrics.json
competition_metrics.json
per_class_metrics.csv
paired_bootstrap.json
large_image_benchmark.json
gate_decision.json
final_report.md / final_report.docx
```

报告必须明确区分：

- 数据收益；
- DINOv3 蒸馏收益；
- 定位教师收益；
- hard-negative 收益；
- 滑窗后处理收益。

任何一个模块没有通过单变量对照，就不能把最终模型的提升归因给它。

---

## 14. 禁止事项

- 不直接使用最新 ZIP 自带划分长训；
- 不把新增数据放入 `VAL/TEST`；
- 不按图片随机拆分同源裁片；
- 不用旧 baseline 初始化 V3 证据实验；
- 不复用旧教师缓存或旧 OOF；
- 不把 DINO stride-16 插值称为真实 P3；
- 不继续把过度自信的原始 25 类 KL 当主方案；
- 不把 F、K、MGD、定位教师、背景头、高分辨率一次性全部叠加；
- 不把未标注预测框自动当作背景；
- 不固定 `confidence=0.6` 或使用 `class_aware=False` 作结论；
- 不用 pseudo-large 高 Recall 宣称真实大图已解决；
- 不用单 seed 最佳 epoch 宣称蒸馏有效；
- 不在 KD 模式启用 `torch.compile`；
- 不在已完成的短训 checkpoint 上用 `--resume --epochs` 假装延长目标轮数；正式长训必须在启动前固定完整日程和新 run-name。

---

## 15. 立即执行顺序

当前只做以下五件事：

1. 保留最新 ZIP，记录 SHA-256；
2. 生成/核对 `official_4481_manifest.csv`；
3. 完成并测试 V3 数据冻结器，输出 D0 审计；
4. 参数化仓库中旧 `scene811_v2/6699` 的硬编码路径与数量；
5. 实现并分别测试 G、P；在 B-official/B-mix 结果出来前不开发复杂 GPL 长训矩阵。

**第一次真正启动 GPU 的任务应该是 B-official 与 B-mix 匹配基线，而不是 FK。**

---

## 16. 方法依据

- [DINOv3](https://arxiv.org/abs/2508.10104)：冻结视觉基础模型语义表征。
- [PKD](https://proceedings.neurips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html)：异构检测器的 Pearson correlation 蒸馏。
- [FGD](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)：前景与全局上下文蒸馏。
- [ScaleKD](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html)：尺度感知知识迁移。
- [Localization Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)：框分布定位蒸馏。
- [CrossKD](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)：降低分类/定位目标冲突。
- [PCGrad](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)：冲突梯度处理。
- [SAHI](https://arxiv.org/abs/2202.06934)：超大图切片推理与小目标检测。
