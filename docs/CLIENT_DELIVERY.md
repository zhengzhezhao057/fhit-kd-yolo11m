# 甲方交付说明：FHiT-KD / Scene811 V3-R10

## 交付物

- 可公开克隆的训练、评估、续训和证据记录代码；
- 防场景泄漏数据冻结器与独立验证器；
- DINOv3-SAT 教师、缓存、G/P/GP 蒸馏和纯 YOLO11m 部署导出；
- 比赛三粗类匹配、七项指标与 10000×10000 时限评估；
- 服务器运行手册、最终方案和论文证据计划。

主入口：

- `README.md`
- `docs/SERVER_RUNBOOK_SCENE811_V3.md`
- `docs/FINAL_TRAINING_PLAN_SCENE811_V3.md`
- `docs/PAPER_EVIDENCE_PLAN.md`
- `docs/SCENE811_V3_SPLIT_REPORT.md`

## 冻结数据

- ID：`scene811_v3_grouped_clean_r10`
- 指纹：`b4367981f59e0d04cf7925587582acb0d3f25a2e9b145dfef662a8da8f0797b9`
- 总计 8211 图；TRAIN/VAL/TEST=`6870/674/667`
- 官方 4481 图全部保留；新增 3730 图只进入 TRAIN；
- 192 对经 DINO 候选与 SIFT-RANSAC 几何证据确认的同场景图已绑定；
- 已知空标签新增图 1 张排除，重复标签行 1 条在派生视图修复；源 ZIP 不修改。

数据和权重因体积与授权不进入公开 GitHub。甲方需手动上传：

```text
scene811_v3_grouped_clean_r10.tar
yolo11m.pt
dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

## 验收边界

代码和数据门禁通过不等于模型已获奖，也不等于蒸馏一定优于 C0。正式结论必须来自服务器上的 `B-official/B-mix/C0/G/P/GP` 单变量结果、三固定种子、部署等价性和比赛七项指标。GP 不优于最佳单分支时，应采用 G 或 P。

旧 `scene811_v2`、6699 数据、F/K/FK 与 V4/V5 结果仅保留为历史问题证据，禁止复用其 teacher cache、checkpoint 或验证结论到 R10。
