---
tags:
  - MOC
  - 自动驾驶
  - 在线地图
type: moc
created: 2026-11-11
---

# BEVLanePlus 索引（MOC）

> BEVLanePlus 自动驾驶在线地图感知模型的文档地图。入口从 [[BEVLanePlus 网络架构]] 开始。

## 核心

- [[BEVLanePlus 网络架构]] — 网络架构总纲（数据流 + 模块划分 + 详解）

## 网络结构解析

- [[bevlaneplus_vs_bevlane_corrected]] — BEVLanePlus 与 BEVLane 的完整差异对比
- [[PillarFeatureNet]] — LiDAR 点云 Pillar 特征编码（体素化→特征增强→MLP+max 池化）
- [[map_encoder中 nav_path_instances，sd_road_instances, lane_nums的含义]] — Nav/SD 矢量先验（MapGraphTransformer）

## 地图先验（LDLite）

- [[ldlite encoder]] — LDLite 数据格式 + encoder 处理逻辑
- [[轻图融合 ldlite data encoder & fusion]] — LDLite 栅格化 + deformable attention 融合

## 训练辅助

- [[dn_targets_implementation_notes]] — DN 目标构造逻辑与类别说明

## 架构图（Excalidraw）

- [[bevlaneplus]] — 主模型架构图
- [[one_model----map_branch(bevlaneplus)]] — 地图分支结构图
- [[bevlaneplus-sparseod-diffusion-e2e]] — SparseOD + Diffusion 端到端图
