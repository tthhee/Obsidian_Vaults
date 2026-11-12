---
tags:
  - 自动驾驶
  - 在线地图
  - BEVLane
  - 网络架构
type: note
created: 2026-11-11
aliases:
  - BEVLane架构
  - BEVLane network architecture
---

# BEVLane 网络架构

> 基于 `Work/bevlane/` 目录下的解析文档整理。BEVLane 是一个**单帧纯视觉在线地图矢量检测模型**（online map perception），是 [[BEVLanePlus 网络架构]] 的前身——采用「图像 backbone/FPN → LSS → BEV encoder → MapQR」的经典路线，输出折线 + 属性 + 拓扑。

## 一、一句话定位

BEVLane = **单帧纯视觉**矢量检测器：2 路前视图像经 LSS 转 BEV，用 30 个 map query + 4 个 junction query 在 2 层 decoder 中解码，输出折线坐标、细分类属性与拓扑关系。

## 二、总体数据流（推理）

```text
2 路前视图像（480×960）
    │  ResNet18 + CustomFPN（stage 3/4，256 通道）
    ▼
LSS 视图变换（深度 1–165m，64 通道）
    │  Voxel Pooling（cumsum）
    ▼
static BEV encoder → BEV feature（128 通道）
    │
    ├─ 30 map query + 4 junction query（nn.Embedding 可学习）
    ▼
2 层 MapQRSparseDecoder（self-attn → deformable cross-attn → FFN）
    │
    ├─ cls_branch（属性分类 3×60）
    ├─ inst_cls_branch（实例分类 6）
    ├─ reg_branch（点坐标 20×2）
    ├─ changed_reg_branch（变化点 2）
    └─ topo_branch（拓扑 4，仅末层）
    ▼
折线 + 属性 + 拓扑
```

## 三、模块划分

| 模块 | 核心组件 | 作用 |
| --- | --- | --- |
| 图像编码 | ResNet18 + CustomFPN | 提取多尺度图像特征（256 通道） |
| 视图变换 | LSSViewTransformer | 相机 → BEV，深度 1–165m |
| BEV 编码 | BEV encoder | 生成 128 通道 BEV feature |
| 检测头 | BEVLaneHeadMapQRV2MTK | MapQR 解码 + 多分支输出 |
| Decoder | MapQRSparseDecoder | 2 层 self/cross attention |
| 匹配 | MapTRAssigner | 匈牙利匹配（cls + Chamfer 距离） |

## 四、分模块详解

### 4.1 图像分支（ResNet18 + FPN + LSS）

- **Backbone**：ResNet18，读取 stage 3、4 特征；FPN（CustomFPN）上采样融合输出 256 通道。
- **相机**：2 路前视（fov120、fov30），输入 480×960×3。
- **LSS**：`LSSViewTransformer` 用 `depth_net` 预测深度分布，`Voxel Pooling`（cumsum）将 frustum 点投影到 ego 坐标系生成 BEV。
- **BEV 范围**：x=[0,153.6]，y=[-25.6,25.6]（grid 128×256，间隔 1.2×0.2m）。

### 4.2 Query 与 Decoder

- **Query**：30 个 map instance query（`instance_embedding [30,128]`）+ 4 个 junction query（`junction_embedding [4,128]`），均用 `nn.Embedding` 可学习；每个 instance 20 个采样点（`pts_embedding [600,128]` / `[80,128]`）。
- **Decoder**：2 层 `MapQRSparseDecoder`，每层 self-attention → deformable cross-attention（BEV 作 value）→ FFN；`BEVRefinementModuleOE` 用 reference points 逐步 refine。
- **多层分支共享**：每层都有各自的 cls/inst/reg/changed/topo branch，但不同层的参数共享。

### 4.3 检测头多分支输出

head 的 `forward` 输出按 decoder 层组织的 list，主要字段：

| 字段 | shape（2 层 × batch 36） | 含义 |
| --- | --- | --- |
| `all_pts_cls_scores` | `[2,36,30,3,60]` | 属性分类（3 slot × 60 通道） |
| `all_inst_cls_scores` | `[2,36,30,6]` | 实例分类（5 类 + 背景） |
| `all_pts_preds` | `[2,36,30,20,2]` | 点坐标回归 |
| `all_changed_preds` | `[2,36,30,2]` | 属性变化点坐标 |
| `all_topology_preds` | `[36,30,30,4]` | 拓扑关系（仅末层） |
| junction 系列 | `[2,36,4,...]` | junction 独立分支 |

### 4.4 属性分类分支（cls_branch）

`cls_branch` 预测**每条线的细粒度属性**，每条 query 输出 `3 × 60 = 180` logits，拆为四组：

| 属性 | 类别数 | 通道 slice |
| --- | --- | --- |
| 车道边界类型 lane_bound | 26 | [0:26] |
| 车道边界颜色 color | 10 | [26:36] |
| 道路边界类型 road_bound | 16 | [36:52] |
| 鱼骨线类型 fishbone | 8 | [52:60] |

- `max_change_num=2`：一条线最多 2 个属性变化点，分成 3 个沿线 slot（不是「预测 3 类」）。
- **独立分支**：road mark（21 类）、实例类别（6 类）、变化坐标（2 通道 sigmoid）各自独立，不占 60 通道。

### 4.5 匹配与 Loss

- **匹配**：`MapTRAssigner` 匈牙利匹配，代价 = 分类代价（focal）+ Chamfer 点距离代价，对 `num_shifts` 个起点版本取最小距离（记录最优 shift）。
- **损失**：`loss()` 编排层 → `loss_single` 逐层算 → `get_targets`/`_get_target_single` 生成 target。主要损失：实例分类（focal）、点回归（PtsL1）、方向、属性分类（sigmoid focal）、变化点、拓扑（cls+dist，仅末层）、junction 系列。
- **门控**：`ldmap_gt_flag`（0=普通线/1=junction/2=guideline）与 `is_train_bev_static` 控制各分支学哪些样本。

## 五、与 BEVLanePlus 的关键差异

| 维度 | BEVLane | BEVLanePlus |
| --- | --- | --- |
| Backbone | ResNet18 | ResNet34 |
| 相机 | 2 前视 | 2 前视 + 2 鱼眼 |
| 模态 | Camera | Camera + LiDAR + 地图先验 |
| Query | 30 map + 4 junction | 241 分类别 anchor |
| Decoder | 2 层单帧 | 3 层稀疏时序 + InstanceBank |
| 输出 | 折线 + 属性 + 拓扑 | geometry + subtype + topology + ID |

详见 [[bevlaneplus_vs_bevlane_corrected]]。

## 六、相关笔记

- [[Loss解析]] — loss 编排、匹配、target 构造全流程
- [[cls_branch_attribute_analysis]] — 属性分类分支的完整调查
- [[bevlaneplus_vs_bevlane_corrected]] — 与 BEVLanePlus 的差异对比
- [[BEVLanePlus 网络架构]] — 升级版架构
- 图：[[bevlane]]
