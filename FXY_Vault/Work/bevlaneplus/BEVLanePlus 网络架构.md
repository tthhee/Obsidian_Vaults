---
tags:
  - 自动驾驶
  - 在线地图
  - BEVLanePlus
  - 网络架构
type: note
created: 2026-11-11
aliases:
  - BEVLanePlus架构
  - BEVLanePlus network architecture
---

# BEVLanePlus 网络架构

> 基于 `Work/bevlaneplus/` 目录下的解析文档整理。BEVLanePlus 是一个**自动驾驶在线地图感知模型**（online map perception），从单帧纯视觉矢量检测器（[[BEVLane]]）升级为**多相机 + 多模态 + 地图先验驱动 + 跨帧实例跟踪**的结构化在线建图模型。

## 一、一句话定位duoku

BEVLanePlus 延续 BEVLane 的「图像 backbone/FPN → LSS → BEV encoder → MapQR」基础路线，但升级为：

> **多相机、多模态 BEV 融合 + Nav/SD 与 LDLite 双地图先验 + 分类别 sparse query + 时序实例记忆 + 结构化拓扑输出** 的在线建图模型。

## 二、总体数据流（推理）

> 数据流分四路并行汇聚，最终进入稀疏时序 decoder 输出结构化地图。按「**感知 → 地图先验 → 时序 → 解码输出**」四阶段展开。

### 2.1 全景图（彩色框架示意图）

![[BEVLanePlus 数据流全景图.svg|639]]

四个阶段用不同色块区分，箭头表示数据流向：

| 阶段 | 色块 | 内容 |
| --- | --- | --- |
| ① 传感器感知 | 蓝 | 前视+鱼眼图像、LiDAR 点云 → camera/LiDAR BEV → ConvFuser |
| ② 地图先验 | 绿 | LDLite 栅格先验 + Nav/SD 矢量先验 → LDLite Fusion |
| ③ 时序记忆 | 紫 | 上一帧实例 → InstanceBank |
| ④ 稀疏解码 | 粉 | 241 anchor → 3 层 decoder → 结构化输出 |

> 说明：三条支流（sensor BEV、地图先验、时序实例）在 decoder 处汇聚；LDLite 先验先通过 deformable cross-attention 注入 sensor BEV，Nav/SD 先验通过 `cross_attn_nav` 进入 decoder，上一帧实例通过 temporal attention 进入 decoder。

### 2.2 线性数据流

```text
┌─ 感知 ─────────────────────────────────────────────┐
│ 前视+鱼眼图像 ──双分支──▶ camera BEV ─┐           │
│ LiDAR ──PillarFeatureNet/DenseResNet─▶ LiDAR BEV ─┤→ ConvFuser ─▶ sensor BEV
└───────────────────────────────────────────────────┘
                                                          │
┌─ 地图先验 ───────────────────────────────────────────┐  │
│ LDLite 属性 ──LDLiteEncoder──▶ LDLite BEV ──────────▶│  │
│ Nav/SD polyline ──MapGraphTransformer──▶ nav feature │  │
└──────────────────────────────────────────────────────┘  │
                                                          ▼
                              deformable cross-attn 融合 ─▶ fused BEV
                                                          │
┌─ 时序 ──────────────────────────────┐                  │
│ 上一帧实例 ──ego-motion──▶ InstanceBank│────────────────┤
└─────────────────────────────────────┘                  ▼
                                             241 分类别 sparse anchors
                                                          │
                                                          ▼
                                   3-layer sparse temporal decoder
                                        + 分阶段 refinement
                                                          │
                                                          ▼
                         geometry + subtype + topology + instance ID
```

## 三、模块划分

| 模块 | 核心组件 | 作用 |
| --- | --- | --- |
| 图像编码 | ResNet34 + CustomFPN | 提取多尺度图像特征（256 通道） |
| 视图变换 | LSSViewTransformer（LSS） | 相机 → BEV，深度 1–165m，64 通道 |
| LiDAR 编码 | PillarFeatureNet + DenseResNet | 点云 → 128 通道 LiDAR BEV |
| 模态融合 | ConvFuser | camera BEV + LiDAR BEV → 128 通道 sensor BEV |
| 地图先验·矢量 | MapGraphTransformer | Nav/SD polyline → 128 维稀疏 token |
| 地图先验·栅格 | LDLiteEncoder + LDLiteFusionModule | LDLite 属性 → BEV 特征并注入 sensor BEV |
| 检测头 | MapQRSparseDecoder + BEVRefinementModuleOE | 分类别 sparse anchor 解码 |
| 时序 | InstanceBankMapOE | 跨帧实例传播、关联、一致性 |
| 训练辅助 | DN（denoising training） | 带噪 GT 稳定匹配 |

## 四、分模块详解

### 4.1 图像分支（Backbone + Neck + LSS）

- **Backbone**：ResNet34（BEVLane 为 ResNet18），读取 stage 3、4 特征。
- **Neck**：CustomFPN 输出 256 通道。
- **双分支**：high-resolution（前视 fov120/fov30）与 low-resolution（鱼眼）各一套 backbone/neck/view-transform/BEV-encoder，BEV 拼接后经 1×1 `bev_fuse_conv` 压回 128 通道。
- **LSS**：深度范围 1–165m，输出 64 通道；相机输入 352×704。

详见 [[PillarFeatureNet]]、[[one_model----map_branch(bevlaneplus)]]。

### 4.2 LiDAR 分支（PillarFeatureNet）

PointPillars 式编码：**体素化 → 特征增强 → MLP + max 池化**，输出每 pillar 64 维特征，再经 DenseResNet 生成 128 通道 LiDAR BEV。详见 [[PillarFeatureNet]]。

### 4.3 地图先验（双路径）

**矢量先验（Nav/SD）**：`MapGraphTransformer` 编码 nav path / SD road 为 128 维稀疏 token，decoder 通过 `cross_attn_nav` 使用。当前 `map_mode_info="only_nav"`。详见 [[map_encoder中 nav_path_instances，sd_road_instances, lane_nums的含义]]。

**栅格先验（LDLite）**：`LDLiteEncoder` 将 LDLite 属性栅格图（section/lane/semantic 共 17 通道）做 embedding 查表求和，得到 `(B,H,W,128)` 地图 BEV；`LDLiteFusionModule` 以 sensor BEV 为 query、LDLite BEV 为 value 做 deformable cross-attention 注入。详见 [[轻图融合 ldlite data encoder & fusion]]、[[ldlite encoder]]。

### 4.4 Query 与 Decoder

- **Query**：241 个分类别 sparse anchor，分 6 组 `[72, 27, 24, 63, 1, 54]`。
- **Decoder**：3 层 `MapQRSparseDecoder`，第 1 层单帧，第 2、3 层启用 temporal attention + navigation cross-attention，由 `BEVRefinementModuleOE` 分阶段更新几何、属性、拓扑。
- **输出**：geometry + 细分类 subtype + topology + instance ID（18 类 lane type、7 类 turn、18 类边界、10 类颜色、9 类 fishbone、20 类 roadmark 等）。

### 4.5 时序（InstanceBank）

`InstanceBankMapOE` 缓存上一帧实例 feature/anchor/confidence，按位姿传播到当前帧，合并更新历史实例，缓存当前高置信实例供下一帧，并计算跨帧一致性 loss。时序实例配额 `[24,9,8,21,0,0]` 共 62 个。

### 4.6 训练辅助（DN）

DN（Denoising）训练：4 个 DN groups + 3 个 temporal DN groups，坐标噪声 `[1.5,1.5]`，最多 24 个 DN GT。`_build_dn_targets` 将异构 GT 整理成「类别 + 展平点坐标 + 可选实例 ID」；进入 loss 前类别被压成**正负二分类**（详见 [[dn_targets_implementation_notes]]）。

## 五、与 BEVLane 的关键差异

| 维度 | BEVLane v035_1 | BEVLanePlus v006 |
| --- | --- | --- |
| Backbone | ResNet18 | ResNet34 |
| 相机 | 2 前视 | 2 前视 + 2 鱼眼 |
| 模态 | Camera | Camera + LiDAR + 地图先验 |
| BEV 范围 | x=[0,153.6]，y=[-12.8,12.8] | x=[0,153.6]，y=[-25.6,25.6] |
| Query | 30 map + 4 junction | 241 分类别 anchor |
| Decoder | 2 层单帧 | 3 层稀疏时序 + InstanceBank |
| 输出 | 折线 + 属性 | geometry + subtype + topology + ID |

详见 [[bevlaneplus_vs_bevlane_corrected]]。

## 六、相关笔记

- [[bevlaneplus_vs_bevlane_corrected]] — 与 BEVLane 的完整差异对比
- [[dn_targets_implementation_notes]] — DN 目标构造与类别说明
- [[PillarFeatureNet]] — LiDAR 点云编码
- [[ldlite encoder]] / [[轻图融合 ldlite data encoder & fusion]] — LDLite 地图先验
- [[map_encoder中 nav_path_instances，sd_road_instances, lane_nums的含义]] — Nav/SD 矢量先验
- 图：[[bevlaneplus]] / [[one_model----map_branch(bevlaneplus)]] / [[bevlaneplus-sparseod-diffusion-e2e]]
