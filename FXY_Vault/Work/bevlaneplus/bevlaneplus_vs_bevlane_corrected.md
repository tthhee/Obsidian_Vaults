# BEVLanePlus v006 与 BEVLane v035_1 实现差异（修正版）

## 1. 对比范围

本报告仅以以下代码为依据：

- BEVLanePlus：`/mnt/cfs-baidu/algorithm/xiangyu.fu/sparselane`
- Plus 配置：`projects/configs/bevlaneplus_v006/bsl.py`
- BEVLane：`/mnt/cfs-baidu/algorithm/xiangyu.fu/J6B/bevdet`
- BEVLane 配置：`configs/psd-bevlane-j6/v035_1_j6/bsl_aug.py`

结论来自配置及其实际引用的 detector、decoder、instance bank、地图编码器、数据 pipeline 和 loss，不是只比较配置字段。

## 2. 核心结论

BEVLanePlus 延续了 BEVLane 的“图像 backbone/FPN → LSS → BEV encoder → MapQR”基础路线，但已经从单帧纯视觉矢量检测器升级成多相机、多模态、地图先验驱动、可跨帧维护实例的结构化在线建图模型。

主要升级为：

1. ResNet18 升级为 ResNet34。
2. 2 个前视相机扩展为 2 个前视 + 2 个鱼眼相机。
3. Camera-only 升级为 Camera + LiDAR BEV 融合。
4. 横向范围从 ±12.8m 扩展为 ±25.6m。
5. 30 个共享 MapQR query + 4 个 junction query，升级为 241 个分类别 sparse anchor query。
6. 2 层单帧 decoder 升级为 3 层 sparse temporal decoder + InstanceBank。
7. 新增 DN、temporal DN、位姿传播、temporal attention 和跨帧一致性 loss。
8. 新增 Nav/SD 矢量先验与 LDLite 栅格先验两条地图先验路径。
9. 输出扩展为几何、细分类属性、拓扑关系和 instance ID。
10. 单帧 BEVMTK pipeline 重构为序列 PhigentMapDataset + Sparse4D adaptor。

## 3. 总览表

| 维度 | BEVLane v035_1 | BEVLanePlus v006 | 变化 |
|---|---|---|---|
| Detector | `SiameseBEVLANE` | `SiameseBEVLANE_FISHEYE` | 增加鱼眼、多模态、先验及时序链路 |
| Backbone | ResNet18 | ResNet34 | 图像特征能力增强 |
| 相机 | cam1、cam0 | cam1、cam0、cam21、cam23 | 增加左右鱼眼 |
| 输入尺寸 | 352×704 | 352×704 | 单相机尺寸不变 |
| 模态 | Camera | Camera + LiDAR + 地图先验 | 多源 BEV |
| BEV 范围 | x=[0,153.6]，y=[-12.8,12.8] | x=[0,153.6]，y=[-25.6,25.6] | 横向覆盖翻倍 |
| LSS grid | x=128，y=128 | x=128，y=256 | 横向栅格翻倍 |
| 静态采样 BEV | x=128，y=128 | x=192，y=256 | 总栅格数约 3 倍 |
| Decoder | `MapQRDecoder`，2 层 | `MapQRSparseDecoder`，3 层 | 稀疏、分阶段 refine、时序化 |
| Query | 30 map + 4 junction | 241 classified anchors | 按要素分配容量 |
| 时序实例 | 无 | 62 个缓存实例 | 跨帧传播和关联 |
| DN | 无 | DN + temporal DN | 稳定匹配和训练 |
| 地图先验 | `ldmap_gt_flag` 参与旧 head loss | Nav/SD encoder + LDLite encoder/fusion | 先验进入特征和 decoder |
| 输出 | 基础折线、属性和辅助 topology | geometry + subtype + topology + ID | 结构化地图 |
| Dataset | `BEVMTKDataset` | `PhigentMapDataset` | 序列化数据组织 |
| Runner | EpochBased，6 epochs | IterBased，按 5 epochs 换算 | 训练调度重构 |
| AdamW | lr=2e-4 | lr=1e-4，部分模块 0.1× | 差分学习率 |

## 4. 图像与 BEV 分支

### 4.1 Backbone

BEVLane 使用 ResNet18；当前仓库的 Plus 配置明确使用 ResNet34。两者都读取 stage 3、4 特征，通过 `CustomFPN` 输出 256 通道，再进入 LSS。输入尺寸均为 352×704，LSS 深度范围均为 1–165m，输出 64 通道，最终图像 BEV 为 128 通道。

因此，Plus 保留 camera-to-BEV 路线，但加深了图像 backbone。

### 4.2 前视 + 鱼眼双分支

BEVLane 参考配置只有 cam1、cam0 两个前视相机。Plus 增加 cam21、cam23 两个鱼眼相机。Plus detector 分别构建 high-resolution/front 与 low-resolution/fisheye 的 backbone、neck、view transformer 和 BEV encoder；两组 BEV feature 拼接后，经 1×1 `bev_fuse_conv` 压回 128 通道。

这增强了近场、侧向、路口、分叉与道路边缘覆盖。

### 4.3 范围和分辨率

BEVLane 的感知范围为 `[0,153.6] × [-12.8,12.8]m`，LSS grid 是 x=128、y=128；static voxel 为 x=1.2m、y=0.2m。

Plus 范围为 `[0,153.6] × [-25.6,25.6]m`，LSS grid 为 x=128、y=256；static voxel 改为 x=0.8m、y=0.2m，对应 x=192、y=256。静态采样栅格数从 16,384 增至 49,152，约 3 倍。

## 5. LiDAR BEV 融合

BEVLane 设置 `use_lidar=False`，point feature 始终为 `None`。

Plus 设置 Camera + LiDAR，运行于 `fusion` 模式：

1. 读取 ego 坐标系点云，最多采样 80,000 点；
2. `PillarFeatureNet` 使用 0.2×0.2×8m voxel；
3. `DenseResNet` 生成 128 通道 LiDAR BEV；
4. `ConvFuser` 将 camera BEV 与 LiDAR BEV 融合为 128 通道；
5. 训练时以 20% 概率丢弃全部 LiDAR，测试时不丢弃。

当前 `pts_bbox_head=None`，说明 LiDAR 是为地图 BEV 提供几何增强，而不是增加独立 3D 检测 head。

## 6. 地图先验

### 6.1 BEVLane 的 `ldmap_gt_flag`

旧 pipeline 收集 `ldmap_gt_flag` 并将其传给 map head loss，但 detector 没有地图 encoder，也没有把地图 feature 融入 sensor BEV。因此它不能等同于 Plus 的 LDLite feature fusion。

### 6.2 Nav/SD 矢量先验

Plus 使用 `SDMapAdaptor` 生成 nav path、SD road 等输入，再由 `MapGraphTransformer` 编码为 128 维 feature。当前 `map_mode_info="only_nav"`，实际选择 nav path；decoder 新增 `cross_attn_nav` 使用该先验。

`use_nav_flag_init=False` 只表示不用导航 feature 初始化 query anchor，不表示导航先验未参与 decoder。

### 6.3 LDLite 栅格先验

Plus 当前配置 `use_ldlite=True`：

1. `LDLiteLoader` 读取 `_ldlite.pkl` 中的 section、lane、semantic、boundary 属性图；
2. `LDLiteEncoder` 对离散属性做 embedding，生成 128 通道、200×200 的 LDLite BEV；
3. LDLite 覆盖 300m×300m，分辨率 1.5m；
4. `LDLiteFusionModule` 以 sensor BEV 为 query、LDLite BEV 为 value；
5. sensor world coordinates 被转换为 LDLite normalized reference points；
6. 使用 8 heads、4 sampling points 的 deformable cross-attention；
7. 经 residual、LayerNorm、FFN 后输入 map decoder。

训练时 LDLite `drop_prob=0.5`，测试时为 0，目标是兼顾先验收益与先验缺失时的降级能力。

## 7. Query 与 decoder 重构

### BEVLane

- 30 个 map query；
- 4 个 junction query；
- `query_embed_type="instance_pts_v2"`；
- 2 层 `MapQRDecoder`；
- self-attention → BEV deformable cross-attention → FFN；
- 每帧独立解码。

### BEVLanePlus

Plus 将 query 划为 6 组：

```text
[72, 27, 24, 63, 1, 54]，共 241 个 classified sparse anchors
```

`anchor_cls_idxs` 被 target sampler、refinement、InstanceBank 和 post-process 共同使用。每类要素拥有独立容量，anchor 和 feature 也可跨帧保存。

最终 `class_names_map` 有 7 类，而 anchor 数组有 6 组；roadmark 等任务在 head/decoder 内有专门分支，不能把数组和最终类别简单一一对应。

Plus decoder 为 3 层：第 1 层是 single-frame decoder，第 2、3 层启用 temporal attention；同时增加 navigation-map cross-attention，并由两级 `BEVRefinementModuleOE` 分阶段更新几何、属性和拓扑。

## 8. InstanceBank 与时序升级

Plus 开启 `temporal_map`、`with_track`、`with_instance_id` 和 `with_trans_loss`。时序实例配额为 `[24,9,8,21,0,0]`，共 62 个。

`InstanceBankMapOE` 负责：

- 缓存上一帧 instance feature、anchor 和 confidence；
- 根据前后帧位姿将 anchor 传播到当前帧；
- 平滑并增强时序 anchor；
- 合并和更新历史实例与当前 query；
- 缓存当前高置信度实例供下一帧使用；
- 计算跨帧 feature/geometry 一致性 loss；
- 提供 instance ID 关联信息。

因此 Plus 从逐帧矢量检测器升级为有在线状态的地图实例跟踪器。

## 9. 标签和输出升级

BEVLane 主类包括 lane、road、stopline、crosswalk、roadmark，并有 junction 辅助类；属性包括 boundary type/color、road boundary、roadmark、fishbone，以及 4 通道辅助 topology。

Plus 实例语义为 lane、bound、horizon line、polygon、road boundary、guideline、roadmark，并联合预测：

- 18 类 lane type；
- 7 类 turn type；
- 18 类 lane-boundary type；
- 10 类 boundary color；
- 9 类 fishbone；
- 9 类 road-boundary subtype；
- 7 类 horizon-line subtype；
- 7 类 polygon subtype；
- 20 类 roadmark subtype；
- 3 类 guideline；
- 16 类 lane-lane topology；
- topology distance、lane distance；
- instance ID。

输出目标从“地图折线检测”升级为“几何 + 属性 + 关系 + 身份”的结构化在线地图。

## 10. 匹配、DN 和 loss

BEVLane 使用 `MapTRAssigner`，classification cost=5、OrderedPtsL1 cost=1；主要 loss 为 Focal、PtsL1、方向、可选 BEV segmentation 和 junction classification。

Plus 使用 `SparsePoint3DTempSubtypeTarget` + `HungarianLinesAssigner`，分别支持 sensor range 和 SD-map range 的 normalized query cost，并开启：

- 4 个 DN groups；
- 3 个 temporal DN groups；
- 坐标噪声 `[1.5,1.5]`；
- 最多 24 个 DN GT；
- negative DN queries。

损失扩展为 instance classification、sparse line、direction、各类 subtype、topology classification/distance、lane distance 和 temporal transformation consistency。当前总权重中 `inst_cls:pts=1.5:0.5=3:1`；它与旧 assigner 的 5:1 cost 位于不同阶段，不能直接视为相同权重。

## 11. 数据和训练策略

### BEVLane

- `BEVMTKDataset`，单帧 pipeline；
- `PrepareImageInputsBevLaneV2`；
- `LoadAnnotationsBEVStaticV2`；
- 颜色、JPEG、YUV、Gaussian/Poisson noise 增强；
- EpochBasedRunner，6 epochs；
- batch size 12，workers 6；
- AdamW lr=2e-4，grad norm 35；
- EMA hook。

### BEVLanePlus

- `PhigentMapDataset`，`with_seq_flag=True`；
- 序列内 augmentation 保持一致；
- `SeqPrepareImageInputsBevLaneV2`；
- `LoadAnnotationsStaticSubType`；
- `LoadPointsFromFile`、`LDLiteLoader`、`SDMapAdaptor`；
- `NuScenesSparse4DAdaptor` 整理稀疏时序 target；
- brightness/contrast/gamma/blur/JPEG/chromatic aberration；
- BEV rotation ±10°，50% 触发；
- IterBasedRunner，按帧数和 5 epochs 换算 max_iters；
- batch size 16，workers 8；
- AdamW lr=1e-4；image backbone、point encoder/backbone 使用 0.1× lr；
- dynamic FP16、gradient clamp=1、grad norm 10；
- EMA 与可视化 hooks。

配置中 detector 的 `use_grid_mask=False`，但 `SeqPrepareImageInputsBevLaneV2` 收到 `grid_mask_config`。是否对原图应用 GridMask需看 pipeline 实现，不能仅根据 detector 开关判断完全关闭。

## 12. 推理数据流

### BEVLane

```text
2路前视图像 → ResNet18/FPN → LSS → static BEV encoder
→ 30 map + 4 junction queries → 2-layer MapQRDecoder
→ vectors + attributes/topology
```

### BEVLanePlus

```text
前视图像 + 鱼眼图像 → 双图像/LSS/BEV分支 → camera BEV ─┐
LiDAR → PillarFeatureNet → DenseResNet → LiDAR BEV ─────┤
                                                       └→ ConvFuser → sensor BEV

LDLite attributes → LDLiteEncoder → LDLite BEV
sensor BEV + LDLite BEV → deformable cross-attention fusion

Nav/SD polylines → MapGraphTransformer → nav feature
上一帧实例 → ego-motion传播 → InstanceBank features/anchors

fused BEV + nav feature + temporal instances
→ 241 classified sparse anchors
→ 3-layer sparse temporal decoder + staged refinement
→ geometry + subtype + topology + instance ID
```

## 13. 升级价值

1. ResNet34 提升图像表征能力。
2. 鱼眼与翻倍横向范围改善近场、侧向和复杂路口覆盖。
3. LiDAR 提升弱光、低纹理、遮挡场景的几何可靠性。
4. Nav/SD 与 LDLite 分别从矢量和栅格层面提供地图先验。
5. 分类别 sparse anchor 提高实例容量并减少不同任务争抢 query。
6. InstanceBank、位姿传播和 temporal loss 降低逐帧抖动并提供 ID 连续性。
7. subtype 与 topology 联合输出更接近下游规划需要的地图结构。
8. LiDAR 20% 丢弃、LDLite 50% 丢弃增强输入缺失时的降级能力。

## 14. 当前配置的限制与公平对比风险

1. Plus 文件明确写着 `LDLiteMap Configuration - Standalone`，是专项配置，不一定代表全部 Plus 训练方案。
2. Plus 使用 `train_list="data/tmp.txt"` 动态计算 max_iters，可能不是正式全量训练清单。
3. Plus 默认 `load_from="ckpt/v006.pth"`，指标可能包含预训练影响。
4. `with_bev_seg=False`，辅助 BEV segmentation head 当前实际关闭。
5. `with_depth_branch=False` 不代表 LSS 不预测 depth distribution，只表示额外深度相关分支关闭。
6. 两版同时改变 backbone、相机数、LiDAR、范围、数据、taxonomy、query、时序和先验，最终指标不能直接归因于某一模块。
7. taxonomy 有合并、拆分和 subtype 迁移，类别 AP 不能只按名称直接对齐。
8. Plus 是有状态时序模型；评测时需明确序列边界和 InstanceBank reset，否则结果不可复现或不公平。

## 15. 建议消融顺序

在统一数据、标签映射、BEV 范围和评测协议下依次增加：

1. BEVLane ResNet18 camera baseline；
2. ResNet34；
3. 扩大范围和提高 x 分辨率；
4. 鱼眼相机；
5. classified sparse anchors/refinement；
6. DN；
7. sequence + InstanceBank + temporal attention；
8. temporal DN + trans loss；
9. LiDAR fusion；
10. Nav/SD prior；
11. LDLite fusion，分别测试先验 0%、50%、100% 可用率；
12. subtype/topology 多任务头。

建议报告 geometry AP/F1、距离/横向分桶、subtype F1、topology F1、temporal jitter、ID switch、恶劣场景、各模态缺失降级、参数量、FLOPs、显存和端到端延迟。

## 16. 最终归纳

```text
BEVLane：单帧纯视觉矢量检测
    ↓
BEVLanePlus：多相机 + LiDAR 大范围 BEV
    ↓
Nav/SD + LDLite 双地图先验
    ↓
分类别 sparse query + 多阶段 refinement
    ↓
InstanceBank 驱动的跨帧结构化在线地图生成
```

最具代际意义的不是单独的 ResNet18→ResNet34，而是 **多模态 BEV、双地图先验、稀疏分类别 query、时序实例记忆、结构化拓扑输出** 的组合升级。
