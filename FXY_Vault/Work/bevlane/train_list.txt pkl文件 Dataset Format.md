
## 分析结论

这个 pkl 是一个 **"索引/列表" 格式**，它本身不直接存样本数据，而是存一份**样本 ID 清单（iid）+ 共享元数据**。真正的每帧数据在 `info_dir` 下的独立小 pkl 里，按需加载。

### 1. 顶层结构（list pkl，3.5MB）

```
{
  'metadata': {
      'version': 'phi_map',
      'match_config': {'ISP_DELT': 0, 'CAM_DELT_MAP': {0:0,1:0,4:0,5:0,6:0,7:0,8:0,20:0,21:0,22:0,23:0}},
      'calibration': { <calib_name>: {sweeps, cams{cam1{...}}, lidar2ego_*, ego2global_*}, ... },   # 多套相机标定
      'e-DYP001_20260626092259_V1': ...,   # 数据版本标识
      'auto_label_version': ...,
      'info_dir': '/mnt/cfs-baidu/public/zhangheng/data/dataset/j6b_dataset/train_df_data_20260806_single/infos',
      'data_root': ...,
  },
  'infos': [ {'iid': '<sample_id>'}, {'iid': '<sample_id>'}, ... ]   # 共 22042 条
}
```

关键点：`infos` 里每个元素**只有一个键 `iid`**，值是形如
`2072943036957659136-17DYP0020_20260623T201622-1783057437551-clip_17DYP0020_20260623T20162217822170427351782217102735@2072943036957659136@1782217042766` 的字符串（`样本token/时间戳/帧号` 拼出来的 ID）。

### 2. iid 的解析（惰性加载）

消费逻辑在 [bevmtk_dataset.py:111-126](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/datasets/bevmtk_dataset.py#L111-L126)：当 `info` 含 `iid` 时，去掉 `.pkl` 后缀，然后去 `metadata['info_dir']/{iid}.pkl` 加载真正的样本文件：

```python
info_file = osp.join(self.metadata['info_dir'], '{}.pkl'.format(info['iid']))
info = pickle.load(open(info_file,'rb'))
```

`info_dir` 目录里共有 **842938** 个单帧 pkl（这是全量池子），本 list 从里面挑了 **22042** 帧作为训练子集。

### 3. 单帧 pkl 的内容（真正的训练样本，约 6KB）

取第一行 iid 对应文件看，顶层键为：

| 键 | 内容 |
|---|---|
| `cams` | `cam1`/`cam0`，各自带 `data_path`（4K 相机 jpg 路径，如 `/mnt/goosefs/.../JPG_4K_CAM_FW/frame-001186_1782217042766.jpg`） |
| `lidar_path` | 空字符串（纯视觉，无点云） |
| `token` | 样本 ID（== iid） |
| `ann_infos` | `pred_scores`、`speed` 等 |
| `calibration` | 标定版本名 `'E_D_YP002_20260613_V2_to_V1'`，用于去 `metadata['calibration']` 里取相机内外参 |
| `road_bound_instances` | 道路边界实例（list of dict：`points`(numpy f8)、`colors`(i8)、`styles`、`types`、`direction`、`fishbone`、`id/ids`、`bound_type`…） |
| `lane_bound_instances` | 车道线实例 |
| `stop_line_instances` | 停止线 |
| `cross_walk_instances` | 人行横道 |
| `road_mark_instances` | 路面标记（箭头/文字等） |
| `attr_info` | 场景属性开关：`Night`、`RainSnow`、`LightInterference`、`Tunnel`、`Construction`、`UrbanCurve`、`has_fishbone`、`has_stopline`、`has_speedbump`、`has_cover`、`has_pinhole`、`has_fisheye` 等几十个布尔量 |
| `ignore_region_instances` | 忽略区域 |
| `ego2global_rotation` / `ego2global_translation` | 自车位姿（numpy 标量/向量） |

相机内外参不在单帧里，而是通过 [complement_info](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/datasets/bevmtk_dataset.py#L127-L128) 从 `metadata['calibration'][info['calibration']]` 合并进来（`sensor2ego_translation/rotation`、`cam_intrinsic`(3×3)、`cam_distorted`(14 维畸变)、`P_ego2img`(3×4)、`height/width` 等）。

### 4. 关于 txt 行末的 ` 0`

[custom_3d.py:75-80](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/datasets/custom_3d.py#L75-L80) 会把 `ann_file` 按空格拆开，`0`/`1` 存为 `self.ldmap_gt_flag`：

- `0` = 普通车道线样本（无 junction GT）
- `1` = 含 junction（路口拓扑）GT 的样本

它随后进入每个样本的 `input_dict`（[bevmtk_dataset.py:135](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/datasets/bevmtk_dataset.py#L135)），训练时用来把 loss 权重按 `ldmap_gt_flag==0 / ==1` 分开清零（见 [custom_lanehead_mapqr_v2_j5.py:981-993](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2_j5.py#L981-L993)），即普通样本不算 junction loss、junction 样本不算普通车道 loss。

### 5. 与训练流程的衔接

`v035_train.txt` 每一行 → 一个 `BEVMTKDataset`（`ann_file=该list pkl`，`load_interval=4`）→ 全部拼成一个 `ConcatDataset` → 交给上一轮讨论的 `DistributedGroupSampler`。所以 "取第一行数据" 只是整个训练集 22042×若干行 中的第一份。

需要我进一步把某个具体字段（比如 `lane_bound_instances` 里一条实例的完整字段和坐标系含义）也展开吗？

## lane_bound_instances 字段展开

`lane_bound_instances` 是单帧 pkl 里的一个 **list**，每个元素是一条**车道线实例**（dict / defaultdict）。我用第一行样本的第一个实例实测解码如下。

### 1. 单条实例的字段结构（共 9 个字段）

| 字段 | 类型 | 含义 |
|---|---|---|
| `points` | ndarray `(N, 2)` float64 | 折线顶点，N 个 `(x, y)`，**BEV 自车坐标系（米）** |
| `colors` | ndarray `(N,)` int64 | 每个点的**颜色**类型，`LaneBoundaryColorType` id |
| `styles` | ndarray `(N,)` int64 | 每个点的**线型**，`LaneBoundaryType` id |
| `types` | ndarray `(N,)` int64 | 每个点的 `LineType` id（道路边界用；车道线里与 styles 冗余） |
| `direction` | 标量 int | 车道方向 `LaneDirectionType`（0=Single, 1=Double） |
| `fishbone` | ndarray `(N,)` int64 | 每个点的**鱼骨线**类型 `FishboneLineType` id |
| `id` | 标量 int | 实例 id |
| `ids` | ndarray `(1,)` int64 | 实例 id 的数组形式 |
| `bound_type` | str | 实例类别：`'lane'`（车道线）/ `'road'`（道路边界） |

注意：`direction`、`id` 是**标量**，其余是逐点数组（每个点一个类型值，允许一条线内类型分段变化）。

### 2. 第一行样本 · 第一个实例的实际值

- `points`：14 个点，从 `(0.29, 4.62)` 到 `(153.60, 4.42)`——一条沿 x 正方向、位于自车左侧约 4.4m 的**直道线**，覆盖 0~153.6m：

```
pt0  x=  0.292  y=4.618     pt7  x= 83.847  y=4.447
pt1  x= 10.921  y=4.581     pt8  x= 97.526  y=4.440
pt2  x= 21.550  y=4.544     pt9  x=111.193  y=4.436
...                         ...
pt13 x=153.600  y=4.424
```

- `colors`：全 `1` = **White**（白色）
- `styles`：全 `2` = **SingleDashed**（单虚线）
- `types`：全 `2`（LineType.LaneMarking，车道线场景下不参与训练）
- `direction`：`0` = Single
- `fishbone`：全 `7` = **None**（非鱼骨线）
- `id`：`69`；`ids`：`[69]`
- `bound_type`：`'lane'`

### 3. 坐标系

`points` 的 `(x, y)` 就是 BEV 自车坐标系，与配置的 [bsl.py:22](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/configs/psd-bevlane-j6/v035_2_j6/bsl.py#L22) `point_cloud_range_bev_static = [0.0, -12.8, -3.0, 153.6, 12.8, 5.0]` 一致：

- `x`：纵向（前进方向），0 ~ 153.6 m
- `y`：横向（左正右负），-12.8 ~ 12.8 m

上面这条线 y≈4.4 即在左前方约 4.4 米处的一条车道线。

### 4. 各 id 的类型映射（[phigent_map_utils.py:78-236](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/datasets/phigent_map_utils.py#L78-L236)）

- **LaneBoundaryType**（styles）：`0 Unknown, 1 SingleSolid 单实线, 2 SingleDashed 单虚线, 3 DoubleSolid, 4 DoubleDashed, 5 左实右虚, 6 右实左虚, 7 ShortDashed, 8 LongDashed, 9 FishBone, …, 16 SolidFishBone, 17 DashedFishBone, 18 CrossTransferLine 待转线, 19 CrossLeadingLine 引导线, 20 StopLine, 22 SpeedBump, 23 DiversionLine, 24 DecelerationLine`
- **LaneBoundaryColorType**（colors）：`0 Unknown, 1 White, 2 Yellow, 3 Orange, 4 Blue, 5 Green, 6 Gray, 7/8 左灰右黄等`
- **FishboneLineType**（fishbone）：`0 Unknown, 1 Right, 2 Left, 3 Double, 4 LeftSawtooth, 5 RightSawtooth, 6 DoubleSawtooth, 7 None`

### 5. 训练时实际被消费的字段（[loading.py:3427-3476](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/datasets/pipelines/loading.py#L3427-L3476) `LoadAnnotationsBEVStaticV2`）

loader 只用到 **4 个字段**，其余（`types/direction/id/ids/bound_type`）被忽略：

1. `points[..., :2]` → 构造成 `LineString`，作为几何真值（回归）
2. `styles` → 经 `lane_bound_type_mapping` 映射成模型 25 类 `lane_bound_classes` 的类别 id（分类）。注意两处后处理：
   - `16(SolidFishBone)→1`、`17(DashedFishBone)→2`（鱼骨实/虚线并入实线/虚线）
   - `18(CrossTransferLine)`、`19(CrossLeadingLine)` 整条丢弃（`continue`）
3. `colors` → 映射到颜色类别（[loading.py:2125-2136](/mnt/cfs-baidu/algorithm/xiangyu.fu/dev-root-path/fxy-bevlane-dev/J6B/bevdet/mmdet3d/datasets/pipelines/loading.py#L2125-L2136) 里除 `White=1`、`Yellow=2` 外其余颜色都折叠到 1）
4. `fishbone` → `enable_fishbone=True` 时映射到 8 类 fishbone 类别

一句话总结：**原始 pkl 里每条车道线 = 一串 `(x,y)` 顶点 + 逐点的「线型/颜色/鱼骨」类型 id + 少量元信息**；训练时只用 `points` 做几何回归，用 `styles/colors/fishbone` 做分类监督。