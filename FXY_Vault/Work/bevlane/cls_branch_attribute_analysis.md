# `cls_branch` 属性分类分支完整调查结论

以下结论针对当前 J6 配置 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py)。

## 1. `cls_branch` 实际预测什么

[`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:36-46`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L36-L46) 中的 `cls_branch` 是**每条预测线的属性分类分支**，不是 query 的实例类别分支：

```python
cls_branch.append(
    self.linear(
        self.embed_dims,
        self.cls_out_channels * (self.max_change_num + 1),
    )
)
```

当前配置中，普通 lane 的 `class_metas` 顺序为：

- `lane_bound_classes`
- `lane_bound_colors`
- `road_bound_classes`
- `road_mark_classes`
- `fishbone_line_classes`

见 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:214-228`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L214-L228)。

但是，`road_mark_classes` 不属于该 `cls_branch`。road mark 有单独的 `road_mark_cls_branch`，定义于 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr.py:90-97`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr.py#L90-L97)，在 V2 forward 中单独传递，见：

- [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:182-199`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L182-L199)
- [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:254-267`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L254-L267)

因此，当前 `cls_branch` 的每个 slot 实际预测四组属性：

1. 车道边界类型；
2. 车道边界颜色；
3. 道路边界类型；
4. 鱼骨线类型。

### 1.1 车道边界类型：26 类

配置位置：[`configs/psd-bevlane-j6/v035_2_j6/bsl.py:105-132`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L105-L132)

```text
Unknown
SingleSolid
SingleDashed
DoubleSolid
DoubleDashed
LeftSolidRightDashed
RightSolidLeftDashed
ShortDashed
LongDashed
FishBone
ShadedArea
LaneVirtualMarking
IntersectionVirualMarking
CurbVirtualMarking
UnclosedRoad
RoadVirtualLine
SolidFishBone
DashedFishBone
CrossTransferLine
CrossLeadingLine
StopLine
CrossStopLine
SpeedBump
DiversionLine
DecelerationLine
Other
```

代码注释写的是“25 classes”，但逐项计算实际为 **26 类**。

### 1.2 车道边界颜色：10 类

配置位置：[`configs/psd-bevlane-j6/v035_2_j6/bsl.py:134-145`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L134-L145)

```text
Unknown
White
Yellow
Orange
Blue
Green
Gray
LeftGrayRightYellow
LeftYellowRightWhite
Other
```

### 1.3 道路边界类型：16 类

配置位置：[`configs/psd-bevlane-j6/v035_2_j6/bsl.py:147-164`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L147-L164)

```text
Unknown
Centerline
LaneMarking
Guardrail
Fence
Curb
Wall
Concrete
Nature
Canopy
Virtual
Clift
Ditch
Waterfilledbarrier
Trafficcone
Other
```

### 1.4 鱼骨线类型：8 类

配置位置：[`configs/psd-bevlane-j6/v035_2_j6/bsl.py:166-175`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L166-L175)

```text
Unknown
Right
Left
Double
LeftSawtooth
RightSawtooth
DoubleSawtooth
None
```

### 1.5 独立的 road mark 分支：21 类

road mark 类别定义于 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:177-199`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L177-L199)：

```text
Unknown
Text
Straight
StraightOrLeft
StraightOrRight
StraightUTurn
LeftTurn
LeftTurnUTurn
LeftTurnAndInterflow
RightTurn
RightTurnAndInterflow
LeftRightTurn
UTurn
NoLeftTurn
NoRightTurn
NoUTurn
StraightLeftRight
StraightULeft
RightUTurn
Cover
Others
```

该分支输出 `num_road_mark_classes=21` 个 logits，见 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:206-212`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L206-L212)，不占用普通 `cls_branch` 的 60 个通道。

---

## 2. `cls_out_channels` 和 `max_change_num`

`max_change_num` 在 V2 构造函数中直接固定为 2：

```python
self.max_change_num = 2
```

见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:29-34`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L29-L34)。它不是从配置文件传入的。

其含义是：一条预测线最多有 **2 个属性变化点**，因此最多分成 **3 个沿线 segment/slot**，并不是“预测 3 个类别”。

```text
线段 0 ──变化点 1── 线段 1 ──变化点 2── 线段 2
```

父类 [`mmdet3d/models/dense_heads/custom_lanehead_urban_v1.py:100-105`](../mmdet3d/models/dense_heads/custom_lanehead_urban_v1.py#L100-L105) 根据 `loss_cls.use_sigmoid` 设置：

```python
self.cls_out_channels = num_classes
```

当前默认 loss 是 sigmoid FocalLoss，见 [`mmdet3d/models/dense_heads/custom_lanehead_urban_v1.py:34-40`](../mmdet3d/models/dense_heads/custom_lanehead_urban_v1.py#L34-L40)，因此没有额外的显式 background 通道。

J6 配置中：

```python
num_cls_out_channels = (
    num_lane_bound_classes
    + num_lane_bound_color_classes
    + num_road_bound_classes
    + num_fishbone_classes
)
```

见：

- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:206-211`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L206-L211)
- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:245-250`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L245-L250)

实际数值为：

```text
26 + 10 + 16 + 8 = 60
```

head 配置将其作为 `num_classes` 传入，见 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:324`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L324)。

所以：

```text
cls_out_channels = 60
max_change_num + 1 = 3
cls_branch 最后一层输出 = 60 × 3 = 180
```

当前配置下，每个普通 query 的输出 shape 为：

```text
[batch, num_vec, 180]
→ [batch, num_vec, 3, 60]
```

其中：

- `3` 是三个沿线属性 slot；
- `60` 是每个 slot 拼接的四组属性 logits。

当前配置的 `num_vec=30`、预测点数为 20，见：

- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:99-102`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L99-L102)
- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:305-324`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L305-L324)

配置使用 `BEVLaneHeadMapQRV2MTK`，见 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:303-304`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L303-L304)。因此，普通输出经过 decoder layer 堆叠后为：

```text
all_pts_cls_scores: [num_decoder_layers, batch, 30, 3, 60]
```

当前 `num_decoder_layer=2`，见 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:95-96`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L95-L96)。

Junction 有独立的 `cls_junction_branch`，见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:48-54`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L48-L54)。当前 `num_classes_junction=1`，相关配置见：

- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:201-203`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L201-L203)
- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:240`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L240)
- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:253`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L253)
- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:326`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L326)

其 shape 类似：

```text
[num_decoder_layers, batch, 4, 3, 1]
```

---

## 3. 实例分类分支与 `cls_branch` 的区别

实例分类分支继承自 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr.py:67-73`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr.py#L67-L73)：

```python
inst_cls_branch.append(
    self.linear(self.embed_dims, self.num_classes_map)
)
```

当前：

```python
classes_bev_static = ("lane", "road", "stopline", "crosswalk", "roadmark")
num_classes_map = len(classes_bev_static) + 1 = 6
```

见：

- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:18-20`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L18-L20)
- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:251`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L251)
- [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:325`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L325)

因此：

- `cls_branch`：预测每条线的细粒度属性，当前为每个 query 输出 `3 × 60`；
- `inst_cls_branch`：预测 query 的地图实例大类/置信度，当前为 6 通道；
- 最终 query 的排序和筛选主要使用 `inst_cls_scores`，而不是 180 通道的 point/segment 属性 logits。

---

## 4. Decoder 和 forward 中的输出

[`mmdet3d/plugins/decoder.py:985-1009`](../mmdet3d/plugins/decoder.py#L985-L1009) 对每个 decoder layer 的输出分别应用：

```python
cls_branches[lid]
inst_cls_branches[lid]
changed_reg_branches[lid]
```

其中：

- `cls_branches[lid]` 输出属性分类；
- `inst_cls_branches[lid]` 输出实例分类；
- `changed_reg_branches[lid]` 输出两个变化坐标。

Decoder 收集各层输出，见：

- [`mmdet3d/plugins/decoder.py:1013-1022`](../mmdet3d/plugins/decoder.py#L1013-L1022)
- [`mmdet3d/plugins/decoder.py:1068-1073`](../mmdet3d/plugins/decoder.py#L1068-L1073)

V2 forward 在 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:244-247`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L244-L247) 将普通属性 logits reshape 为：

```python
[nl, bs, num_vec, max_change_num + 1, cls_out_channels]
```

当前即：

```text
[nl, bs, 30, 3, 60]
```

输出字典见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:254-267`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L254-L267)，主要包括：

```text
all_pts_cls_scores
all_inst_cls_scores
all_pts_preds
all_changed_preds
all_pts_junction_cls_scores
all_inst_junction_cls_scores
all_pts_junction_preds
all_changed_junction_preds
all_topology_preds
all_road_mark_cls_scores
all_fishbone_changed_preds
```

`all_changed_preds` 是另外的 regression branch 输出，定义于 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:67-73`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L67-L73)，shape 为：

```text
[nl, bs, num_vec, 2]
```

它不是 `cls_branch` 的一部分。

---

## 5. Target 构造

当前配置实际使用 `BEVLaneHeadMapQRV2MTK`，所以 active target 函数位于：

- [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:423-619`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L423-L619)

而不是父类 V2 的基础版本 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:272-351`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L272-L351)，后者只构造单 slot target。

### 5.1 Query 与 GT 匹配

MTK target 首先读取 `gt_lanes.inst_labels`，并按照普通 lane、junction、guide line 过滤 GT，见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:443-458`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L443-L458)。

随后调用 assigner：

```python
self.assigner.assign(
    inst_cls_score,
    pts_pred,
    selected_gt_shifts_pts,
    gt_labels,
)
```

见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:458`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L458)。

正负 query 由 assigner 产生，见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:460-465`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L460-L465)。匹配后的 GT 点 shift 通过 `order_index` 对齐，见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:487-496`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L487-L496)。

### 5.2 三个属性 target

对每个 query，构造 3 个 slot 的 target：

- `lane_bound_type_labels`
- `lane_bound_color_labels`
- `road_bound_type_labels`

见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:537-547`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L537-L547)。

这些 target 的 shape 在 flatten 前为：

```text
[num_preds, 3]
```

正样本从匹配到的 GT 三个 slot 中取标签；未匹配 query 使用 `self.num_classes` 作为 sentinel。当前普通 lane 中该 sentinel 为 60。

正样本的三个 slot 都获得 `type_label_weights=1`，负样本为 0，见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:498-500`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L498-L500)。

鱼骨线 target 在 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:508-535`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L508-L535) 构造：

- `fishbone_type_labels`：`[num_preds, 3]`
- `fishbone_changed_coords_labels`：`[num_preds, 2]`

road mark target 单独在 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:555-561`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L555-L561) 构造，只有匹配到 road mark 的 query 才有有效权重。

### 5.3 属性变化点 GT

三个 slot 的标签来自 [`mmdet3d/datasets/phigent_map_utils.py:775-844`](../mmdet3d/datasets/phigent_map_utils.py#L775-L844) 中的 `sample_by_change_points`。该逻辑会：

1. 检测线上的属性类型变化；
2. 最多保留 2 个变化点；
3. 将首段及变化后的段类型写入最多 3 个 slot；
4. 将变化坐标转换为沿线长度的归一化距离；
5. 不足的 slot 用 0 padding。

数据对象本身也将 `max_change_num` 固定为 2，见 [`mmdet3d/datasets/phigent_map_utils.py:321-324`](../mmdet3d/datasets/phigent_map_utils.py#L321-L324)。

生成的结果写回以下字段：

```text
lane_bound_labels
lane_bound_color_labels
road_bound_labels
changed_1d_coords_labels
fishbone_change_point
```

见 [`mmdet3d/datasets/phigent_map_utils.py:693-702`](../mmdet3d/datasets/phigent_map_utils.py#L693-L702)。

变化坐标 target 为：

```text
[num_preds, 2]
```

见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:549-553`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L549-L553)。其权重仅对实际大于 0 的变化坐标置 1，因此 padding 的变化位置不参与 regression loss。

---

## 6. Loss

`loss_single` 在 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:668-706`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L668-L706) 中组织单个 decoder layer 的 target 和 loss。

随后将普通 point 分类 logits reshape 为：

```python
pts_cls_scores.reshape(-1, self.cls_out_channels)
```

见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:753-765`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L753-L765)。

代码按 `class_metas` 切分属性通道，但显式跳过 `road_mark_classes`：

```python
if meta_type == "road_mark_classes":
    continue
```

见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:759-764`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L759-L764)。

因此，当前 60 通道实际切为：

```text
[0:26]   lane_bound_classes
[26:36]  lane_bound_colors
[36:52]  road_bound_classes
[52:60]  fishbone_line_classes
```

road mark 使用独立的 `road_mark_cls_scores`，见：

- [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:755-756`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L755-L756)
- [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:799-806`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L799-L806)

### 6.1 三个普通属性 loss

以下三个 loss 分别计算：

- lane bound type；
- lane bound color；
- road bound type。

代码位置：[`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:787-797`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L787-L797)。

它们均使用 `self.loss_cls`，当前为 sigmoid FocalLoss，使用 `type_label_weights`，并以：

```text
cls_avg_factor × num_pts_per_vec
```

作为 avg factor，再乘 `loss_weights['sub_cls']`。

未匹配 query 的 sentinel 会被替换为当前属性 slice 的通道数，作为 sigmoid loss 使用的 background index，见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:784-789`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L784-L789)。

鱼骨线属性 loss 在 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:808-817`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L808-L817)，其中 `-1` 标签被屏蔽。

road mark loss 位于 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:799-806`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L799-L806)，使用独立的 21 通道输出。

实例主分类 loss 使用 `inst_cls_scores`，见 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:784-785`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L784-L785)，与 `cls_branch` 的属性 loss 分开。

### 6.2 变化坐标 loss

普通变化坐标 loss 位于 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:823-826`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L823-L826)，计算形式为：

```python
self.loss_pts(
    changed_preds.sigmoid().view(-1),
    changed_coords_labels,
    changed_coords_label_weights,
    ...,
)
```

鱼骨线变化坐标 loss 位于 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:818-821`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L818-L821)。

因此，变化坐标不是 180 通道分类输出，而是单独的 2 通道 sigmoid regression。

点坐标和方向 loss 分别位于 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:901-927`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L901-L927)。

---

## 7. 解码和最终使用

Head 的 `get_lanes` 位于 [`mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py:353-373`](../mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L353-L373)，调用：

```python
self.bbox_coder.decode(preds_dicts)
```

并返回普通 lane、junction、guide line 三组结果。

当前配置使用：

```python
type="PhiGentLaneCoderV3"
```

见 [`configs/psd-bevlane-j6/v035_2_j6/bsl.py:379-389`](../configs/psd-bevlane-j6/v035_2_j6/bsl.py#L379-L389)。

### 7.1 普通 lane 解码

`PhiGentLaneCoderV3.decode` 位于 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:814-879`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L814-L879)，读取最后一个 decoder layer 的：

- `all_pts_cls_scores`
- `all_inst_cls_scores`
- `all_pts_preds`
- `all_changed_preds`
- `all_road_mark_cls_scores`
- `all_fishbone_changed_preds`

普通 lane 进入 `decode_single`，见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:863-869`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L863-L869)。

`decode_single` 位于 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:626-737`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L626-L737)，其主要步骤如下。

#### 步骤 1：reshape 属性 logits

将 point 属性 logits reshape 为：

```text
[max_num, 3, -1]
```

当前普通 lane 即：

```text
[30, 3, 60]
```

见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:629-638`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L629-L638)。

#### 步骤 2：拆分属性通道

按 `class_metas` 切分属性 slice，并跳过 `road_mark_classes`，见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:634-638`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L634-L638)。

#### 步骤 3：实例置信度排序

对实例 logits 做 sigmoid，在 map class 维度取最大值、排序，并据此重排 query，见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:639-651`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L639-L651)。

因此，最终 query 的置信度和保留顺序主要由 `inst_cls_scores` 决定。

#### 步骤 4：属性类别解码

对每个属性 slice 的 3 个 slot 做 argmax，见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:655-660`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L655-L660)，得到每条 query 的三个分段属性类别 ID。

#### 步骤 5：点坐标反归一化

对预测点进行反归一化，见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:667-674`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L667-L674)。

最终普通 lane 的 `pts` 至少包含：

```text
lines
lane_bound_classes
lane_bound_colors
road_bound_classes
change_coords
```

见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:680-694`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L680-L694)。

启用 fishbone 时，还包含：

```text
fishbone_line_classes
fishbone_changed_coords
```

见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:695-697`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L695-L697)。

road mark 分支单独 argmax 为：

```text
road_mark_classes
```

见：

- [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:652-663`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L652-L663)
- [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:685-694`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L685-L694)

实例类别则作为最终结果中的：

```text
labels
```

而不是 `lane_bound_classes` 等细粒度属性类别。Topology 在 query 筛选后同步重排，见 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:699-732`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L699-L732)。

Junction 通过 `decode_single_junction` 单独解码，调用位置为 [`mmdet3d/core/bbox/coders/phigent_lane_coder.py:871-873`](../mmdet3d/core/bbox/coders/phigent_lane_coder.py#L871-L873)。

---

## 8. 总结

`cls_branch` 的准确含义是：

> 对每条 query 的最多三个沿线属性段，联合预测车道边界类型、车道边界颜色、道路边界类型和鱼骨线类型；当前每段 60 类，每条 query 输出 `3 × 60 = 180` 个 logits。road mark、实例类别、变化坐标和 junction 均由独立分支处理。
