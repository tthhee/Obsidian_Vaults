# `_build_dn_targets` 实现逻辑与 DN 类别说明

## 1. `_build_dn_targets` 的实现逻辑

函数位于：

```text
projects/mmdet3d_plugin/models/custom_lanehead_mapqr_v2.py:346
```

它的作用是把数据集中按地图元素类型分别保存的 GT，整理成 DN（Denoising Training）采样器需要的三组、按 batch 分开的目标：

```python
cls_target: List[Tensor[Ni]]
reg_target: List[Tensor[Ni, num_sample * 2]]
id_target: Optional[List[Tensor[Ni]]]
```

DN 训练会根据真实地图实例生成一批带噪声的 query/anchor，让 decoder 学习把带噪目标恢复成真实类别和真实点坐标。

### 1.1 检查必要的 GT 字段

首先要求输入包含三种基本地图实例：

```python
required = [
    "gt_map_lane_instances",
    "gt_map_stopline_instances",
    "gt_map_crosswalk_instances",
]
```

具体要求如下：

- lane 必须包含中心线 `gt_map_pts`；
- lane 必须包含左边界 `gt_map_left_pts`；
- lane 必须包含右边界 `gt_map_right_pts`；
- stopline 必须包含 `gt_map_pts`；
- crosswalk 必须包含 `gt_map_pts`。

任意字段缺失都会直接返回：

```python
return None, None, None
```

这里检查的是字段是否存在，并不要求字段内一定有非空实例。因此，一张图没有 stopline 是允许的，只要 `gt_map_pts` 字段存在并保存为空 tensor。

### 1.2 定义各类 GT 到训练目标的映射

`target_specs` 是整个函数的核心配置：

```python
target_specs = [
    ("gt_map_lane_instances", "gt_map_pts",       0, "gt_map_instance_ids", 0),
    ("gt_map_lane_instances", "gt_map_left_pts",  1, "gt_map_instance_ids", 1),
    ("gt_map_lane_instances", "gt_map_right_pts", 1, "gt_map_instance_ids", 2),
    ("gt_map_stopline_instances", "gt_map_pts",   2, "gt_map_instance_ids", 3),
    ("gt_map_crosswalk_instances", "gt_map_pts",  3, "gt_map_instance_ids", 4),
]
```

每个元素依次表示：

```text
GT 根字段、点坐标字段、DN 类别编号、实例 ID 字段、ID 类型槽
```

映射关系如下：

| 地图目标 | 回归点字段 | `cls_id` | `type_slot` |
|---|---|---:|---:|
| 车道中心线 | `gt_map_pts` | 0 | 0 |
| 车道左边界 | `gt_map_left_pts` | 1 | 1 |
| 车道右边界 | `gt_map_right_pts` | 1 | 2 |
| 停止线 | `gt_map_pts` | 2 | 3 |
| 人行横道 | `gt_map_pts` | 3 | 4 |
| 道路边界，可选 | `gt_map_pts` | 4 | 5 |

左、右边界的分类标签都是 `1`，说明二者属于同一个语义类别；但它们的 `type_slot` 分别为 `1` 和 `2`，用于构造不同的时序实例 ID。

如果存在道路边界 GT，也会动态追加：

```python
(
    "gt_map_road_bound_instances",
    "gt_map_pts",
    4,
    "gt_road_bound_instance_ids",
    5,
)
```

### 1.3 兼容 DataContainer 和不同 batch 包装形式

#### `unwrap()`

```python
def unwrap(value):
    if value.__class__.__name__ == "DataContainer":
        return value.data
    return value
```

如果输入是 MMCV 的 `DataContainer`，则取出其中的 `.data`。

#### `as_batch_list()`

```python
def as_batch_list(value):
    value = unwrap(value)
    if isinstance(value, (list, tuple)):
        return [unwrap(x) for x in value]
    return [value]
```

它保证结果一定是 batch list：

```text
[样本 0 的 GT, 样本 1 的 GT, ...]
```

即使输入只是单个 tensor，也会转换成长度为 1 的 list。

#### `select_batch()`

```python
def select_batch(value, batch_idx, batch_size):
    value = unwrap(value)
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return unwrap(value[batch_idx])
    return value
```

如果 value 看起来是 batch list，就取第 `batch_idx` 项；否则认为 value 本身已经是当前样本的数据。

### 1.4 将点坐标统一展平

`flatten_points()` 将不同格式的 GT 点转换成：

```text
[num_instances, num_sample * 2]
```

每一行相当于：

```text
[x0, y0, x1, y1, ..., xK, yK]
```

#### 空 GT

```python
if pts.numel() == 0:
    return pts.new_zeros((0, self.num_sample * 2))
```

返回 shape 固定的空 tensor。

#### 四维输入

```python
if pts.dim() == 4:
    return pts.flatten(2, 3)[:, 0]
```

通常输入 shape 为：

```text
[num_instances, num_orders, num_sample, 2]
```

首先变成：

```text
[num_instances, num_orders, num_sample * 2]
```

然后通过 `[:, 0]` 选择第 0 种点序或方向，得到：

```text
[num_instances, num_sample * 2]
```

部分矢量地图 GT 会保存正序、逆序等多种等价排列；这里固定选择第一种。

#### 三维输入

```python
if pts.dim() == 3 and pts.shape[-1] == 2:
    return pts.flatten(1, 2)
```

将：

```text
[num_instances, num_sample, 2]
```

转换成：

```text
[num_instances, num_sample * 2]
```

其他格式则保留第一维作为实例数，其余维度全部展开。

### 1.5 将复杂 instance ID 转成“一实例一个 ID”

`ids_to_tensor()` 将不同数据格式的 ID 统一成：

```text
[num_instances]
```

并转换为 `torch.long`。

一个地图实例的 ID 有时不是单个整数，而可能包含多条 component line 的 ID。函数通过 `first_valid_id()`：

1. 递归展开 list、tuple 或 tensor；
2. 找出所有大于等于 0 的 ID；
3. 返回第一个合法 ID；
4. 如果全都无效，则返回第一个值；
5. 如果完全为空，则返回 `-1`。

例如：

```text
[[-1, 23, 24], [31, 32]]
```

会转换成：

```text
[23, 31]
```

这样一个地图实例只对应一个稳定 ID，避免 component 数量不同导致实例 ID 和回归目标错位。

### 1.6 给 instance ID 加上目标类型编码

`build_part_ids()` 最终构造的 ID 为：

```python
return ids * 8 + int(type_slot)
```

例如，原始 lane instance ID 为 `10`：

```text
中心线：10 * 8 + 0 = 80
左边界：10 * 8 + 1 = 81
右边界：10 * 8 + 2 = 82
```

这样既保留三者共同的原始 lane ID 信息，又防止时序 DN 逻辑把中心线、左边界和右边界错误地当成同一个回归目标。

当前 `type_slot` 只使用 `0～5`，因此乘以 `8` 后，低 3 bit 足以保存目标类型。

如果出现以下任一情况，该部分 ID 会返回 `None`：

- ID 字段不存在；
- ID 数量小于实例数量；
- 存在负数 ID。

ID 是可选结果，分类和回归目标仍然可以正常构造。

### 1.7 逐 batch 构造各类型目标

batch size 从 lane 中心线字段推断：

```python
batch_size = len(
    as_batch_list(data_dict["gt_map_lane_instances"]["gt_map_pts"])
)
```

然后逐张图初始化：

```python
reg_parts = []
cls_parts = []
id_parts = []
```

对 `target_specs` 中的每种目标，依次进行：

1. 读取并展平点坐标；
2. 如果该类型为空，则跳过；
3. 将点坐标添加到 `reg_parts`；
4. 根据 `cls_id` 生成分类标签并添加到 `cls_parts`；
5. 尝试构造 instance ID 并添加到 `id_parts`。

例如，存在 3 条 lane center 时，分类目标为：

```text
[0, 0, 0]
```

存在 2 条 stopline 时，分类目标为：

```text
[2, 2]
```

### 1.8 将 lane center/left/right 按实例交错排列

如果直接拼接，lane 目标的顺序是：

```text
center0, center1, center2,
left0,   left1,   left2,
right0,  right1,  right2
```

代码通过：

```python
lane_reg_parts = torch.stack(reg_parts[:3], dim=1).flatten(0, 1)
```

将其调整为：

```text
center0, left0, right0,
center1, left1, right1,
center2, left2, right2
```

分类和 ID 也进行相同调整。

假设有两个 lane，最终得到：

```text
cls = [0, 1, 1, 0, 1, 1]
```

对应 ID 为：

```text
lane_id0 * 8 + 0,
lane_id0 * 8 + 1,
lane_id0 * 8 + 2,
lane_id1 * 8 + 0,
lane_id1 * 8 + 1,
lane_id1 * 8 + 2
```

完成 lane 交错后，再拼接 stopline、crosswalk 和可选的 road boundary。

这里隐含一个数据约束：lane center、left、right 的实例数必须一致且 shape 完全相同。

代码通过：

```python
reg_parts[0].shape == reg_parts[1].shape == reg_parts[2].shape
```

进行判断。不过它还默认前三个非空 `reg_parts` 就是 center、left、right。如果某种 lane 点为空，而后面的 stopline 非空，前三项可能不再代表三种 lane 部件。因此正常数据应保证一条 lane 的 center、left、right 同时存在。

### 1.9 处理整张图没有 GT 的情况

如果当前样本没有任何地图 GT，会生成 shape 明确的空 tensor：

```text
reg: [0, num_sample * 2], float
cls: [0], long
id:  [0], long
```

这样后续 sampler 可以统一处理空样本，而不会出现 `torch.cat([])` 报错。

### 1.10 拼接每张图的所有目标

非空情况下：

```python
reg_target.append(torch.cat(reg_parts, dim=0))
cls_target.append(torch.cat(cls_parts, dim=0))
```

最终每张图得到：

```text
reg_target[b]: [Ni, num_sample * 2]
cls_target[b]: [Ni]
```

其中：

```text
Ni = 3 * lane 数
   + stopline 数
   + crosswalk 数
   + road boundary 数
```

一条 lane 会被拆成三个 DN 目标：中心线、左边界和右边界。

### 1.11 ID 采用“全有或全无”策略

只有当前图的所有非空目标都有合法 ID，并且：

```python
len(id_parts) == len(reg_parts)
```

才会拼接 ID。

只要 batch 中任意样本或任意目标类型缺少合法 ID：

```python
all_have_instance_ids = False
```

最终整个 batch 都返回：

```python
id_target = None
```

这样可以防止后续时序匹配把不完整的 ID 当成完整数据使用。

不过当前调用位置显式设置了：

```python
gt_instance_id = None
dn_metas = self.sampler.get_dn_anchors(
    dn_cls_target, dn_reg_target, gt_instance_id
)
```

因此，当前版本虽然 `_build_dn_targets()` 能构造 `id_target`，调用方暂时丢弃了它，实际采用的是 static DN，而不是基于 instance ID 的 temporal DN。

### 1.12 输出如何进入 DN 训练

函数返回：

```python
return cls_target, reg_target, id_target
```

整体数据流如下：

```text
_build_dn_targets()
        │
        ▼
sampler.get_dn_anchors(cls_target, reg_target, None)
        │
        ├── dn_anchor：在 GT 附近添加噪声后的 anchor
        ├── dn_reg_target：原始点坐标
        ├── dn_cls_target：正负 DN 标签
        ├── dn_attn_mask：不同 DN group 之间的注意力隔离
        └── valid_mask：padding 中哪些位置有效
        │
        ▼
将 dn_anchor 拼接到普通 reference_points 后面
        │
        ▼
Transformer decoder 预测去噪结果
        │
        ▼
DN classification loss + DN point regression loss
```

一句话概括：`_build_dn_targets()` 把 lane、边界、停止线、人行横道等异构 GT，统一整理成按 batch 保存的“类别 + 展平点坐标 + 可选稳定实例 ID”，再交给 sampler 生成带噪 query，用于 decoder 的去噪辅助训练。

## 2. DN target 的类别与非 DN 目标类别是否一致

结论是：**不完全一致，需要区分 `_build_dn_targets()` 生成的原始类别和真正进入 DN classification loss 的类别。**

### 2.1 `_build_dn_targets()` 生成的原始类别

该函数生成的是地图元素的语义类别：

```text
0：lane center
1：lane left/right boundary
2：stopline
3：crosswalk
4：road boundary
```

这些类别用于 `sampler.get_dn_anchors()` 生成 DN query，语义来源与普通 query 表示的地图目标基本相同。

但它不一定与非 DN 分支最终使用的分类 head 类别编号逐项相同，因为普通分支可能由不同的 head 分别处理 lane、boundary、polygon 等目标。

### 2.2 真正进入 DN loss 后变成二分类

在 `prepare_for_dn_loss()` 中，原始 DN 类别被转换为：

```python
dn_pos_mask = raw_dn_cls_target >= 0

dn_cls_target = torch.where(
    dn_pos_mask,
    raw_dn_cls_target.new_ones(raw_dn_cls_target.shape),
    raw_dn_cls_target.new_zeros(raw_dn_cls_target.shape),
).to(torch.int64)
```

也就是：

```text
原始类别 >= 0  →  1，正 DN query
原始类别 < 0   →  0，负 DN query
```

对应关系为：

```text
lane center (0) ───────┐
boundary (1) ──────────┤
stopline (2) ──────────┼──→ DN classification target = 1
crosswalk (3) ─────────┤
road boundary (4) ─────┘

噪声生成的负目标 (-1) ─────→ DN classification target = 0
```

因此，DN loss 最终没有区分五种地图语义类别，而只判断一个 DN query 是正目标还是负目标。

### 2.3 DN 与非 DN 目标的关系

| 阶段 | DN query | 普通 query |
|---|---|---|
| GT 来源 | 相同的地图 GT | 相同的地图 GT |
| query 生成方式 | 在 GT 附近添加噪声 | 可学习或预设 anchor |
| `_build_dn_targets` 类别 | 0～4 的地图类型 | 不直接使用该函数 |
| 最终 classification target | 二分类：正目标 1 / 负目标 0 | 由普通 matching 和各分类分支决定 |
| regression target | 对应 GT 点坐标 | 匹配得到的 GT 点坐标 |

所以：

- DN 和非 DN 目标的 GT 语义来源基本一致；
- `_build_dn_targets()` 的初始编码保留地图类型；
- 当前实现会在真正计算 DN 分类损失时将类别压缩成正负二分类；
- 因此最终 DN 类别监督与普通分支的类别监督并不完全一致；
- DN 点回归 loss 只对 `raw_dn_cls_target >= 0` 的正 DN query 计算。

### 2.4 需要额外确认的配置

DN loss 使用：

```python
self.loss_inst_cls
```

因此还需要结合配置中的 `loss_inst_cls` 和 decoder 的 `inst_cls_layers` 输出维度判断：

- 如果该 head 本身是“是否为有效地图实例”的二分类 head，当前实现是匹配的；
- 如果该 head 原本设计为多语义类别输出，那么 DN loss 把 target 压成 `0/1`，意味着这里只会监督类别索引 0 和 1。

