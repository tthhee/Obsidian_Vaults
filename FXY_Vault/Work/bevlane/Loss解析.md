# 一, `loss()` 处理流程详解

`loss()` 是整个 head 损失计算的**编排层入口**，负责把 decoder 每一层都跑一遍 `loss_single`，再组织成返回给训练框架的 `loss_dict`。

> 位置：[custom_lanehead_mapqr_v2.py:983-1105](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L983-L1105)

---

## 1. 函数签名与输入

```python
def loss(self, gt_vecs_list, preds_dicts, is_train_bev_static, ldmap_gt_flag):
```

|输入|含义|
|---|---|
|`gt_vecs_list`|每样本一个 `PhigentInstanceLines` GT 容器对象（含坐标 + 各类标签）|
|`preds_dicts`|`forward()` 返回的所有预测（**每个字段是按 decoder 层组织的 list**）|
|`is_train_bev_static`|`[bs]` 布尔张量，该图静态 BEV 是否参与训练|
|`ldmap_gt_flag`|`[bs]` 整型张量，`0`=普通线 / `1`=junction / `2`=guideline|

---

## 2. 拆解预测

```python
all_pts_cls_scores = preds_dicts['all_pts_cls_scores']     # 每层一个 [bs, ...]
all_inst_cls_scores = preds_dicts['all_inst_cls_scores']
all_road_mark_cls_scores = preds_dicts['all_road_mark_cls_scores']
num_dec_layers = len(all_pts_cls_scores)                   # decoder 层数
if all_road_mark_cls_scores is None:
    all_road_mark_cls_scores = [None] * num_dec_layers     # 无 road mark 就补 None 占位
all_pts_preds  = preds_dicts['all_pts_preds']
all_changed_preds = preds_dicts['all_changed_preds']
all_fishbone_changed_preds = [None]*num_dec_layers if ... is None else ...
all_pts_junction_cls_scores = preds_dicts['all_pts_junction_cls_scores']
...
```

关键点：`forward()` 里这些字段大多是 `list`，**每个元素对应一个 decoder 层**（`with_box_refine` 时逐层输出）。`num_dec_layers = len(all_pts_cls_scores)` 就取自这里。

---

## 3. 特殊处理：拓扑只在最后一层算

```python
all_topology_preds = [None] * (num_dec_layers-1) + [preds_dicts['all_topology_preds']]
```

拓扑分支只在**最后一个 decoder 层**输出预测，所以前面各层塞 `None`，表示「这些层不算拓扑损失」。

---

## 4. 设备与逐层重复

```python
device = all_pts_cls_scores.device
is_train_bev_static_list = [is_train_bev_static.to(device) for _ in range(num_dec_layers)]
ldmap_gt_flag_list = [ldmap_gt_flag.to(device) for _ in range(num_dec_layers)]

gt_shifts_pts_list = []
for gt_vec in gt_vecs_list:
    instances_tensor = gt_vec.instances
    gt_shifts_pts_list.append(instances_tensor.to(device))   # 每样本 [n, shifts, fixed_num, 2]

all_gt_vecs_list = [gt_vecs_list for _ in range(num_dec_layers)]
all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
```

- 门控量 `is_train_bev_static`、`ldmap_gt_flag` 和两份 GT（容器 + 坐标）都**按层数复制**，因为每层都要用同一份 GT 重新匹配、重新算损失。
- `gt_vecs_list`（完整容器）和 `gt_shifts_pts_list`（只有坐标）分两条线往下传。

---

## 5. 逐层调用 loss_single

```python
losses_cls, losses_pts, losses_dir, losses_lane_cls, losses_changed_coords, \
losses_fishbone_changed_coords, loss_topo_cls, loss_topo_dist, \
loss_junction_cls, lane_junction_type_loss, losses_changed_coords_junction, \
loss_junction_pts, loss_junction_dir = multi_apply(
    self.loss_single, all_pts_cls_scores, all_inst_cls_scores, all_road_mark_cls_scores,
    all_pts_preds, all_changed_preds, all_fishbone_changed_preds, all_topology_preds,
    all_gt_shifts_pts_list, is_train_bev_static_list,
    all_pts_junction_cls_scores, all_inst_junction_cls_scores, all_pts_junction_preds,
    all_changed_junction_preds, all_gt_vecs_list, ldmap_gt_flag_list)
```

`multi_apply` 把每层的对应元素拿出来，**逐层**调用 `loss_single`。结果每一项都是「长度 = num_dec_layers 的 list」：

- `losses_cls[i]` = 第 i 层的实例分类损失，等等。

---

## 6. 可选辅助损失

### 6.1 guideline 引导线

```python
if self.with_guideline:
    loss_guideline_pts, loss_guideline_dir = multi_apply(
        self.loss_single_guideline, all_pts_guideline_preds, all_gt_shifts_pts_list, ldmap_gt_flag_list)
```

只在 `with_guideline` 时算，针对 `ldmap_gt_flag==2` 的样本做引导线回归。

### 6.2 BEV 分割辅助

```python
if self.aux_seg['use_aux_seg'] and self.aux_seg['bev_seg'] and preds_dicts['seg'] is not None:
    extended_mask = is_train_bev_static.view(num_imgs,1,1,1).expand_as(seg_output)
    seg_gt = torch.stack([gt_vecs_list[i].bev_seg_mask for i in range(num_imgs)], dim=0)
    modified_seg_output = torch.where(extended_mask, seg_output, -1e6)   # 不训练样本压成极小值
    loss_bev_seg = self.loss_bev_seg(modified_seg_output, modified_seg_gt) * ...
```

---

## 7. 组织 loss_dict（核心输出）

只把**最后一层**的损失以正式名字放进 dict，用于 backward；中间层用 `d0.`/`d1.` 前缀记录（作监控日志）。

### 7.1 最后一层

```python
loss_dict['loss_cls'] = losses_cls[-1]
loss_dict['loss_pts'] = losses_pts[-1]
loss_dict['loss_dir'] = losses_dir[-1]
loss_dict['loss_change'] = losses_changed_coords[-1]
loss_dict['loss_topo_cls'] = loss_topo_cls[-1]
loss_dict['loss_topo_dist'] = loss_topo_dist[-1]
loss_dict['loss_junction_cls'] = loss_junction_cls[-1]
loss_dict['loss_junction_pts'] = loss_junction_pts[-1]
loss_dict['loss_junction_dir'] = loss_junction_dir[-1]
loss_dict['loss_changed_coords_junction'] = losses_changed_coords_junction[-1]
loss_dict['lane_junction_type_loss'] = lane_junction_type_loss[-1]

# 子语义分类损失：按 class_metas 动态命名
losses_lane_cls_names = ['loss_lane_bound_type_cls', 'loss_lane_bound_color_cls', 'loss_road_bound_type_cls']
if 'road_mark_classes' in self.class_metas:
    losses_lane_cls_names.append('loss_road_mark_type_cls')
if 'fishbone_line_classes' in self.class_metas:
    loss_dict['loss_fishbone_change'] = losses_fishbone_changed_coords[-1]
    losses_lane_cls_names.append('loss_fishbone_type_cls')

for index, name in enumerate(losses_lane_cls_names):
    loss_dict[name] = losses_lane_cls[-1][index]
```

`losses_lane_cls[-1]` 是个 list（lane_bound type / color / road_bound type / road_mark / fishbone），按索引拆到不同名字下。

### 7.2 中间层

```python
num_dec_layer = 0
for loss_cls_i, loss_pts_i, loss_dir_i, loss_change_i in zip(
        losses_cls[:-1], losses_pts[:-1], losses_dir[:-1], losses_changed_coords[:-1]):
    loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
    loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
    ...
    num_dec_layer += 1
```

用 `[:-1]` 排除最后一层，中间层全部加 `d0.`/`d1.` 前缀（如 `d0.loss_cls`）。junction 的各损失同样处理。

### 7.3 guideline

```python
if self.with_guideline:
    loss_dict['loss_guideline_pts'] = loss_guideline_pts[-1]
    loss_dict['loss_guideline_dir'] = loss_guideline_dir[-1]
    # 中间层同样加 dN. 前缀
```

---

## 8. 流程总览

```text
loss(gt_vecs_list, preds_dicts, is_train_bev_static, ldmap_gt_flag)
│
├─ 1. 拆 preds_dicts → 各预测按 decoder 层组成 list
├─ 2. 拓扑预测只在末层，其余层置 None
├─ 3. 门控量 & GT 按层复制
├─ 4. multi_apply(loss_single) 逐层算损失
│      └─ loss_single: 匹配(target) → 门控 → 各类分类/回归/方向/变化点/拓扑损失
├─ 5. 可选: guideline 损失 (flag==2)
├─ 6. 可选: bev_seg 辅助损失
└─ 7. 组织 loss_dict
       ├─ 末层损失 → loss_xxx (参与 backward)
       ├─ 中间层 → d0.xxx / d1.xxx (监控)
       └─ 返回 dict
```

---

## 一句话总结

`loss()` 是**编排层**：把 forward 输出的逐层预测、逐层复制后的 GT 和门控量，用 `multi_apply` 喂给 `loss_single` 逐层算损失，再补上 guideline/分割辅助损失，最后把「最后一层损失作为正式 loss、中间层损失加 `dN.` 前缀」整理成 `loss_dict` 返回。


# 二, `loss_single` 处理流程详解

`loss_single` 是**单个 decoder 层**的损失计算函数，接收一个 decoder 层的全部预测和该 batch 的 GT，输出该层的各类损失。

> 位置：custom_lanehead_mapqr_v2.py:651-948

## 1. 函数签名与输入参数

def loss_single(self,
                pts_cls_scores,           # 点级分类分数 [bs, nv, (max_change_num+1)*cls_out_channels]
                inst_cls_scores,          # 实例级分类分数 [bs, nv, num_classes_map]
                road_mark_cls_scores,     # road mark 分类分数（可为 None）
                pts_preds,                # 线点坐标预测 [bs, nv, num_pts, 2]
                changed_preds,            # 变化点坐标预测
                fishbone_changed_preds,   # fishbone 变化点预测（可为 None）
                topology_preds,           # 拓扑预测（可为 None，末层才有）
                gt_shifts_pts_list,       # GT 点坐标 [每样本 n, shifts, fixed_num, 2]
                is_train_bev_static,      # [bs] 静态 BEV 是否训练
                pts_junction_cls_scores,  # junction 点级分类
                inst_junction_cls_scores, # junction 实例分类
                pts_junction_preds,       # junction 点坐标
                changed_junction_preds,   # junction 变化点
                gt_vecs_list=None,        # GT 完整容器（含各类标签）
                ldmap_gt_flag=None,       # [bs] 0=普通线/1=junction/2=guideline
                ):

## 2. 步骤一：batch 拆成逐图 list

num_imgs = pts_cls_scores.size(0)
num_pts_per_vec = pts_preds.size(-2)
pts_cls_scores_list = [pts_cls_scores[i] for i in range(num_imgs)]
inst_cls_scores_list = [inst_cls_scores[i] for i in range(num_imgs)]

把 [bs, ...] 的预测按 batch 维拆成 num_imgs 个元素，因为后续 get_targets 内部用 multi_apply 逐图做目标分配。同时构造 is_junction_list = [False]*num_imgs（第一趟普通线分支）。

## 3. 步骤二：两次 get_targets

### 3.1 普通线分支（is_junction=False）
cls_reg_targets = self.get_targets(
    pts_cls_scores_list, inst_cls_scores_list, road_mark_cls_scores_list,
    pts_preds_list, changed_preds_list, topology_preds_list,
    gt_shifts_pts_list, is_train_bev_static_list, ldmap_gt_flag_list,
    gt_vecs_list, is_junction_list)

### 3.2 junction 分支（is_junction=True）
is_junction_list = [True]*num_imgs
cls_reg_junction_targets = self.get_targets(
    pts_junction_cls_scores_list, ..., [None]*num_imgs,  # 拓扑传 None
    ..., is_junction_list)

两次区别：喂不同的预测（普通线 head vs junction head），is_junction 开关不同（决定匹配时筛哪些 GT 实例），拓扑预测 junction 分支传 None。

get_targets → _get_target_single 内部完成：assigner 匹配 → 生成各类 target。

## 4. 步骤三：权重门控调制

普通线分支只保留 ldmap_gt_flag==0 的样本，junction 分支只保留 ==1：
label_weights_list[idx] = label_weights_list[idx] * is_train_bev_static[idx] * (ldmap_gt_flag[idx]==0).float() + 1e-10

关键：+1e-10 避免纯 0 造成除零/梯度问题。

## 5. 步骤四：拼接 target
labels = torch.cat(labels_list, 0)
...（各类 target 拼成整 batch）

## 6. 步骤五：拆分分类分数到各语义类别
按 class_metas 把 pts_cls_scores 的最后一维切出 lane_bound_classes / lane_bound_colors / road_bound_classes / fishbone_line_classes 等；junction 同理用 class_metas_junction。

## 7. 步骤六：计算 avg_factor
cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
if self.sync_cls_avg_factor:
    cls_avg_factor = reduce_mean(...)  # 跨卡同步
cls_avg_factor = max(cls_avg_factor, 1)

## 8. 步骤七：计算各类损失

### 8.1 实例级分类 loss_cls / loss_junction_cls
### 8.2 子语义分类 losses_lane_cls（lane_bound/color/road_bound/road_mark/fishbone）
### 8.3 fishbone 损失（可选）
### 8.4 变化点回归 losses_changed_coords
### 8.5 junction 类型（bitwise 多标签）lane_junction_type_loss
### 8.6 拓扑损失 loss_topo_cls / loss_topo_dist（仅末层）
### 8.7 点回归 + 方向 loss_pts / loss_dir / loss_junction_pts / loss_junction_dir

## 9. 步骤八：NaN 清理与返回
torch.nan_to_num 处理所有损失，返回 13 项损失。

## 10. 流程总览

loss_single(...)
├─ 1. batch 拆成逐图 list
├─ 2. 两次 get_targets（普通线 / junction）
├─ 3. 权重门控（flag==0 / flag==1）
├─ 4. 拼接 target
├─ 5. 拆分 cls_scores
├─ 6. 计算 avg_factor
├─ 7. 逐项算损失
├─ 8. NaN 清理
└─ 9. 返回 13 项损失

## 一句话总结
loss_single 对单个 decoder 层：先拆 batch → 分普通线/junction 两趟做 assigner 匹配生成 target → 用 ldmap_gt_flag/is_train_bev_static 门控各分支学哪些样本 → 拼接 target → 拆分分类分数 → 逐项计算实例分类、子语义分类、变化点、拓扑、点回归、方向等损失 → NaN 清理后返回。


# 三, `get_targets` 与 `_get_target_single` 详解

这两个函数负责**从预测 + GT 生成训练用的 target（标签、坐标、权重）**。`get_targets` 是 batch 级包装器，`_get_target_single` 是单图的核心实现。

## 一、`get_targets`

### 作用

批处理包装器：对 batch 里每张图调用 `_get_target_single`，然后汇总正负样本数。

### 实现

```python
def get_targets(self, pts_cls_scores_list, inst_cls_scores_list, road_mark_cls_scores_list,
                pts_preds_list, changed_preds_list, topology_preds_list,
                gt_shifts_pts_list, is_train_bev_static_list,
                ldmap_gt_flag_list=None, gt_lanes_list=None, is_junction_list=None):
    # 逐图调用 _get_target_single
    (labels_list, label_weights_list, pts_targets_list, pts_weights_list,
     pos_inds_list, neg_inds_list, lane_targets_list) = multi_apply(
        self._get_target_single,
        pts_cls_scores_list, inst_cls_scores_list, road_mark_cls_scores_list,
        pts_preds_list, changed_preds_list, topology_preds_list,
        gt_shifts_pts_list, is_train_bev_static_list,
        ldmap_gt_flag_list, gt_lanes_list, is_junction_list)

    # 汇总正负样本数（用于 avg_factor）
    num_total_pos = sum((inds.numel() for inds in pos_inds_list))
    num_total_neg = sum((inds.numel() for inds in neg_inds_list))

    return (labels_list, label_weights_list, pts_targets_list, pts_weights_list,
            num_total_pos, num_total_neg, lane_targets_list)
```

**要点：**

- `multi_apply` 把各 list 的第 i 个元素（第 i 张图）一起传给 `_get_target_single`；
- 输入的形参名带 `_list` 后缀，都是「每图一个元素」的 list；
- `num_total_pos/neg` 是**整个 batch** 的正/负样本总数，之后用于算 `cls_avg_factor`。

## 二、`_get_target_single`

单张图的 target 生成，是整条 loss 链路的「数据准备」核心。

### 1. 读取基础信息

```python
device = pts_pred.device
num_preds = pts_pred.size(0)          # query 数（num_vec）
num_gts = gt_shifts_pts.shape[0]      # GT 实例数
pts_pred = pts_pred.contiguous()
inst_labels = gt_lanes.inst_labels    # 每个 GT 的类别号
```

### 2. 用 `junction_flag_mask` 筛选 GT 实例

```python
junction_flag_mask = gt_shifts_pts.new_full((num_gts,), 1, dtype=torch.bool)
gt_labels = gt_shifts_pts.new_full((num_gts,), 1, dtype=torch.long)
for i in range(gt_labels.shape[0]):
    if inst_labels[i] != gt_lanes.junction_inst_label:
        gt_labels[i] = inst_labels[i]
        junction_flag_mask[i] = True if not is_junction else False
    else:
        junction_flag_mask[i] = False if not is_junction else True
    if inst_labels[i] == 7:           # 类别 7 两边都剔除
        junction_flag_mask[i] = False

gt_labels = gt_labels[junction_flag_mask]
selected_gt_shifts_pts = gt_shifts_pts[junction_flag_mask, ...]
```

- `is_junction=False` → 只保留**非 junction** GT 参与匹配；
- `is_junction=True` → 只保留 **junction** GT。

### 3. assigner 匹配

```python
assign_result, order_index = self.assigner.assign(inst_cls_score, pts_pred,
                                                  selected_gt_shifts_pts, gt_labels)
```

匈牙利匹配，返回 `assign_result`（含 `gt_inds`）和 `order_index`（最优 shift 版本）。

### 4. 提取正负样本索引

```python
pos_inds = (assign_result.gt_inds > 0).nonzero(as_tuple=False).squeeze(-1).unique()
neg_inds = (assign_result.gt_inds == 0).nonzero(as_tuple=False).squeeze(-1).unique()
pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1   # 转 0-based GT 索引
```

### 5. roadmark 索引

```python
roadmark_inds = []
for i in range(len(assign_result.gt_inds)):
    if assign_result.gt_inds[i] > 0 and gt_labels[assign_result.gt_inds[i]-1] == 5:
        roadmark_inds.append(i)
```

找出匹配到 **road mark（label 5）** 的 query，用于给 road_mark 分类单独赋权重。

### 6. 实例分类 target

```python
labels = gt_shifts_pts.new_full((num_preds,), self.num_classes, dtype=torch.long)
labels[pos_inds] = gt_labels[pos_assigned_gt_inds]   # 匹配到的填 GT 类别
label_weights = gt_shifts_pts.new_ones(num_preds)
```

### 7. ignore_mask 处理

```python
ignore_mask = gt_lanes.ignore_mask
if ignore_mask is not None:
    h, w = ignore_mask.shape
    centers = (pts_pred.mean(dim=1) * torch.tensor([w, h]).to(pts_pred)).long()
    mask = (ignore_mask[centers[:,1], centers[:,0]] == 255)
    label_weights[mask] = 0   # 中心落在忽略区域的 query 权重置 0
```

### 8. 取最优 shift

```python
if order_index is None:
    assigned_shift = gt_labels[pos_assigned_gt_inds]
else:
    assigned_shift = order_index[pos_inds, pos_assigned_gt_inds]   # 每个匹配对的最优起点版本
```

### 9. 点回归 target

```python
pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
pts_weights = torch.zeros_like(pts_targets)
pts_weights[pos_inds] = 1.0
pts_targets[pos_inds] = selected_gt_shifts_pts[pos_assigned_gt_inds, assigned_shift]
```

用匹配到的最优 shift 版本的 GT 点坐标作为回归 target。

### 10. 语义类别 target

从 `gt_lanes` 取各类标签，按 `junction_flag_mask` 和 `pos_assigned_gt_inds` 对齐到 query：

- `lane_bound_types` / `lane_bound_colors` / `road_bound_types`：`[num_preds, max_change_num+1]`，默认填 `num_classes`（或 `num_classes_junction`）；
- `changed_coords_labels`：变化点坐标，`[num_preds, max_change_num]`，无效为 0；
- `road_mark_type_labels` + `road_mark_label_weights`（只有 roadmark_inds 处权重为 1）；
- `fishbone_*`（可选）：fishbone 类型和变化点。

### 11. 组装 lane_targets 字典

```python
lane_targets = dict(
    lane_bound_type_labels=..., lane_bound_color_labels=..., road_bound_type_labels=...,
    changed_coords_labels=..., road_mark_type_labels=...,
    type_label_weights=..., changed_coords_label_weights=..., road_mark_label_weights=...,
    pos_inds=..., fishbone_type_labels=..., fishbone_changed_coords_labels=...,
    fishbone_changed_coords_label_weights=...)
```

### 12. 拓扑 target

仅 `topo_preds is not None` 时生成：

- 有 GT 拓扑（`gt_lanes.topo_matrix` / `topo_dist_map` 不为 None）：按 pos_inds 对齐生成 `topo_cls_targets`（连通关系）和 `topo_dist_targets`（相对距离），并给距离权重 `==1`/`==2` 的（后继/前驱）加重到 5；
- 无 GT（0 实例）：全零占位。

### 13. 按 ldmap_gt_flag 清空正样本

```python
if not is_train_bev_static or \
   (ldmap_gt_flag == 1 and not is_junction) or \
   (ldmap_gt_flag == 0 and is_junction):
    neg_inds = torch.cat((pos_inds, neg_inds)).unique()
    pos_inds = torch.tensor([]).to(neg_inds)
```

当「图的真实类型」与「当前分支」不匹配，或该图不参与静态训练时，把正样本清空（全部归为负样本）。

### 14. 返回

```python
return (labels, label_weights, pts_targets, pts_weights,
        pos_inds, neg_inds, lane_targets)
```

## 三、两个函数的关系总览

```text
get_targets（batch 级）
│
├─ multi_apply 逐图调用 _get_target_single
│     └─ _get_target_single（单图）
│           ├─ 读 GT → junction_flag_mask 筛选实例
│           ├─ assigner.assign 匹配
│           ├─ 提取 pos/neg/roadmark 索引
│           ├─ 生成 labels / pts_targets / 各类语义 target / 拓扑 target
│           └─ 按 ldmap_gt_flag 清空不匹配分支的正样本
├─ 汇总 num_total_pos / num_total_neg
└─ 返回 7 项结果
```

## 一句话总结

`get_targets` 用 `multi_apply` 把单图 target 生成任务分发给 `_get_target_single`，并汇总正负样本数；`_get_target_single` 则完成「**筛选 GT 实例 → assigner 匹配 → 按匹配结果把各类 GT 标签对齐到 query → 生成 target 和权重**」的全过程，是整个 loss 计算的 target 源头。


# 四, `assigner.assign` 计算流程解析 (匈牙利匹配)

`assigner.assign` 是 **MapTR 风格的匈牙利匹配**，核心思想：给每个 GT 找一个（且只有一个）最匹配的预测 query，其余 query 当背景。

> 实现位置：[maptr_assigner.py:230-299](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/mmdet3d/core/bbox/assigners/maptr_assigner.py#L230-L299)

---

## 1. 调用时的参数对应

```python
self.assigner.assign(inst_cls_score, pts_pred, selected_gt_shifts_pts, gt_labels)
#                     ↓              ↓         ↓                     ↓
#                   cls_pred      pts_pred   gt_pts              gt_labels
```

|形参|含义|shape|
|---|---|---|
|`cls_pred`|预测的实例类别 logits|`[num_preds, num_classes]`|
|`pts_pred`|预测的线点坐标|`[num_preds, num_pts_per_line, 2]`|
|`gt_pts`|GT 点坐标（带 shift 维度）|`[num_gts, num_shifts, num_pts_per_gtline, 2]`|
|`gt_labels`|GT 实例标签|`[num_gts]`|

---

## 2. 计算步骤

### 步骤 1：初始化

所有 query 的 `assigned_gt_inds` 先置 `-1`。若 GT 或预测为空，直接返回全背景（`0`）。

```python
assigned_gt_inds = pts_pred.new_full((num_preds,), -1, dtype=torch.long)
if num_gts == 0 or num_preds == 0:
    if num_gts == 0:
        assigned_gt_inds[:] = 0
    return AssignResult(...), None
```

### 步骤 2：算匹配代价 `cost`

目标是得到一个 `[num_preds, num_gts]` 的代价矩阵，元素 =「这个 query 匹配这个 GT 要花多大代价」。代价由两项相加：

#### (a) 分类代价 `cls_cost`

用 `ClassificationCost`（focal loss 形式）计算：预测对 GT 类别给出的概率越低，代价越大。

```python
cls_cost = self.cls_cost(cls_pred, gt_labels)
```

#### (b) 点距离代价 `pts_cost`（核心）

```python
normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)   # GT 坐标归一化到 [0,1]
pts_pred_interpolated = ...                                    # 点数不一致时插值对齐
pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_pts)  # Chamfer 距离
pts_cost_ordered = pts_cost_ordered.view(num_preds, num_gts, num_orders)
pts_cost, order_index = torch.min(pts_cost_ordered, 2)         # 对 num_shifts 取最小
```

- `self.pts_cost` 是 **Chamfer Distance**：把两条线当成两个点集，计算「source 每个点到 target 最近点的距离」+「target 每个点到 source 最近点的距离」双向求和。它衡量的是**两条线作为点集靠得多近**，对点的顺序天然不敏感。
- 关键在 `view(..., num_orders)` 后 `torch.min(..., dim=2)`：对每个 `(pred, gt)` 组合，在 `num_shifts` 个起点版本里**挑距离最小的那个**，同时记录是哪个 shift（`order_index`）——这就是「换起点对齐」的落地实现。

最后合并：

```python
cost = cls_cost + pts_cost   # [num_preds, num_gts]
```

### 步骤 3：匈牙利匹配

```python
cost = cost.detach().cpu()
matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
```

用 scipy 的 `linear_sum_assignment`（匈牙利算法）在代价矩阵上求**总代价最小的一对一匹配**：

- 一对一：每个 GT 只能配一个 query，每个 query 只能配一个 GT；
- `matched_row_inds` = 被选中的 query 索引；
- `matched_col_inds` = 对应的 GT 索引。

### 步骤 4：写回匹配结果

```python
assigned_gt_inds[:] = 0                                      # 默认全当背景
assigned_gt_inds[matched_row_inds] = matched_col_inds + 1    # 匹配上的标成 1-based GT 号
assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
```

**返回值：**

1. `AssignResult`：包含
    - `assigned_gt_inds`（`0` = 背景，`>0` = 匹配到的 GT 编号，1-based）
    - `assigned_labels`（匹配到的 GT 类别）
2. `order_index`：每个 query 若匹配成功，用的是 GT 的**哪个 shift 版本**。

---

## 3. 回到 loss 里的后续处理

拿到 `assign_result` 后（[custom_lanehead_mapqr_v2.py:460-465](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L460-L465)）：

```python
pos_inds = (assign_result.gt_inds > 0).nonzero(...)   # 匹配上的 query = 正样本
neg_inds = (assign_result.gt_inds == 0).nonzero(...)  # 没匹配的 = 负样本
pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1  # 转回 0-based 的 GT 索引
```

`order_index` 后续用于取出匹配到的最优 shift：

```python
assigned_shift = order_index[pos_inds, pos_assigned_gt_inds]
pts_targets[pos_inds] = selected_gt_shifts_pts[pos_assigned_gt_inds, assigned_shift]
```

---

## 一句话总结

`assigner.assign` = **「分类代价 + Chamfer 点距离代价」拼成一个代价矩阵 → 对 `num_shifts` 个起点版本取最小距离（并记住最优起点）→ 匈牙利算法做一对一最优匹配 → 输出每个 query 是正样本（匹配哪个 GT）还是负样本（背景）**。它是决定「哪个预测去学哪条 GT 线」的核心环节