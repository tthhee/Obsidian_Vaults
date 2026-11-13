下面把 `map_head` 里 `query_embed_type == 'instance_pts_v2'` 这条路径的 query 构建过程完整梳理一遍。这是一个**两层/两阶段 query**设计：**instance 级内容 query + point 级几何(参考点)**。

## 一、相关维度（按当前配置）

| 符号 | 值 | 来源 |
|---|---|---|
| `num_vec` | 30 | 车道线实例数 [bsl_aug.py:99](configs/psd-bevlane-j6/v035_3_j6/bsl_aug.py#L99) |
| `num_junction_vec` | 4 | [bsl_aug.py:100](configs/psd-bevlane-j6/v035_3_j6/bsl_aug.py#L100) |
| `num_pts_per_vec` | 20 | `fixed_ptsnum_per_pred_line` |
| `num_query` | 600 = 30×20 | `num_vec * num_pts_per_vec` |
| `num_junction_query` | 80 = 4×20 | |
| `embed_dims` | 128 | `numC_LaneFormer = numC_Head` |

## 二、两个 Embedding 的初始化

在 [custom_lanehead_mapqr.py:170-175](mmdet3d/models/dense_heads/custom_lanehead_mapqr.py#L170-L175)：

```python
self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)      # (30, 128)
self.pts_embedding     = nn.Embedding(self.num_query, self.embed_dims)    # (600, 128)
self.junction_embedding    = nn.Embedding(self.num_junction_vec, self.embed_dims)  # (4, 128)
self.junction_pts_embedding= nn.Embedding(self.num_junction_query, self.embed_dims) # (80, 128)
```

注意关键区别：`instance_embedding` 只有 **30 个**（每条车道线一个），而 `pts_embedding` 有 **600 个**（30 条线 × 每条 20 个点）。

## 三、Transformer 入口：把两个 Embedding 变成 query / reference_points

在 [transformer_decoder.py:299-336](mmdet3d/plugins/transformer_decoder.py#L299-L336)，`BEVLanePerceptionTransformerMapQR.forward`：

```python
pos_embeds = pos_embeds.unsqueeze(0).expand(bs, -1, -1)   # (bs, 600, 128)
inst_embeds = inst_embeds.unsqueeze(0).expand(bs, -1, -1) # (bs, 30, 128)

reference_points = self.reference_points(pos_embeds)      # (bs, 600, 2)
reference_points = self.hs(reference_points)              # hard sigmoid → [0,1]

query     = inst_embeds.permute(1, 0, 2)   # (30,  bs, 128)  ← instance 级内容 query
query_pos = pos_embeds.permute(1, 0, 2)   # (600, bs, 128)  ← point 级
```

核心结论：

- **`query` = `instance_embedding`**，形状 `(num_vec=30, bs, 128)`，每条车道线一个可学习内容向量。
- **`reference_points` = `MLP(pts_embedding)`**，形状 `(bs, 600, 2)`，即 30 条线 × 20 个点的**初始 2D 归一化坐标**（经过 `Linear(128→128)→ReLU→Linear(128→2)` + hard sigmoid）。这就是"instance_pts"里 pts 的含义——初始几何先验。
- `query_pos`（600 个 pts_embedding 本体）传进 decoder，但在 layer 内**实际没被用到**（下面会说明真正的点位置编码是 `query_pos_sampler` 从参考点坐标现算的）。

## 四、Decoder Layer：scatter → cross_attn → gather

这是 MapQR 的核心机制，在 [decoder.py:1286-1311](mmdet3d/plugins/decoder.py#L1286-L1311) 的 `cross_attn` 分支：

```python
nv, bs, ed = query.shape                    # (30, bs, 128)，instance 级
reference_points = kwargs['reference_points'][:,:,0,:].permute(1,0,2)  # (600, bs, 2)
pts_query_pos = self.query_pos_sampler(reference_points)  # (600, bs, 128) 点位置编码

# ① scatter：把每条 instance query 复制到它的 20 个点上
query = query.unsqueeze(1).expand(nv, nppv, bs, ed).view(nv*nppv, bs, ed)  # (600, bs, 128)
query = pts_query_pos + query               # 加上点位置编码

# ② cross_attn：600 个点 query 用可变形注意力去 BEV 上采样
query = self.attentions[attn_index](query, key, value, ..., reference_points=...)

# ③ gather：把 20 个点的结果拼回来，压缩回 instance 级
query = query.view(nv, nppv, bs, ed).permute(0,2,1,3).view(nv, bs, nppv*ed)  # (30, bs, 2560)
query = self.gather(query)                  # Linear(20*128→128) → (30, bs, 128)
```

几个关键点：

1. **self_attn 在 instance 级**（[decoder.py:1266-1280](mmdet3d/plugins/decoder.py#L1266-L1280)）：30 条线之间做自注意力，交换拓扑/上下文信息。
2. **cross_attn 在 point 级**：600 个点各自用自己的 reference_point 做可变形采样（`num_points=4, num_levels=1`）。
3. **点位置编码 `pts_query_pos` 是动态的**：由 `query_pos_sampler(Linear(2→128))` 从当前 reference_point 坐标在线生成，所以每层 refine 后点位置编码会跟着参考点一起变，而不是固定的 `pts_embedding`。
4. **gather** 用 `Linear(embed_dims×20 → embed_dims)` 把 20 个点特征压回一个 instance 向量，保持 decoder 全程 `(30, bs, 128)` 的 instance 级表示。

## 五、参考点迭代更新（box refine）

每层 decoder 之后，[decoder.py:1002-1008](mmdet3d/plugins/decoder.py#L1002-L1008)：

```python
tmp = reg_branches[lid](output)                    # (bs, 30, 40=20*2)
tmp = 2*self.hs(tmp.view(bs, nv, ed//2, 2).view(bs,-1,2)) - 1.0   # 残差 [-1,1]
new_reference_points = torch.clamp(tmp + reference_points, 0.0, 1.0)
reference_points = new_reference_points.detach()   # 更新并 detach，供下一层使用
inter_pts_coord = new_reference_points.view(bs, nv, -1, 2)  # (bs, 30, 20, 2)
```

即 reg 分支输出的是**相对参考点的残差**（`2·hardsigmoid-1` 映射到 `[-1,1]`），逐层累加到参考点上。最终 `outputs_pts_coords` 形状为 `(num_layers, bs, 30, 20, 2)`。

## 六、最终输出

head 的 `forward`（[custom_lanehead_mapqr_v2.py:304-306](mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L304-L306)）把输出 reshape 成：

- `outputs_pts_coords` → `(nl, bs, num_vec=30, num_pts_per_vec=20, 2)` 车道线点坐标
- `outputs_inst_classes` → `(nl, bs, 30, num_classes_map)` 实例前景/背景
- `outputs_pts_classes` → `(nl, bs, 30, max_change_num+1, cls_out_channels)` 逐点类别

## 小结（一张图）

```
pts_embedding (600,128) ──MLP+hs──> reference_points (bs,600,2)  [初始几何]
instance_embedding (30,128) ───────> query (30,bs,128)           [实例内容]

每层 MapQRDecoderLayer:
  self_attn  : 30 条线互相 attention
  cross_attn : scatter(30→600) → +query_pos_sampler(ref_pt) → 可变形attn → gather(600→30)
  reg refine : reg_branch → 残差更新 reference_points
```

和你上一轮问的 `sampling_offsets` 衔接起来：cross_attn 里那 600 个点 query，每个都会预测 `H·L·P·2 = 8×1×4×2` 的采样偏移，采样点围绕各自的 `reference_point`（20 个点的归一化坐标）展开。


# transformer layer输出格式与解析
`transformer`（`BEVLanePerceptionTransformerMapQR`）最终返回的是 `MapQRDecoder` 的输出。因为配置里 `aux_topo.use_aux_topo=True`、`return_intermediate=True`，所以它返回一个 **7 元组**，head 在 [custom_lanehead_mapqr_v2.py:246-250](mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L246-L250) 里这样解包：

```python
outputs_pts_classes, outputs_inst_classes, outputs_pts_coords, \
outputs_changed_coords, outputs_fishbone_changed_coords, \
outputs_road_mark_classes, outputs_topology = outputs
```

## 一、返回元组各元素的格式

设 `nl=num_decoder_layer=2`、`bs`、`num_vec=30`、`nppv=num_pts_per_vec=20`、`max_change_num=2`。

| 元素 | 形状（stack 后） | 含义 | 层级 |
|---|---|---|---|
| `outputs_pts_classes` | `(nl, bs, 30, (max_change_num+1)·cls_out_channels)` | 逐段属性分类 logits | segment 级 |
| `outputs_inst_classes` | `(nl, bs, 30, num_classes_map=6)` | 每条线的实例分类 logits | **instance 级** |
| `outputs_pts_coords` | `(nl, bs, 30, 20, 2)` | 每条线 20 个点的坐标 | **point 级** |
| `outputs_changed_coords` | `(nl, bs, 30, 2)` | 变化点（分段点）坐标 | instance 级 |
| `outputs_fishbone_changed_coords` | `(nl, bs, 30, 2)` | 鱼骨线变化点 | instance 级 |
| `outputs_road_mark_classes` | `(nl, bs, 30, num_road_mark_classes)` | 路面标记分类 | instance 级 |
| `outputs_topology` | `(bs, 30, 30, topo_out_channels=4)` | 线-线拓扑关系 | instance 对 |

> `outputs_inst_classes` 的 channel 是 `num_classes_map = len(classes_bev_static)+1 = 6`（lane/road/stopline/crosswalk/roadmark + 背景），由 [custom_lanehead_mapqr.py:72](mmdet3d/models/dense_heads/custom_lanehead_mapqr.py#L72) `Linear(embed_dims, num_classes_map)` 决定。

## 二、head 里的 reshape（[custom_lanehead_mapqr_v2.py:304-306](mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L304-L306)）

`outputs_pts_classes` 刚出来是打包过的，head 再拆成段维度：

```python
nl, bs = outputs_pts_coords.shape[:2]          # 从 (nl, bs, 30, 20, 2) 取前两维
outputs_pts_classes = outputs_pts_classes.view(
    nl, bs, self.num_vec, self.max_change_num+1, self.cls_out_channels)
# → (nl, bs, 30, 3, cls_out_channels)
```

也就是每条线的属性分类里有 `max_change_num+1=3` 个"段"（一条线最多被 2 个变化点切成 3 段，如实线→虚线→实线），每段一份 class logits。

## 三、如何解析出 instance 和 pts

**instance 和 pts 是天然按 index 对齐的**——第 `i` 条线（`i∈[0,30)`）的实例分类和它的 20 个点用同一个下标 `i` 取：

```python
# 取最后一层（通常用 [-1] 做最终预测）
inst_cls   = outputs_inst_classes[-1]   # (bs, 30, 6)   每条线的实例类别 logits
pts_coords = outputs_pts_coords[-1]     # (bs, 30, 20, 2) 每条线的点坐标
pts_cls    = outputs_pts_classes[-1]    # (bs, 30, 3, cls_out_channels)

# 对 batch b、第 i 条线：
inst_cls[b, i]    # shape (6,)  → argmax 得到该实例的前景/地图类别
pts_coords[b, i]  # shape (20, 2) → 这条线的 20 个点（此时还是归一化 [0,1] 坐标）
pts_cls[b, i]     # shape (3, cls_out_channels) → 3 段的属性类别
```

真正的"解析成最终输出"是在 coder 里做的，`PhiGentLaneCoderV3.decode`（[phigent_lane_coder.py:814-879](mmdet3d/core/bbox/coders/phigent_lane_coder.py#L814-L879)）取的都是**最后一层** `[-1]`：

1. **instance**：`inst_cls_scores.sigmoid()` → `scores, labels = max(dim=1)`，`labels` 就是每条线的实例类别（lane/road/stopline/…）。
2. **pts**：`pts_preds` 过 `denormalize_2d_pts(pts_preds, pc_range)`，把归一化坐标还原成物理米制坐标，`final_pts_preds['lines']` 就是 `(num_vec, 20, 2)` 的物理点。
3. 再按 score 排序、`score_threshold` 过滤、`post_center_range` 裁剪，得到最终 `predictions_dict = {scores, labels, pts, topo}`。

## 四、和 transformer 内部输出的对应关系

- **instance 内容**来自 `instance_embedding`，经 decoder 后输出 `(bs, 30, 128)` 的 instance 特征，再由 `inst_cls_branches` 投影成 `(bs, 30, 6)`。
- **pts 坐标**就是 decoder 里逐层 refine 的 `reference_points`（`(bs, 600, 2)`），在 [decoder.py:1008](mmdet3d/plugins/decoder.py#L1008) 里 `view(bs, nv, -1, 2)` 变成 `(bs, 30, 20, 2)`。
- **pts 属性分类**由 `cls_branches` 从 instance 特征投影出 `(bs, 30, (max_change_num+1)·cls_out_channels)`，head 再拆成 `(bs, 30, 3, cls_out_channels)`。

所以一句话总结：

> decoder 输出是「按层打包的 7 元组」；**instance 信息在 `outputs_inst_classes`（每条线一个 6 维 logits），pts 信息在 `outputs_pts_coords`（每条线 20 个 2D 点，归一化坐标）**，两者通过 `num_vec` 维度的同一个下标 `i` 对应；最终由 coder 取 `[-1]` 层做 sigmoid/denormalize 得到物理坐标和类别。


# 每一层transformer layer输出的output解析
这里的 `output` 是**每个 decoder layer 输出的 instance 级特征**。关键要理解：虽然 cross-attention 内部展开到了 600 个点，但在 [MapQRDecoderLayer](mmdet3d/plugins/decoder.py#L1308-L1309) 的最后又做了 `gather`，把 20 个点压缩回**一条线一个向量**，所以 layer 的最终输出就是 instance 级。

## 一、`output` 的 shape

[decoder.py:994-995](mmdet3d/plugins/decoder.py#L994-L995)：

```python
output = layer(...)                # (num_vec=30, bs, embed_dims=128)  ← layer 内最后一维是 gather 回的 instance
output = output.permute(1, 0, 2)   # (bs, 30, 128)
```

所以 `output` 是 **`(bs, 30, 128)`**：每个 batch、每条车道线实例一个 128 维特征向量。所有 branches 都消费这**同一个** instance 级向量，各自独立解析成不同输出。

## 二、各 branch 的定义与解析

每个 branch 都是一个小 MLP（`num_reg_fcs=2` 层 `Linear(128→128)+ReLU` + 一个输出 `Linear`），在 [custom_lanehead_mapqr.py:56-121](mmdet3d/models/dense_heads/custom_lanehead_mapqr.py#L56-L121) 和 [custom_lanehead_mapqr_v2.py:36-108](mmdet3d/models/dense_heads/custom_lanehead_mapqr_v2.py#L36-L108) 里定义：

| branch | 输出 Linear 维度 | 输出 shape | 解析结果 |
|---|---|---|---|
| `cls_branches[lid]` | `128 → cls_out_channels·(max_change_num+1)` | `(bs, 30, cls_out_channels·3)` | 段级属性分类，head 再 view 成 `(bs,30,3,cls_out_channels)` |
| `inst_cls_branches[lid]` | `128 → num_classes_map=6` | `(bs, 30, 6)` | 实例类别（lane/road/stopline/…） |
| `changed_reg_branches[lid]` | `128 → max_change_num=2` | `(bs, 30, 2)` | 变化点（分段点）坐标 |
| `fishbone_changed_reg_branches[lid]` | `128 → 2` | `(bs, 30, 2)` | 鱼骨线变化点 |
| `reg_branches[lid]` | `128 → code_size·num_pts_per_vec=40` | `(bs, 30, 40)` | 20 个点的坐标残差 |
| `road_mark_cls_branches[lid]` | `128 → num_road_mark_classes` | `(bs, 30, num_road_mark_classes)` | 路面标记类别 |

## 三、`reg_branches` 如何从 40 维"解析"出 20 个点

这是最值得注意的一处，[decoder.py:1002-1008](mmdet3d/plugins/decoder.py#L1002-L1008)：

```python
tmp = reg_branches[lid](output)                        # (bs, 30, 40)
bs, nv, ed = tmp.shape                                  # nv=30, ed=40
# 40 = 20 点 × 2 坐标，先按点拆开
tmp = 2*self.hs(tmp.view(bs, nv, ed//2, 2).view(bs, -1, 2)) - 1.0   # (bs, 600, 2)
new_reference_points = torch.clamp(tmp + reference_points, 0.0, 1.0)
reference_points = new_reference_points.detach()        # 更新参考点
inter_pts_coord = new_reference_points.view(bs, nv, -1, 2)  # (bs, 30, 20, 2)
```

解析步骤：

1. `Linear(128→40)` 一口气输出 30 条线 × 20 点 × 2 坐标 = 40 维。
2. `view(bs, 30, 20, 2)` 把 40 维拆成「20 个点，每点 (x,y)」。
3. `2·hardsigmoid(x) - 1` 把 logits 映射成 `[-1,1]` 的**残差**（不是绝对坐标）。
4. 残差 + 当前 `reference_points` → 新参考点，`clamp` 到 `[0,1]`。
5. `view(bs, 30, 20, 2)` 得到这条线的 20 个点坐标。

也就是说：**每个点不是单独一个 query 单独输出，而是由 instance 级向量通过一个线性层"一次性铺出"全部 20 个点**（类似 MapTR 的做法），点与点之间靠这个 MLP 的输出维度顺序隐式对齐。

## 四、完整数据流（每层）

```
layer 输出: (30, bs, 128)  instance 级
   │ permute
   ▼
output: (bs, 30, 128)
   ├─ inst_cls_branches   → (bs, 30, 6)          instance 类别
   ├─ cls_branches        → (bs, 30, C·3)        段级属性类别
   ├─ reg_branches        → (bs, 30, 40) ──view──→ (bs, 30, 20, 2) 点坐标
   ├─ changed_reg_branches→ (bs, 30, 2)          变化点
   └─ road_mark_cls_branches → (bs, 30, R)       路面标记
```

最后这些每层的结果用 `inter_*` 列表收集，`return_intermediate=True` 时 `torch.stack` 成 `(nl, bs, ...)` 返回——这就是上一问里 7 元组各元素的来源。