信息已经完整。下面解释这三个变量以及后续处理链路。

## 三个变量分别是什么

它们在数据侧由 [map.py:2440-2447](projects/mmdet3d_plugin/datasets/pipelines/map.py#L2440-L2447) 写入 `results`，本质是三个 dict：

| 变量 | 来源 dict | 内容 | 含义 |
| --- | --- | --- | --- |
| `nav_path_instances` | `gt_nav_path_dict` | `gt_map_pts`（路径点）+ `gt_map_instance_labels`（实例标签） | **导航路径 / 引导线**（guideline）：导航规划出的自车行驶路线，只有 `map_classes` 含 `"guideline"` 时才加载 |
| `sd_road_instances` | `gt_sd_road_dict` | `gt_map_pts`（道路点）+ `gt_sd_road_labels`（道路类别标签） | **SD 地图道路**（standard-definition map）：标准精度地图里的道路 polyline 元素 |
| `lane_nums` | `gt_lane_num_dict` | `gt_lane_num_labels` | **车道数量**：每条道路对应的车道数标签 |

从 [transform.py:135-139](projects/mmdet3d_plugin/datasets/pipelines/transform.py#L135-L139) 能直接看到每个 dict 里各 key 的取值方式：

```python
map_graph = sd_road_instances["gt_map_pts"]       # SD 道路坐标点
map_category = sd_road_instances["gt_sd_road_labels"]  # SD 道路类别
map_lane_nums = lane_nums["gt_lane_num_labels"]    # 车道数标签
nav_path = nav_path_instances["gt_map_pts"]       # 导航路径坐标点
nav_path_category = nav_path_instances["gt_map_instance_labels"]  # 导航路径实例标签
```

注意这些 dict 都是**多实例**结构：`gt_map_pts` 是一个列表，每个元素是一条 polyline 的采样点张量（在 loading 阶段已经统一采样到固定点数，见 [loading.py:3436-3457](projects/mmdet3d_plugin/datasets/pipelines/loading.py#L3436-L3457) 的 `num_sample` 采样逻辑）。

## 在 forward 里如何选择与处理

回到 [siamesebevlane_fisheye.py:347-366](projects/mmdet3d_plugin/models/siamesebevlane_fisheye.py#L347-L366)，逻辑分三步：

**1. 取数据并做非空校验**（[351-355](projects/mmdet3d_plugin/models/siamesebevlane_fisheye.py#L351-L355)）：只要 `map_encoder` 存在，`nav_path_instances` 和 `sd_road_instances` 二者缺一就报错。

**2. 按 `map_mode_info` 选择喂给 encoder 的 graph**（[356-361](projects/mmdet3d_plugin/models/siamesebevlane_fisheye.py#L356-L361)）：

- `map_mode_info == "only_nav"`：用**导航路径**作为 map graph
  - `map_graph = nav_path_instances["gt_map_pts"]`
  - `map_category = nav_path_instances["gt_map_instance_labels"]`
- 否则（默认）：用 **SD 道路**作为 map graph
  - `map_graph = sd_road_instances["gt_map_pts"]`
  - `map_category = sd_road_instances["gt_sd_road_labels"]`

即：encoder 里 `map_graph`/`map_category` 的具体语义取决于 `map_mode_info`，二者用的是同一套「点 + 类别」接口，只是切换了数据源。

**3. 取出车道数并调用 encoder**（[363-366](projects/mmdet3d_plugin/models/siamesebevlane_fisheye.py#L363-L366)）：

```python
if self.map_encoder.use_lane_num and lane_nums is not None:
    lane_nums = lane_nums["gt_lane_num_labels"]
map_feats = self.map_encoder(map_graph, map_category, lane_nums)
```

`map_encoder` 是通过 `build_neck` 构建的 `MapGraphTransformer`（注册在 [map_graph_encoder.py:106](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L106)），只有 `use_lane_num=True` 时车道数才会真正参与计算。

## map_encoder（MapGraphTransformer）内部如何消化

`forward(map_graph, onehot_category, lane_nums)`（[map_graph_encoder.py:231](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L231)）：

1. **reshape 成统一格式**（[236-249](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L236-L249)）：
   - 坐标点 `graph_polylines.view(-1, num_sample, 2)` —— 每条 polyline 固定 `num_sample` 个 2D 点
   - 类别标签 `.view(-1, num_sample, 1)`
   - 车道数 `.view(-1, 2, num_sample)`（2 个通道 × num_sample 采样点）

2. **坐标特征编码**（[266-286](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L266-L286)）：`pre_map_feat_fc` → 沿点维度 max 池化 → 池化结果 repeat 回每个点并 concat → flatten → `post_map_feat_fc`，得到每条 polyline 一个 `dmodel` 维向量 `[B, N, dmodel]`。

3. **类别特征（onehot）**（[293-304](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L293-L304)）：`build_onehot` 后经 `label_fc` 到 `dmodel` 维，**在序列维度 concat** 到坐标特征后面（相当于把每条 polyline 的坐标 token 和类别 token 都放进序列）。

4. **车道数特征**（[306-321](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L306-L321)）：`lane_num_fc` 到 `dmodel` 维，同样 concat 进序列。

5. **加 modality / positional embedding**（[323-336](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L323-L336)）：`modality_embedding` 区分「坐标/类别/车道数」三种 token 类型，`positional_embedding` 区分 polyline 序号。

6. **Transformer encoder**（[338-339](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L338-L339)）：多模态 token 序列过 `nlayers` 层 self-attention，输出 `map_feats` 作为地图的稀疏 token 特征，后续喂给 `forward_map_train` / decoder 做跨模态融合。

## 附带的 pos_fc 分支

[siamesebevlane_fisheye.py:367-377](projects/mmdet3d_plugin/models/siamesebevlane_fisheye.py#L367-L377)：当 `self.pos_fc` 存在时，会把**导航路径点**（`nav_path_instances["gt_map_pts"]`）用 `normalize_2d_pts(nav_pts, self.map_encoder.sd_range)` 归一化到 SD 范围后 concat，再过一个 `pos_fc` 得到 `map_embeds`（导航路径的 positional embedding），与 `map_feats` 一起作为 `map_embeds` 传给下游 head。

---

一句话总结：`nav_path_instances` 是**导航引导线**、`sd_road_instances` 是**SD 地图道路**、`lane_nums` 是**道路车道数**；模型里先按 `map_mode_info` 决定拿「导航路径」还是「SD 道路」当主图，再把坐标、类别、车道数三种模态各自编码成 `dmodel` 维 token 拼进序列，过 Transformer 得到地图稀疏特征 `map_feats`。

-------
# 类别标签和车道数 具体的表达方式

## 1. 类别标签（`gt_sd_road_labels`）

**它是逐点（per-point）的二分类标签，不是一条线一个类别复制 num_sample 次。**

生成处 [map.py:1244-1252](projects/mmdet3d_plugin/datasets/pipelines/map.py#L1244-L1252)：

```python
if not "is_routing_vertex" in sd_instance:
    sd_instance["types"] = [0] * shape_points.shape[0]
elif not sd_instance.get("is_routing_vertex", False):
    sd_instance["types"] = [0] * shape_points.shape[0]
else:
    sd_instance["types"] = [1] * shape_points.shape[0]
```

每个 shape point 对应一个类型值，`types` 数组长度 = 点数。config 注释也印证了 [bsl.py:522](projects/configs/bevlaneplus_v005/bsl.py#L522)：

> `num_road_types=2,  # notes: 2 for is_routing and not_routing`

所以语义是：
- `0` = 非路由顶点（普通道路点）
- `1` = 路由顶点（is_routing，路口/导航关键点）

重采样对齐在 [loading.py:3456-3458](projects/mmdet3d_plugin/datasets/pipelines/loading.py#L3456-L3458)：采样点用 `argmin` 找到最近邻原始点 `matched_idx`，再 `labels[matched_idx]` 继承标签。所以最终是**每个采样点一个标签**，形状 `(N, num_sample)`。

forward 里的 `label_category.view(-1, self.num_sample, 1)`（[map_graph_encoder.py:242](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L242)）**不是复制**，只是把 `(N, num_sample)` 补出最后一维 `1`，纯粹为了和坐标特征 `(N, num_sample, 2)` 在 concat 时维度对齐。标签本来就已经逐点存在了。

之后 `build_onehot`（[map_graph_encoder.py:215-229](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L215-L229)）再把 `(B, N, num_sample)` 的整数标签扩成 `(B, N, num_sample, num_road_types)` 的 one-hot，经 `label_fc` 压到 `dmodel` 维。

## 2. 车道数（`gt_lane_num_labels`）

**它不是逐点的，而是"每条道路两端各一个车道数标量"，用 one-hot 编码成固定长度向量。** 生成处 [map.py:1294-1321](projects/mmdet3d_plugin/datasets/pipelines/map.py#L1294-L1321)：

```python
head_lane_num_label = np.zeros(self.fixed_sd_road_num)
tail_lane_num_label = np.zeros(self.fixed_sd_road_num)
head_lane_num = ...["gt_sd_road_lanes_nums"][i][0]     # 道路起点端的车道数
tail_lane_num = ...["gt_sd_road_lanes_nums"][i][-1]    # 道路终点端的车道数
head_lane_num_label[head_lane_num - 1] = 1              # one-hot
tail_lane_num_label[tail_lane_num - 1] = 1
... np.hstack([head_lane_num_label, tail_lane_num_label])
```

拆解：

- 每条 SD road 的 `lanes_nums` 列表，取首元素（head）和末元素（tail）两个整数。
- 每个整数编码成**长度为 `fixed_sd_road_num` 的 one-hot 向量**：第 `lane_num - 1` 个位置置 1（所以车道数的有效取值范围必须 `< fixed_sd_road_num`）。
- head、tail 各一条 one-hot，`hstack` 成 `(2 * fixed_sd_road_num,)`。

因此 `gt_lane_num_labels` 整体形状是 `(N, 2 * num_sample)`（`num_sample = fixed_sd_road_num`，见 config [bsl.py:524](projects/configs/bevlaneplus_v005/bsl.py#L524) 与 [bsl.py:944](projects/configs/bevlaneplus_v005/bsl.py#L944) 都用 `fixed_ptsnum_per_gt_line`）。

forward 里的 `lane_num.view(-1, 2, self.num_sample)`（[map_graph_encoder.py:248](projects/mmdet3d_plugin/models/necks/map_graph_encoder.py#L248)）只是把扁平存储的 `(N, 2*num_sample)` 还原成 2 个通道。**这里的 `2` 是「head / tail」两端，不是「每个点」；`num_sample` 那一维是 one-hot 的编码长度，不是采样点**——它只是数值上恰好等于采样点数。

## 总结对比

| | 类别标签 | 车道数 |
| --- | --- | --- |
| 粒度 | **逐点**（每个采样点一个标签） | **逐实例**（每条道路一个标量 → one-hot） |
| 语义 | 0=普通点，1=路由顶点（routing vertex） | 道路两端（head/tail）各自的车道数量 |
| 原始形状 | `(N, num_sample)` | `(N, 2 * num_sample)`（2 个 one-hot 拼接） |
| reshape | `(-1, num_sample, 1)` 补最后一维对齐坐标 | `(-1, 2, num_sample)` 拆成 head/tail 两通道 |
| 是否"每个点复制" | 否——本来逐点就有标签 | 否——它是标量的 one-hot 编码，跟采样点无关 |

一个容易踩的坑：`lane_num.view(-1, 2, num_sample)` 里的 `num_sample` 维度看起来像"每个采样点一个值"，实际是 one-hot 编码长度（用 `np.zeros(fixed_sd_road_num)` 分配，数值上刚好等于采样点数）。真正"逐点"的只有类别标签。

