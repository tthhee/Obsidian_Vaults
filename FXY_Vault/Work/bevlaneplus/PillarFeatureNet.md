读完了 [pillarnet.py](projects/mmdet3d_plugin/models/backbones/pillarnet.py)。这里实现的是 **PointPillars** 的 pillar 特征编码流程，核心是 `PillarFeatureNet`（[pillarnet.py:270](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L270)），整体分三步：**体素化 → 特征增强 → MLP + max 池化**。

## 整体入口

`forward`（[pillarnet.py:309](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L309)）串联两个阶段：

```python
voxels, coors, num_points = self.forward_preprocess(points)   # 阶段1：体素化
features, coors, grid_sizes = self.forward_encoder(...)        # 阶段2：特征聚合
```

输入 `points` 是 `(B, N, 4)`，每个点为 `(x, y, z, intensity)`。

## 阶段 1：点云体素化（pillar 划分）

`PointCloudVoxelization`（[pillarnet.py:18](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L18)）调用 mmcv 的 `Voxelization`，逐 batch 处理（[pillarnet.py:52-53](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L52-L53)），把散乱点云划分成一个个 pillar（柱体，即 z 轴不做划分的 voxel）：

- `voxel`: `(M, max_points, 4)` —— 每个 pillar 内的点（不足补零，超过截断）
- `coor`: `(M, 4)` —— pillar 坐标，格式 `(batch_idx, z, y, x)`
- `num_point`: `(M,)` —— 每个 pillar 实际含有的点数

这里做了两步后处理保证形状固定（方便后续部署/量化）：
- **超过 `max_voxels` 时随机采样**：点数 ≥2 的算有效体素，优先保留；有效体素不足时再从无效体素里随机补齐（[pillarnet.py:55-72](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L55-L72)）。
- **不足 `max_voxels` 时补零 padding**（[pillarnet.py:75-83](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L75-L83)）。

最后把各 batch 结果 concat，并在 `coors` 第一列填上 batch 序号（[pillarnet.py:87](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L87)）。

## 阶段 2：特征增强（FeatAggregation）

`FeatAggregation.forward`（[pillarnet.py:205](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L205)）是核心，对每个 pillar 内的原始 4 维特征扩充出 5 维几何特征，得到 9 维：

1. **相对 pillar 中心的偏移 `f_center`（2 维）**
   - 先算 pillar 中心：`(coor + 0.5) * voxel_size + point_cloud_range`（[pillarnet.py:207-208](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L207-L208)）
   - 每个点减去中心，得到 `(Δx_c, Δy_c)`

2. **归一化原始特征**
   - x/y/z 分别除以点云范围对应维度的最大绝对值，intensity 除以 255（[pillarnet.py:216-219](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L216-L219)）

3. **相对 pillar 内点均值的偏移 `f_cluster`（3 维）**
   - 求 pillar 内所有点的均值 `points_mean`（[pillarnet.py:222-223](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L222-L223)）
   - 每个点减去均值，得到 `(Δx_m, Δy_m, Δz_m)`

4. **拼接 + mask**（[pillarnet.py:233-234](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L233-L234)）
   - `4（归一化后原始特征） + 3（f_cluster） + 2（f_center） = 9 维`，即 `num_input_features + 5`
   - 用 `mask`（`arange < num_points_per_voxel`）把 padding 的点全部置 0，避免填充点污染后续 max 池化

## 阶段 3：PFN 层 + max 池化

`num_filters` 默认 `[64, 64]`，前面拼上 9 维后变成 `[9, 64, 64]`，循环构造两层 `PFNLayer`（[pillarnet.py:188-201](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L188-L201)）。

`PFNLayer`（[pillarnet.py:97](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L97)）每层做：`Linear → BN1d → ReLU6`，然后**对 point 维度做 max 池化**：

- 非最后一层：把 max 后的结果 `repeat` 拼回每个点，实现类似 VFE 的逐点特征增强（[pillarnet.py:132-137](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L132-L137)）。
- 最后一层（`last_layer=True`）：只返回 max 结果，完成从「逐点特征」到「pillar 级特征」的聚合（[pillarnet.py:132-133](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L132-L133)）。

最终输出 `features` 形状 `(M, 64)`，即每个 pillar 一个 64 维特征向量。

## 输出整理

`forward_encoder`（[pillarnet.py:302-307](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L302-L307)）最后返回三个量：

- `features`: `(M, 64)` —— pillar 特征
- `coors`: `(M, 3)` —— 取 `(batch_idx, x, y)` 供 scatter 到 BEV 网格
- `grid_sizes`: `(H, W)` —— 由点云范围和 voxel_size 算出的 BEV 图尺寸

---

补充一点：文件里还有一套 **J5 量化版本**（`FeatAggregationJ5`、`PillarFeatureNetJ5`、`PFNLayerJ5`，[pillarnet.py:139](projects/mmdet3d_plugin/models/backbones/pillarnet.py#L139) 起），算法逻辑和上面完全一致，只是把 inplace 操作、切片、求和等换成了 `FloatFunctional` / `Div` 等地平线（Horizon J5/J6）量化友好的算子，方便导出 QAT 定点模型。如果你关心的是量化部署这条链路，我可以再展开讲两者算子的对应关系。