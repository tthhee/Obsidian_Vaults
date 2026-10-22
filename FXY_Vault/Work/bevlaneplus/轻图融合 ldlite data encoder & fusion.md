## 1. ldlite_data 的数据格式

`ldlite_data` 是 `LDLiteLoader`（pipeline）产出的一个 **dict**，里面是三个**栅格化 BEV 属性图**（numpy int 数组），来源是把 LDLite 地图数据从 `.pkl` 解析出来后逐格 rasterize 而成。格式见 [ldlite_loader.py:5-11](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L5-L11)：

|key|形状|通道语义|
|---|---|---|
|`section_attr_maps`|`(H, W, 6)`|`[section_class, section_type, section_direction, positive_slope, toll_type, nav_path]`|
|`lane_attr_maps`|`(H, W, 5)`|`[lane_type, turn_type, trans_type, link_id, link_type]`|
|`semantic_attr_maps`|`(H, W, 6)`|`[traffic_element, sign, junction, boundary_type, boundary_style, boundary_color]`|

- **H、W** 由 `pc_range`（默认 `[-150,-150,0,150,150,0]`）和 `resolution`（默认 1.5m）决定，默认是 `(200, 200)`（[ldlite_loader.py:57-58](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L57-L58)）。
- **每个 grid cell 存的是整数属性索引**，不是连续值——后续直接当 embedding 表的查表下标用。
- **+1 渲染约定**：因为 `0` 表示空网格，绝大多数属性值都 +1 存储；例外是 `nav_path`（二值）、`turn_type`（bitmask）、`traffic_element`（取值 1~3），见 [ldlite_loader.py:10-11](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L10-L11)。

三个通道组的具体语义（来自 [ldlite_embedding.py](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_embedding.py) 的 docstring）：

- **section（道路段）**：道路等级、道路类型、行驶方向、正向坡度、收费类型、是否导航路径。
- **lane（车道）**：车道类型、转向类型（turn_type，bitmask）、过渡类型、`link_id`（映射到 0~64）、`link_type`（来自 lane_link 数据）。
- **semantic（语义元素）**：`traffic_element`（斑马线=1、停止线=2、信号灯=3）、交通标志、路口、以及边界三属性（type/style/color，左右合并）。

## 2. ldlite_encoder 的处理逻辑

`LDLiteEncoder`（[ldlite_encoder.py:37](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_encoder.py#L37)）整体公式就是模块 docstring 里那句：

```
E_LDLite = E_Section + E_Lane + E_Semantic
```

具体分四步：

**① 输入预处理 `_prepare_inputs`**（[ldlite_encoder.py:132-176](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_encoder.py#L132-L176)）：三个 numpy 数组转成 `torch.long` tensor，缺 batch 维就补 `unsqueeze(0)`；缺数据时用全 0 占位。

**② 逐类属性 embedding（`LDLiteEmbeddingModule`）**：对每类属性图的**每个通道**建一张独立的 `nn.Embedding` 表，逐通道查表后**相加**，得到每类一个 `(B, H, W, embed_dim)` 特征图：

- `SectionEmbedding`：6 张表，`class_embed + type_embed + dir_embed + slope_embed + toll_embed + nav_path_embed`（[ldlite_embedding.py:65-72](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_embedding.py#L65-L72)）
- `LaneEmbedding`：5 张表（[ldlite_embedding.py:112-118](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_embedding.py#L112-L118)）
- `SemanticEmbedding`：6 张表（[ldlite_embedding.py:164-171](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_embedding.py#L164-L171)）

每个 cell 的语义就是 `e(u,v) = embed_1[a1] + embed_2[a2] + ... + embed_N[aN]`（[ldlite_embedding.py:11-13](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_embedding.py#L11-L13)）。

**③ 类别间聚合（相加）**（[ldlite_encoder.py:114-115](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_encoder.py#L114-L115)）：

```python
class_feat = section_feat + lane_feat + semantic_feat
```

**④ 归一化**（[ldlite_encoder.py:120](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_encoder.py#L120)）：

```python
class_feat = self.output_norm(class_feat) * self.output_scale  # LayerNorm + embed_dim**-1 缩放
```

输出 `ldlite_bev` 形状 `(B, H, W, embed_dim)`（默认 embed_dim=128）。

**注意**：positional encoding 不在这里加，而是在 fusion 模块里加（[ldlite_encoder.py:11-12](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_encoder.py#L11-L12)）。

## 3. 附：你选中的 fusion 代码在做什么

`LDLiteFusionModule.forward`（[ldlite_fusion.py:192](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L192)）把**传感器 BEV 当 query、LDLite 地图 BEV 当 key/value**，做 deformable cross-attention：

1. 形状统一成 `(B, H, W, C)`（[ldlite_fusion.py:206-213](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L206-L213)）。
2. 两者各自加 positional encoding（[218-224](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L218-L224)）。
3. 关键一步是**坐标系对齐**：把 sensor BEV 的网格中心从 sensor 范围（默认 `[0,-25.6,0,153.6,25.6,0]`）换算到 LDLite 范围（`[-150,-150,150,150]`）得到 reference points（[230-245](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L230-L245)），这样跨分辨率、跨范围的两个 BEV 空间才能正确对位。
4. deformable cross-attn → 残差+LN → FFN → 残差+LN（[255-271](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L255-L271)）。
5. 输出 `(B, H_s, W_s, C)`，保持 **sensor BEV 的空间尺寸**。

回到 [siamesebevlane_fisheye.py:388-394](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/siamesebevlane_fisheye.py#L388-L394)：`sensor_bev = bev_feats[0]`（相机 BEV 的第一层特征，`(B,C,H_s,W_s)`）作为 query，融合后 `permute(0,3,1,2)` 还原成 `(B,C,H_s,W_s)`，再 `bev_feats = [fused_bev]` 替换掉原来的 sensor BEV，交给下游 head。

---

一句话总结：`ldlite_data` 是三张「每个格子存整数属性索引」的 BEV 栅格图（section/lane/semantic 共 17 个属性通道），`ldlite_encoder` 对每个属性通道做 embedding 查表求和，三类再加总并经 LayerNorm，得到 `(B,H,W,128)` 的地图 BEV 特征，最后在 fusion 里作为 key/value 通过 deformable cross-attention 注入传感器 BEV。

----
# ldlite 数据处理的通俗说明
LDLite 是一种**矢量轻量地图**（坐标点 + 属性），`LDLiteLoader` 做的事就是：**把这张矢量地图「画」成一张 BEV 俯视网格图**，每个格子填一个整数（类别编号）。这样地图就变成了跟图像一样规整的张量，神经网络才好吃。

---

## 第一步：原始数据长什么样

从 `.pkl` 读出来的 LDLite 是一堆**矢量结构**，解析成几类（[ldlite_loader.py:255-263](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L255-L263)）：

|结构|是什么|关键字段|
|---|---|---|
|`sections`|道路段（一条条路）|`center_points`（一串坐标）+ `section_class`/`section_type`/`section_direction`/`slope`/`toll` 等属性|
|`lanes`|车道|`center_points` + `lane_type`/`turn_type`/`trans_type` + `left_line_id`/`right_line_id`|
|`lane_links`|车道之间的连接|`ref_points` + `link_type`|
|`lines`|边界线|`segments`，每段有 `line_type`/`line_style`/`color`|
|`semantics`|语义元素|斑马线、停止线、红绿灯、路牌（都是**点**）、路口（**多边形**）|
|`nav_paths`|导航路径|一串坐标|

**关键认知**：这些数据是「坐标 + 属性」的形式，是稀疏的、不定长的，直接进网络很麻烦。所以要栅格化。

---

## 第二步：建一张网格

在 `__init__` 里先定好网格（[ldlite_loader.py:57-58](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L57-L58)）：

```python
W = (x_max - x_min) / resolution   # 默认 (150-(-150))/1.5 = 200
H = (y_max - y_min) / resolution   # 200
```

也就是把 300m × 300m 的地面（`pc_range = [-150,-150,0,150,150,0]`）切成 **200×200 个格子**，每个格子 1.5m。这就是一张 BEV 俯视图的画布。

---

## 第三步：把矢量元素「画」到格子上

这是核心。对每个元素，做三步：**坐标转格子 → 裁剪到范围内 → 用 OpenCV 画上去**。

### 坐标转格子 `_world_to_grid`（[391-395](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L391-L395)）

```python
u = floor((x - x_min) / resolution)   # 世界 x → 网格列
v = floor((y - y_min) / resolution)   # 世界 y → 网格行
```

比如世界坐标 `(0, 0)`（车正前方中心）→ 格子 `(u=100, v=100)`，正好是画布中心。

### 画线：道路 / 车道 / 边界线

以一条道路 section 为例（[564-586](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L564-L586)）：

```
假设一条路 center_points = [(0,0), (15,0), (30,0)]，section_class=3，section_type=2

1. 把三个点换算成格子 → [(100,100), (110,100), (120,100)]（一条横线）
2. 用 cv2.polylines 沿这些格子画线：
   - 通道0（section_class）被涂上 3+1=4
   - 通道1（section_type）被涂上 2+1=3
   - ... 每个属性画到自己那条通道
3. 结果：这条路经过的那一排格子，6 个通道里分别存着对应的类别编号
```

所以 `section_attr_maps` 是 `(200, 200, 6)`：**6 张独立的"图层"，每张图层画的是同一种属性**。

### 画点：斑马线 / 停止线 / 红绿灯 / 路牌

用 `cv2.circle` 画小圆点（[698-712](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L698-L712)）。比如一个斑马线点在格子 `(105, 98)`，就把 `semantic_attr_maps[98, 105, 0]` 设成 1（斑马线=1，停止线=2，红绿灯=3）。

### 画面：路口 junction

用 `cv2.fillPoly` 把多边形内部整片填满（[658-677](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L658-L677)）。

### 覆盖顺序（很重要）

栅格化是有先后顺序的，**后画的覆盖先画的**：

- `traffic_element` 通道：斑马线 → 停止线 → 红绿灯，后面覆盖前面（[501-508](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L501-L508)）。
- 车道边界：先画**右边界**，再画**左边界**，冲突时左覆盖右（[522-562](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L522-L562)）。

---

## 第四步：每个格子的值是什么？（这是最容易误解的点）

**格子里的数字是「类别编号」，不是像素亮度、不是连续特征。**

一张 `section_attr_maps` 的某个格子 `[v, u, 0] = 4`，意思是「这个格子所在位置的道路，其 section_class 是 3（存成 3+1=4）」。

所以这三张图本质上是 **17 张「类别标签图」**：

```
section  (6 通道)：道路等级图、道路类型图、方向图、坡度图、收费图、导航路径图
lane     (5 通道)：车道类型图、转向图、过渡图、连接id图、连接类型图
semantic (6 通道)：交通元素图、路牌图、路口图、边界类型图、边界样式图、边界颜色图
```

---

## 第五步：为什么 +1？

因为 **0 被约定为「空」**（这个格子没有地图元素）。所以所有非空属性都 +1 存，比如属性值是 1 的存成 2，避免和"空"混淆（[ldlite_loader.py:10-11](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/datasets/pipelines/ldlite_loader.py#L10-L11)）。

三个例外不用 +1：`nav_path`（本身二值 0/1）、`turn_type`（bitmask）、`traffic_element`（本来就是 1/2/3 表示类型）。

---

## 完整流程图

```
.pkl 文件（矢量 LDLite 地图）
        │  pickle 加载 + parse_ldlite
        ▼
sections / lanes / lane_links / lines / semantics / nav_paths
        │  BDA 旋转对齐到自车坐标系
        ▼
逐元素栅格化（坐标→格子 → cv2 画线/点/面）
        │
        ▼
ldlite_data = {
    section_attr_maps:  (200, 200, 6)  int  ← 6 张属性类别图
    lane_attr_maps:     (200, 200, 5)  int  ← 5 张属性类别图
    semantic_attr_maps: (200, 200, 6)  int  ← 6 张属性类别图
}
        │  LDLiteEncoder：每个通道一张 nn.Embedding 查表
        ▼
(B, 200, 200, 128) 的 BEV 特征图
```

---

## 一个具体的格子走完全程

假设车前方 30m 处有个斑马线，正下方 5m 处有条道路：

1. 斑马线是语义元素里的一个「点」，`point=(30, 5)`，type=1。
2. 栅格化：`u = floor((30+150)/1.5)=120`，`v = floor((5+150)/1.5)=103`，于是 `semantic_attr_maps[103, 120, 0] = 1`（traffic_element 通道，斑马线=1）。
3. 这条道路是 section，center_points 经过格子 `(110,103)`，`section_class=3`，于是 `section_attr_maps[103, 110, 0] = 4`（3+1）。
4. 进 encoder：`SectionEmbedding.class_embed[4]` 查出 128 维向量，`SemanticEmbedding.traffic_element_embed[1]` 也查出 128 维向量，各自由对应格子的通道贡献，最后全图求和 + LayerNorm，得到 `(200,200,128)` 的地图 BEV 特征。

---
# ldlite fusion中的融合方式(deformable attention)
读完了 [decoder.py:149](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L149) 的 `CustomMSDeformableAttention`。它就是 **Deformable DETR 论文里的 MultiScaleDeformableAttention**（可变形注意力），和普通 Transformer attention 的核心区别是：**每个 query 不是看所有 key，而是只采样 K 个位置做注意力**。下面按「原理 → 结构 → forward 流程 → fusion 里的具体用法」讲。

## 1. 它解决什么问题

标准 self/cross-attention 是「每个 query 和所有 key 算一遍」，当 BEV 序列很长（比如 200×200=40000 个格子）时计算量是 O(N²)，且注意力分散。可变形注意力改成：**每个 query 只关注它附近的 K 个采样点**（这里 `num_points=4`），复杂度降为 O(N·K)，同时注意力更聚焦于空间邻域。

## 2. 四个可学习的投影层（init_layers）

[decoder.py:227-236](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L227-L236)：

```python
self.sampling_offsets    = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)  # 学"采样去哪"
self.attention_weights   = nn.Linear(embed_dims, num_heads * num_levels * num_points)      # 学"每个采样点多重要"
self.value_proj          = nn.Linear(embed_dims, embed_dims)                                # value 投影
self.output_proj         = nn.Linear(embed_dims, embed_dims)                                # 输出投影
```

注意：**没有 key 投影**。可变形注意力不需要 key，采样位置完全由 query 通过 `sampling_offsets` 学出来。

一个巧妙的初始化在 [init_weights:240-253](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L240-L253)：`sampling_offsets` 的 bias 被初始化成「8 个 head 均匀指向圆周 8 个方向、4 个采样点依次变远」的放射状分布，让训练初期不同 head 探索不同方向、不同采样点探索不同距离。

## 3. forward 的完整流程

签名见 [decoder.py:262-310](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L262-L310)，关键步骤：

### ① value 投影并分头（[328-331](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L328-L331)）

```python
value = self.value_proj(value)                        # (B, N_k, C)
value = value.view(bs, num_value, num_heads, -1)      # 按 head 拆
```

### ② 从 query 学出「采样偏移」和「注意力权重」（[333-339](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L333-L339)）

```python
sampling_offsets = self.sampling_offsets(query).view(
    bs, num_query, num_heads, num_levels, num_points, 2)   # 每个采样点一个 (dx, dy)
)
attention_weights = self.attention_weights(query).view(
    bs, num_query, num_heads, num_levels * num_points
)
attention_weights = attention_weights.softmax(-1)           # 归一化成权重
```

### ③ 计算实际采样位置（[344-352](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L344-L352)）

```python
offset_normalizer = [W, H]                                  # 特征图宽高
sampling_locations = (
    reference_points[..., None, :]                          # 参考点（归一化坐标）
    + sampling_offsets / offset_normalizer[..., None, :]    # 偏移量（像素）→ 归一化偏移
)
```

即：`采样位置 = 参考点 + 学习偏移/特征图尺寸`。参考点来自 fusion 里算好的坐标映射，偏移是网络学的，两者都在 [0,1] 归一化坐标里。

### ④ 双线性采样 + 加权求和（[366-388](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L366-L388)）

```python
output = MultiScaleDeformableAttnFunction.apply(     # CUDA 版（im2col 高效实现）
    value, spatial_shapes, level_start_index,
    sampling_locations, attention_weights, self.im2col_step,
)
# 或 CPU/无 CUDA 时：multi_scale_deformable_attn_pytorch(...)
```

这一步就是：在每个 `sampling_locations` 用**双线性插值**从 value 图上取特征，再按 `attention_weights` 加权求和，得到每个 query 的注意力输出 `(B, N_q, C)`。

### ⑤ 输出投影 + 残差（[390-396](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/modules/decoder.py#L390-L396)）

```python
output = self.output_proj(output)
return self.dropout(output) + identity    # identity 默认就是 query
```

## 4. 在 ldlite fusion 里的具体配置

[ldlite_fusion.py:146-153](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L146-L153)：

```python
self.deform_attn = CustomMSDeformableAttention(
    embed_dims=128, num_heads=8, num_levels=1, num_points=4,
    dropout=0.0, batch_first=True,
)
```

对应含义：

|参数|值|含义|
|---|---|---|
|`embed_dims`|128|特征维度|
|`num_heads`|8|8 个头，每头 16 维|
|`num_levels`|1|只有 1 层特征（ldlite 单尺度）|
|`num_points`|4|每个 query 采样 4 个点|

调用处 [ldlite_fusion.py:256-262](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L256-L262)：

```python
attn_out = self.deform_attn(
    query=query,                    # (B, N_q, 128)：sensor BEV 展平，作为 query
    value=value,                    # (B, N_k, 128)：ldlite BEV 展平，作为 value
    reference_points=reference_points,  # (B, N_q, 1, 2)：sensor 网格→ldlite 坐标系的归一化坐标
    spatial_shapes=[[H_l, W_l]],    # ldlite 特征图尺寸
    level_start_index=[0],
)
```

**数据流**：sensor BEV 的每个格子（query）通过 `reference_points` 找到自己在 ldlite 地图上的对应位置，再学一个小的偏移去 ldlite 上采样 4 个点，加权求和——等于「sensor 每个格子去地图的邻近区域查信息」。这正是把 LDLite 地图知识注入传感器 BEV 的机制。

## 5. 一个具体的注意力计算例子

以 sensor BEV 的第 i 个 query（假设它对应世界坐标 (0,0)，即地图中心）为例：

1. `reference_points[i] = (0.5, 0.5)`（归一化后地图中心）。
2. `sampling_offsets[i]` 学出 8×1×4×2 = 64 个数，分成 8 head × 4 点 × (dx,dy)。
3. 某 head 第 1 个点偏移 (dx,dy)=(1.5, -1.5)，除以 ldlite 尺寸 (200,200) 得归一化偏移 (0.0075, -0.0075)，采样位置 ≈ (0.5075, 0.4925)，即世界坐标约 (2.25, -2.25)m 处。
4. 在该位置对 ldlite value 图做双线性插值取 16 维向量，4 个点各取一个，按 `attention_weights`（softmax 后）加权求和，得到这个 head 的输出。
5. 8 个 head 拼接 → `output_proj` → 加上 query 残差 → 输出。

---

**一句话总结**：`deform_attn` 是可变形交叉注意力——sensor BEV 当 query，ldlite 地图 BEV 当 value，每个 query 用「参考点 + 学习偏移」在 value 上双线性采样 4 个点、加权求和，从而把地图信息对齐注入传感器特征。

一个顺带观察：`CustomMSDeformableAttention.forward` 内部已经做了 `+ identity`（identity 默认是 query），而 fusion 里又写了 `out = query + self.dropout(attn_out)`（[ldlite_fusion.py:265](vscode-webview://0f5sbahmeduphi4ivlc26do6s6ss2h8jshoi81lt0vjgq8rhn2or/projects/mmdet3d_plugin/models/necks/ldlite_fusion.py#L265)），所以这里存在一次「双重残差」（query 被加了两次）。不影响功能正确性，但如果后续要精简或对齐公式，可以留意这点。