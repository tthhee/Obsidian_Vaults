# DVGT-2：论文分析

> 标题：*DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale*  
> 来源：[arXiv:2604.00813](https://arxiv.org/abs/2604.00813)  
> 版本：v3，2026 年 4 月 24 日更新  
> 作者：Sicheng Zuo、Zixun Xie、Wenzhao Zheng、Shaoqing Xu、Fang Li、Hanbing Li、Long Chen、Zhi-Xin Yang、Jiwen Lu  
> 代码：[github.com/wzzheng/DVGT](https://github.com/wzzheng/DVGT)  
> 论文原文未说明正式发表的会议或期刊。

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | 端到端自动驾驶、视觉几何、流式 3D 重建、轨迹规划 |
| 核心范式 | Vision-Geometry-Action（VGA） |
| 核心模型 | Streaming Driving Visual Geometry Transformer（DVGT-2） |
| 核心问题 | 多帧几何模型计算和缓存随序列增长，无法满足在线驾驶实时性 |
| 关键方法 | Temporal Causal Attention、历史特征缓存、Sliding-Window Streaming |
| 联合输出 | 当前帧 dense 3D pointmaps、相对 ego-pose、未来轨迹 |
| 训练数据 | nuScenes、OpenScene、Waymo、KITTI、DDAD |
| 规划评测 | NAVSIM v1/v2、nuScenes open-loop |
| 几何评测 | Accuracy、Completeness、Abs Rel、$\delta<1.25$、Pose AUC |
| 参数规模 | 约 1.8B |
| 代表效率 | OpenScene 上约 0.27 s/帧；长序列保持近似恒定开销 |
| 代表规划结果 | NAVSIM v1：DVGT-2 88.6 PDMS，DVGT-2-NAVSIM 90.3；NAVSIM v2：88.9 / 89.6 EPDMS |

## 二、极简全文核心总结

DVGT-2 提出 Vision-Geometry-Action 范式，以密集 3D 几何而非稀疏感知或语言描述作为自动驾驶决策基础。模型使用时间因果注意力和固定长度滑动缓存，在线联合预测当前帧局部 3D pointmaps、相对自车位姿与未来轨迹，避免重复处理历史帧。模型在多数据集上获得较强几何重建效果，并在不同相机配置下无需微调即可完成规划。

## 三、研究背景与研究意义

### 3.1 端到端自动驾驶表示的演化

论文对比三类范式：

```text
传统 E2E：视觉 → 稀疏感知表示 → 轨迹
VLA：视觉 → 语言描述/推理 → 轨迹
VGA：视觉 → 密集 3D 几何 → 轨迹
```

传统方法常依赖：

- 3D bounding boxes；
- map elements；
- occupancy voxels；
- tracking 和 motion prediction。

这些表示具有信息稀疏、任务定义强、量化误差或任务间误差传递等问题。

VLA 模型通过语言描述提供高层语义，但自然语言较粗粒度，难以完整表达精确的空间几何关系。

### 3.2 为什么使用密集 3D 几何

车辆运行在三维世界中，密集 pointmap 可以直接描述：

- 前景物体表面；
- 道路和背景结构；
- 深度和空间位置；
- 多帧静态结构；
- 物体与自车的几何关系。

论文提出 VGA 的核心判断：

> 密集 3D 几何是连接视觉输入与驾驶动作的更直接、更完整的中间表示。

### 3.3 现有几何模型无法直接在线部署

DVGT、VGGT 等模型通常对一段完整视频批量处理：

$$
\mathcal{O}(T^2)
$$

当每次新增一帧时，历史帧会被反复计算，产生：

- 高延迟；
- 高显存；
- 随序列长度增长的成本；
- 无法支持无限长度驾驶视频。

Full-history streaming 虽然缓存历史特征，但缓存长度随时间线性增长：

$$
\mathcal{O}(T)
$$

DVGT-2 的目标是通过固定长度缓存实现：

$$
\mathcal{O}(W)\approx\mathcal{O}(1)
$$

其中 $W$ 是滑动窗口长度。

## 四、核心方法、模型、公式与流程

### 4.1 VGA 整体范式图

论文 Figure 2 对比传统稀疏感知、VLA 和 VGA：

![VGA paradigm comparison](https://arxiv.org/html/2604.00813v3/x2.png)

> **图 2：端到端自动驾驶范式对比。** 传统方法依赖稀疏感知表示，VLA 方法增加语言描述，而 VGA 直接重建密集 3D 几何并用于规划。图片来源：[论文 Figure 2](https://arxiv.org/html/2604.00813v3/x2.png)。

VGA 的数学定义为：

$$
\mathbf{A}_t,
\mathbf{P}_{t-T:t},
\mathbf{E}_{t-T:t}
=
\mathcal{M}_{VGA}(\mathbf{I}_{t-T:t})
$$

其中：

- $\mathbf{I}_{t-T:t}$：历史和当前多视角图像；
- $\mathbf{P}_{t-T:t}$：密集 3D pointmaps；
- $\mathbf{E}_{t-T:t}$：自车位姿；
- $\mathbf{A}_t$：当前时刻未来轨迹。

### 4.2 三种在线几何处理范式图

论文 Figure 3 对比 batch、full-history streaming 和 sliding-window streaming：

![Streaming geometry paradigms](https://arxiv.org/html/2604.00813v3/x3.png)

> **图 3：几何重建范式对比。** Batch 模型计算所有帧之间的两两关系，复杂度为 $\mathcal{O}(T^2)$；full-history streaming 维护不断增长的历史缓存，单帧成本为 $\mathcal{O}(T)$；DVGT-2 仅访问固定窗口 $W$，实现每帧 $\mathcal{O}(W)$ 的稳定开销。图片来源：[论文 Figure 3](https://arxiv.org/html/2604.00813v3/x3.png)。

| 范式 | 历史处理 | 单帧复杂度 | 长视频适应性 |
|---|---|---:|---|
| Batch | 每次联合处理全部帧 | $\mathcal{O}(T^2)$ | 差 |
| Full-history streaming | 缓存全部历史 | $\mathcal{O}(T)$ | 成本持续增长 |
| DVGT-2 sliding window | 仅缓存最近 $W$ 帧 | $\mathcal{O}(W)$ | 好 |

### 4.3 Sliding-Window Streaming

批处理模型：

$$
\mathbf{P}_{t-T:t},\mathbf{E}_{t-T:t}
=
\mathcal{G}_{batch}(\mathbf{I}_{t-T:t})
$$

Full-history streaming：

$$
\mathbf{P}_t,\mathbf{E}_t,\mathbf{C}_{t-T:t}
=
\mathcal{G}_{stream}
\left([
\mathbf{I}_t,\mathbf{C}_{t-T:t-1}
\right])
$$

DVGT-2 使用固定窗口：

$$
\mathbf{P}_t,\mathbf{E}_t,
\mathbf{C}_{t-W+1:t}
=
\mathcal{G}_{window}
\left([
\mathbf{I}_t,\mathbf{C}_{t-W:t-1}
\right])
$$

关键设计：

1. 当前帧 pointmap 在当前 ego 坐标系中预测；
2. ego-pose 只预测相对于上一帧的位姿；
3. 历史缓存使用 FIFO 更新；
4. 不再依赖第一帧作为全局坐标原点；
5. 缓存长度固定，支持任意长度视频。

### 4.4 DVGT-2 总体架构图

论文 Figure 4 展示模型内部模块：

![DVGT-2 overall architecture](https://arxiv.org/html/2604.00813v3/x4.png)

> **图 4：DVGT-2 总体架构。** 模型由 image encoder、带时间因果注意力的 geometry transformer，以及 geometry、pose、trajectory 三类预测头组成。当前帧特征与历史缓存共同进行空间—时间推理，联合输出局部 3D 几何、相对自车位姿和未来轨迹。图片来源：[论文 Figure 4](https://arxiv.org/html/2604.00813v3/x4.png)。

### 4.5 DVGT-2 输入输出

当前帧多视角图像：

$$
\mathbf{I}_t\in\mathbb{R}^{V\times H\times W\times3}
$$

历史缓存：

$$
\mathbf{C}_{t-W:t-1}
$$

模型输出：

1. 当前帧局部多视角 3D pointmaps：

$$
\mathbf{P}_t\in\mathbb{R}^{V\times H\times W\times3}
$$

2. 当前帧相对于前一帧的 ego-pose：

$$
\mathbf{E}_t\in\mathbb{R}^{7}
$$

其中包含 3D translation 和 4D rotation quaternion。

3. 未来 $N$ 步轨迹：

$$
\mathbf{A}_t\in\mathbb{R}^{N\times3}
$$

每个轨迹点包含 $x,y$ 和 yaw。

整体输出：

$$
\mathbf{A}_t,\mathbf{P}_t,\mathbf{E}_t,
\mathbf{C}_{t-W+1:t}
=
\mathcal{M}_{DVGT-2}
\left(
\mathbf{I}_t,
\mathbf{C}_{t-W:t-1}
\right)
$$

### 4.6 Image Encoder 与 Token 构造

使用预训练视觉基础模型提取图像 token：

$$
\mathbf{F}_t^{vis}=\mathcal{E}(\mathbf{I}_t)
$$

论文额外加入：

- pose tokens $\mathbf{F}_t^{pose}$；
- trajectory tokens $\mathbf{F}_t^{traj}$。

统一输入 token：

$$
\mathbf{F}_t
=
[\mathbf{F}_t^{vis},
\mathbf{F}_t^{pose},
\mathbf{F}_t^{traj}]
$$

这样，模型不仅提取视觉局部信息，还显式为 pose 和 planning 聚合全局上下文。

### 4.7 Geometry Transformer 与注意力结构

Geometry Transformer 接收当前帧 token 和历史缓存：

$$
\mathbf{G}_t^{vis},
\mathbf{G}_t^{pose},
\mathbf{G}_t^{traj}
=
\mathcal{G}
\left(
\mathbf{F}_t,
\mathbf{C}_{t-W:t-1}
\right)
$$

每个 transformer block 依次包含：

```text
Intra-View Local Attention
        ↓
Cross-View Spatial Attention
        ↓
Temporal Causal Attention
```

#### Intra-View Local Attention

建模单个图像内部的局部 token 关系，恢复细粒度空间结构。

#### Cross-View Spatial Attention

在当前帧不同摄像头之间建立空间关系，融合多视角信息。

#### Temporal Causal Attention

当前帧 token 作为 query，历史缓存作为 key/value：

$$
Q=\text{current frame},
\qquad
K,V=\text{historical cache}
$$

只允许当前帧读取历史信息，不访问未来帧，符合在线因果推理。

### 4.8 Relative Temporal Position 与缓存更新

为了让缓存特征在未来时刻可复用，论文不使用依赖绝对时间位置的编码，而采用相对时间位置编码 MRoPE-I。

缓存更新采用 FIFO：

$$
\mathbf{C}_{t-W+1:t}
=
\operatorname{FIFO}
\left(
\mathbf{C}_{t-W:t-1},
\hat{\mathbf{G}}_t
\right)
$$

其中：

- 删除最早的历史特征；
- 加入当前帧中间特征；
- 每层 transformer 都缓存中间特征；
- 缓存大小始终固定。

### 4.9 Efficient Online Inference 图

论文 Figure 5 展示在线推理和缓存更新：

![DVGT-2 efficient inference](https://arxiv.org/html/2604.00813v3/x5.png)

> **图 5：DVGT-2 高效在线推理。** 每次输入当前帧多视角图像和最近 $W$ 帧缓存，模型只计算当前帧与固定历史窗口之间的关系，输出当前帧预测后更新缓存，避免重复计算历史帧。图片来源：[论文 Figure 5](https://arxiv.org/html/2604.00813v3/x5.png)。

推理流程：

```text
当前帧 I_t + 历史缓存 C_{t-W:t-1}
              ↓
      当前帧图像编码
              ↓
    空间注意力 + 时间因果注意力
              ↓
        输出 P_t, E_t, A_t
              ↓
   丢弃最旧特征，加入当前特征
              ↓
        得到 C_{t-W+1:t}
```

### 4.10 三类 Prediction Heads

Geometry head：

$$
\mathbf{P}_t
=\mathcal{H}_{vis}(\mathbf{G}_t^{vis})
$$

论文使用 DPT head，将视觉 token 解码为 dense 3D pointmaps。

Pose head：

$$
\mathbf{E}_t
=\mathcal{H}_{pose}(\mathbf{G}_t^{pose})
$$

使用 anchor-based truncated diffusion 预测相对位姿。

Trajectory head：

$$
\mathbf{A}_t
=\mathcal{H}_{traj}(\mathbf{G}_t^{traj})
$$

同样使用 anchor-based truncated diffusion 预测未来驾驶轨迹。

### 4.11 训练流程

论文采用两阶段训练：

#### 阶段一：Geometry Reconstruction Pretraining

- 在混合驾驶数据集上进行几何重建预训练；
- 主要学习 dense pointmaps 和 ego-pose；
- 不启用流式机制或规划目标。

#### 阶段二：Vision-Geometry-Action Training

- 启用 streaming 机制；
- 加入轨迹规划监督；
- 联合学习几何、位姿和驾驶动作。

#### 阶段三：NAVSIM Fine-tuning

- 在 NAVSIM 上进行专门微调；
- 得到 DVGT-2-NAVSIM；
- 固定 8 views、4 frames，对齐 NAVSIM 设置。

训练数据的 views 和 frames 随机变化：

- 2–8 个视角；
- 2–24 帧；
- 2 Hz 采样。

### 4.12 GitHub 源码对应关系

论文对应实现位于官方仓库 [wzzheng/DVGT](https://github.com/wzzheng/DVGT)：

| 论文模块 | 主要源码 | 作用 |
|---|---|---|
| DVGT-2 主模型 | [`dvgt/models/architectures/dvgt2.py`](https://github.com/wzzheng/DVGT/blob/main/dvgt/models/architectures/dvgt2.py) | 组织 aggregator、pose head、DPT point head，并提供训练和流式推理入口 |
| Geometry Transformer | [`dvgt/models/backbones/dvgt2_aggregator.py`](https://github.com/wzzheng/DVGT/blob/main/dvgt/models/backbones/dvgt2_aggregator.py) | 图像 patch embedding、特殊 token、交替空间—时间注意力和 KV cache |
| Attention | [`dvgt/models/layers/attention.py`](https://github.com/wzzheng/DVGT/blob/main/dvgt/models/layers/attention.py) | QKV 投影、QK normalization、RoPE/MRoPE、scaled dot-product attention |
| 模型架构配置 | [`configs/`](https://github.com/wzzheng/DVGT/tree/main/configs) | 定义 backbone、head、窗口和训练配置 |

源码层面的主调用链为：

```text
DVGT2.forward / DVGT2.inference
        ↓
DVGT2Aggregator
        ↓
DINOv3 patch embedding
        ↓
Intra-view attention
        ↓
Cross-view attention
        ↓
Cross-frame causal attention + KV cache
        ↓
aggregated token list
        ├─ DPTHead → points / points_conf
        └─ ego_pose_head → pose / trajectory-related outputs
```

> 源码核对：上述调用关系来自 `dvgt2.py` 的 `forward()` / `inference()`，以及 `dvgt2_aggregator.py` 的 `forward()` 和三个 `_process_*_attention()` 方法。源码中函数和变量命名可能随仓库更新变化，本文按论文对应的公开版本解释。

### 4.13 `DVGT2` 主类：训练前向与流式推理

源码中的 `DVGT2` 继承 `torch.nn.Module`，构造函数主要创建三个组件：

```python
self.aggregator = DVGT2Aggregator(...)
self.ego_pose_head = instantiate(ego_pose_head_conf, ...)
self.point_head = DPTHead(..., output_dim=4, ...)
```

对应关系：

- `aggregator`：视觉和时序几何特征提取；
- `ego_pose_head`：相对 ego-pose 和轨迹相关输出；
- `point_head`：DPT 风格 dense 3D pointmap 和 confidence。

#### 训练或整段序列前向

`forward(images, ego_status)` 接收形状为 `[B,T,V,3,H,W]` 的图像序列。核心代码逻辑可概括为：

```python
aggregated_tokens, patch_start_idx, _ = self.aggregator(images)

ego_tokens = aggregated_tokens[-1]
ego_pose_outputs = self.ego_pose_head(ego_tokens, ego_status)

points, points_conf = self.point_head(
    aggregated_tokens,
    images=images,
    patch_start_idx=patch_start_idx,
)
```

这里 `aggregated_tokens[-1]` 表示最后一个 geometry-transformer block 的输出；`point_head` 则利用多层 intermediate tokens 进行 dense prediction。

#### 流式推理

`inference(images, ego_status, infer_window=4)` 按时间循环处理：

```python
for t in range(T):
    image = images[:, t:t+1]
    aggregator(
        image,
        past_key_values=past_key_values,
        use_cache=True,
        past_frame_idx=t,
    )
    update_pose_cache()
    decode_current_points()
    outputs_list.append(predictions)

final_outputs = merge_stream_outputs(outputs_list)
```

关键点：

1. 每次只输入当前帧，时间维度为 1；
2. `past_key_values` 保存 cross-frame attention 的历史 K/V；
3. 当超过 `infer_window` 时，源码重新整理并截断历史 KV；
4. pose token 使用独立缓存拼接最近窗口的 token；
5. 每帧输出后由 `merge_stream_outputs` 合并时间维度。

### 4.14 `DVGT2Aggregator`：从图像到时空 token

`DVGT2Aggregator` 是源码中实现论文 geometry transformer 的核心类，默认配置与论文实现一致：

| 参数 | 默认值 |
|---|---:|
| `patch_size` | 16 |
| `embed_dim` | 1024 |
| `depth` | 24 |
| `num_heads` | 16 |
| `aa_order` | `intra_view → cross_view → cross_frame` |
| `use_causal_mask` | 可配置 |
| `future_frame_window` | 8 |
| `relative_pose_window` | 1 |

源码首先对图像做归一化并展平 batch、时间和视角维度：

```python
images = (images - mean) / std
images = images.view(B * T * V, 3, H, W)
patch_tokens = self.patch_embed.forward_features(images)
```

随后将特殊 token 与 patch token 拼接：

```python
tokens = torch.cat([
    ego_pose_token,
    register_token,
    patch_tokens,
], dim=1)
```

这对应论文中的：

```text
视觉 patch tokens
+ pose tokens
+ register tokens
→ geometry transformer 输入
```

源码通过 `get_mrope_interleave_index()` 生成三维位置索引，并通过 `MRopeInterleaveEmbedding` 注入视角、空间和时间位置信息。

### 4.15 三种注意力的源码实现

#### 4.15.1 Intra-view Attention

源码 `_process_intra_view_attention()` 将 token 维度保持为：

```text
[B*T*V, P, C]
```

即每张图像单独进行局部 token 交互：

```python
tokens, _ = block(tokens, pos=pos)
intermediates.append(tokens.view(B, T, V, P, C))
```

作用是提取单视角内的局部视觉和几何结构。

#### 4.15.2 Cross-view Attention

源码先将同一时间帧的多个视角拼接：

```python
tokens_reshaped = tokens.view(B*T, V*P, C)
```

此时同一时刻的所有 camera tokens 参与 attention，再 reshape 回 `[B*T*V,P,C]`。

作用是完成当前帧内的多摄像头空间融合。

#### 4.15.3 Cross-frame Causal Attention

源码将 token 重排为：

```text
[B*V, T*P, C]
```

这样每个 camera 的 token 沿时间维排列，当前帧可读取历史帧：

```python
tokens_reshaped = tokens.view(B,T,V,P,C) \\
    .transpose(1,2) \\
    .contiguous() \\
    .view(B*V,T*P,C)
```

在流式模式下，`AttentionMRoPE` 接收 `past_key_value`，把历史 K/V 与当前帧 K/V 拼接：

```python
if past_key_value is not None:
    past_k, past_v = past_key_value
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)
new_kv = (k, v)
```

这正是论文所说的 temporal causal attention + feature cache。

### 4.16 Attention 与 MRoPE

源码 `AttentionMRoPE` 的主要步骤：

```python
qkv = self.qkv(x)
q, k, v = qkv.unbind(0)
q, k = self.q_norm(q), self.k_norm(k)
q, k = self.rope(q, k, pos)
x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
x = self.proj(x)
```

与普通 self-attention 相比，核心差异是：

- 使用 QK normalization；
- 使用 MRoPE 注入多维位置；
- 支持 `past_key_value` KV cache；
- 使用 PyTorch fused scaled dot-product attention。

对于训练序列，源码可以构造下三角 `cached_time_mask`，确保当前时间步不会访问未来帧：

$$
M_{t,s}=1\quad\text{only if}\quad s\le t
$$

### 4.17 源码中的缓存更新机制

源码缓存不是缓存最终 pointmap，而是缓存 geometry transformer 中间层的 K/V：

```text
当前帧 token
    ↓ geometry block
当前层 K/V
    ↓
写入 past_key_values
    ↓
下一帧作为历史 K/V 使用
```

当历史超过窗口时，源码通过 `einops.rearrange` 将扁平 token 维恢复为时间维，再保留最近帧：

```python
past_k = rearrange(
    past_k,
    'v head (t p) d -> v head t p d',
    t=infer_window,
)
past_k = past_k[:, :, -infer_window+1:]
```

这比重新编码历史图像更高效，但也意味着：

- cache 的 shape 必须严格匹配窗口和 patch 数；
- 改变视角数或窗口大小时需要同步处理 reshape；
- 过期历史信息被主动丢弃。

### 4.18 Prediction Heads 的代码路径

源码中 geometry 和 planning 的输出路径并不完全对称：

```text
aggregated_tokens_list
    ├─ point_head
    │    ├─ DPT decoder
    │    ├─ points
    │    └─ points_conf
    │
    └─ ego_pose_head
         ├─ pose encoding
         ├─ relative pose decoding
         └─ trajectory post-processing
```

`DPTHead` 使用多层 transformer intermediate tokens，而 pose head 主要使用最后一层并截取 pose window 对应 token。源码中的 pose/trajectory diffusion head 配置由 Hydra `instantiate()` 动态构建，因此具体 head 结构由配置文件决定，而不是硬编码在 `dvgt2.py` 中。

### 4.19 源码级伪代码

```python
class DVGT2(nn.Module):
    def forward(self, images, ego_status=None):
        tokens, patch_start, _ = self.aggregator(images)

        pose = self.ego_pose_head(
            tokens[-1][:, :, :, :self.ego_pose_window],
            ego_status,
        )
        points, confidence = self.point_head(
            tokens,
            images=images,
            patch_start_idx=patch_start,
        )
        return {**pose, "points": points, "points_conf": confidence}

    def inference(self, images, ego_status=None, infer_window=4):
        cache = None
        outputs = []

        for t in range(num_frames):
            current = images[:, t:t+1]
            tokens, patch_start, cache = self.aggregator(
                current,
                past_key_values=cache,
                use_cache=True,
                past_frame_idx=t,
            )
            pose = self.ego_pose_head(tokens[-1], ego_status)
            points = self.point_head(tokens, current, patch_start)
            outputs.append({**pose, **points})

        return merge_stream_outputs(outputs)
```

这段伪代码对应论文的完整因果链：

$$
\text{当前图像}
\rightarrow
\text{patch/token 编码}
\rightarrow
\text{视内注意力}
\rightarrow
\text{跨视角注意力}
\rightarrow
\text{跨帧 KV cache 注意力}
\rightarrow
\begin{cases}
\text{局部 3D geometry}\\
\text{相对 pose}\\
\text{未来 trajectory}
\end{cases}
$$

### 4.20 代码实现与论文描述的差异/注意事项

1. **论文使用“滑动窗口”描述缓存，源码通过 `past_key_values` 和显式 reshape/truncate 实现。** 复现时不能只按论文公式拼接图像，应保持 KV 的 `[view, head, time*patch, dim]` 组织方式。
2. **论文将三类 attention 描述为 transformer 模块，源码按 `aa_order` 分成三组 block。** 默认顺序是 `intra_view → cross_view → cross_frame`。
3. **源码输出的 pointmap 默认是局部/当前坐标系结果。** 全局评估通过 pose decode 和 `accumulate_transform_points_and_pose_to_first_frame()` 累积变换。
4. **训练和推理代码路径不同。** 训练时可使用 gradient checkpointing；推理时开启 KV cache，且当前输入通常只包含 1 帧。
5. **planning head 不在主类中完全展开。** `ego_pose_head` 由 Hydra 配置动态实例化，详细 diffusion 结构应继续查看对应 `configs/` 和 `dvgt/models/heads/` 文件。

## 五、核心创新点与传统方法对比

### 5.1 Vision-Geometry-Action 范式

论文不把语言作为主要中间表示，而是将 dense 3D geometry 作为视觉到动作之间的核心桥梁：

$$
\text{Image}
\rightarrow
\text{Dense 3D Geometry}
\rightarrow
\text{Trajectory}
$$

### 5.2 Sliding-window Streaming

相比 batch 和 full-history streaming，DVGT-2 使用固定大小历史缓存，使长视频推理的内存和延迟不随序列长度增长。

### 5.3 局部几何 + 相对位姿

模型不再将所有历史点统一预测到第一帧坐标系，而是：

- 当前帧预测局部 pointmap；
- 预测相对上一帧的 pose；
- 通过累计相对 pose 进行全局转换。

优点是局部几何更容易学习、缓存可复用；代价是全局几何和 pose 会出现累计漂移。

### 5.4 联合几何与规划

DVGT-2 不仅输出几何，还在同一个 geometry transformer 后面同时输出：

```text
Dense pointmaps + ego pose + future trajectory
```

这使几何表征直接参与规划，而不是作为独立感知模块输出后再交给下游。

### 5.5 多相机配置零微调迁移

论文声称同一个训练好的 DVGT-2 可以直接适配不同摄像头配置进行规划，无需 fine-tuning。训练时随机采样 views 和 aspect ratio 是实现这一泛化的重要策略。

### 5.6 对比总结

| 方法 | 中间表示 | 历史处理 | 复杂度趋势 | 规划输出 |
|---|---|---|---:|---|
| 传统 E2E | 稀疏 boxes/map/occupancy | 依具体方法 | 通常较低 | 有 |
| VLA | 语言/视觉语义 | 依模型 | 计算较重 | 有 |
| DVGT | 全序列 dense geometry | Batch | $\mathcal{O}(T^2)$ | 可有 |
| StreamVGGT | 全历史特征 | Full-history cache | $\mathcal{O}(T)$ | 主要几何 |
| DVGT-2 | 局部 dense geometry | Fixed sliding window | $\mathcal{O}(W)$ | 几何 + pose + 轨迹 |

## 六、理论分析与关键假设

### 6.1 复杂度分析

论文的核心复杂度结论：

- Batch pairwise processing：$\mathcal{O}(T^2)$；
- Full-history streaming：$\mathcal{O}(T)$；
- Sliding-window streaming：$\mathcal{O}(W)$。

当 $W$ 固定时，单帧推理成本近似常数：

$$
\lim_{T\rightarrow\infty}\text{cost per frame}=\mathcal{O}(W)
$$

### 6.2 关键假设

#### 局部几何足够支持规划

模型假设当前帧局部 pointmap 加上有限历史窗口，已经足以支持当前驾驶决策。

#### 相对位姿可以稳定累积

通过每帧相对 pose 将局部几何转换为全局坐标，需要假设短期 pose 误差不会快速发散。

#### 历史缓存的特征可复用

相对时间编码和缓存设计假设历史中间特征不会因绝对时间变化而失效。

#### Dense geometry 比稀疏表示更适合决策

论文将 dense 3D geometry 视为更完整的决策信息，但这需要通过规划任务结果验证，而不是单独由几何重建指标推出。

### 6.3 理论没有保证的内容

方法不保证：

- 固定窗口一定包含规划所需的全部长期信息；
- 累计相对 pose 不会发生长期漂移；
- dense pointmap 一定优于 occupancy、map 或语言表示；
- 不同相机配置无需微调时在所有传感器域都稳定；
- 几何重建误差下降必然带来规划性能提升。

### 6.4 局部与全局评价的张力

DVGT-2 原生预测局部 pointmaps，因此局部 ray depth 更容易准确；但全局 pointmap 需要累计 pose：

$$
\hat{P}^{global}_t
=
\operatorname{Transform}
\left(
\hat{P}^{local}_t,
\prod_{i=1}^{t}\hat{E}_i
\right)
$$

相对 pose 误差会累积，使全局几何和全局 pose 评价不利。这是方法设计的直接 trade-off。

## 七、实验设计与结果分析

### 7.1 数据集与训练

混合数据集：

| 数据集 | Train scenes | Test scenes | Views |
|---|---:|---:|---:|
| nuScenes | 700 | 150 | 6 |
| OpenScene | 19,376 | 2,026 | 8 |
| Waymo | 798 | 202 | 5 |
| KITTI | 138 | 13 | 2 |
| DDAD | 150 | 50 | 6 |

所有视频按 2 Hz 采样。训练混合比例：

$$
\text{nuScenes:OpenScene:Waymo:KITTI:DDAD}=6:77:6:5:6
$$

使用 MoGe-2 生成 dense depth supervision，并通过阈值过滤提高几何标签质量。

### 7.2 模型设置

- ViT-L，使用 DINOv3 预训练；
- Geometry Transformer：24 blocks；
- 特征维度：1024；
- attention heads：16；
- 参数量约 1.8B；
- pose/trajectory 各使用 20 个 diffusion anchors；
- 训练总计约 10 天，使用 64 张 H20 GPU；
- AdamW，峰值学习率 $1\times10^{-4}$；
- 8K iteration linear warmup；
- bfloat16、gradient checkpointing、gradient clipping。

### 7.3 几何重建结果：OpenScene

| 方法 | 范式 | Acc ↓ | Comp ↓ | Abs Rel ↓ | $\delta<1.25$ ↑ | AUC ↑ | 时间 |
|---|---|---:|---:|---:|---:|---:|---:|
| DVGT | Full-Seq. | 0.412 | 0.491 | 0.048 | 0.971 | 76.6 | 1.88s |
| StreamVGGT | Streaming | 2.209 | 2.060 | 0.303 | 0.620 | 74.1 | 1.94s |
| Driv3R | Streaming | 0.884 | 1.693 | 0.188 | 0.740 | — | 0.56s |
| DVGT-2 | Streaming | **0.440** | **0.450** | **0.040** | **0.977** | 70.3 | **0.27s** |

DVGT-2 以约 0.27 s/帧取得较强局部深度和点图质量，但 AUC pose 指标低于 DVGT，反映局部效率和全局位姿之间的折中。

### 7.4 其他几何重建结果

#### nuScenes

DVGT-2：

- Acc：0.775；
- Completeness：0.792；
- Abs Rel：0.055；
- $\delta<1.25$：0.965；
- AUC：84.5。

#### Waymo

DVGT-2：

- Acc：1.238；
- Completeness：1.367；
- Abs Rel：0.073；
- $\delta<1.25$：0.949；
- AUC：85.9。

#### DDAD

DVGT-2：

- Acc：1.770；
- Completeness：1.837；
- Abs Rel：0.093；
- $\delta<1.25$：0.919；
- AUC：92.5。

论文重点强调 DVGT-2 的 ray depth 指标在多个数据集上较强，但全局 point reconstruction 和 pose 受累计误差影响。

### 7.5 NAVSIM v1 闭环规划

| 方法 | 输入 | 辅助监督 | PDMS |
|---|---|---|---:|
| Hydra-MDP | Camera + LiDAR | Map + Box | 86.5 |
| DiffusionDrive | Camera + LiDAR | Map + Box | 88.1 |
| DriveSuprim | Camera + LiDAR | Map + Box | 89.9 |
| DriveVLA-W0 | Camera | Future States | 90.2 |
| DVGT-2 | Camera | Dense Geometry | 88.6 |
| DVGT-2-NAVSIM | Camera | Dense Geometry | **90.3** |

DVGT-2 foundation model 不做 NAVSIM 专门微调时达到 88.6；专门 fine-tune 后达到 90.3，超过表中多数方法。

### 7.6 NAVSIM v2 闭环规划

| 方法 | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| DVGT-2 | 97.8 | 97.2 | 99.6 | 99.9 | 88.4 | 97.3 | 98.1 | 98.2 | 83.2 | 88.9 |
| DVGT-2-NAVSIM | 98.7 | 97.9 | 99.7 | 99.9 | 87.9 | 98.0 | 98.2 | 98.2 | 77.0 | **89.6** |

DVGT-2-NAVSIM 的 EPDMS 达到 89.6。其 EC 分数低于基础 DVGT-2，说明综合分数提升并不代表所有子指标都同步提升。

### 7.7 nuScenes open-loop 规划

DVGT-2：

- 平均 L2：0.78 m；
- 平均 collision rate：0.19%。

使用历史帧平均的版本：

- 平均 L2：0.41 m；
- 平均 collision rate：0.22%。

说明历史信息和输入处理方式会明显影响 open-loop 规划结果。

### 7.8 Window size 消融

| Window size | Acc ↓ | Abs Rel ↓ |
|---:|---:|---:|
| 2 | 0.613 | 0.042 |
| 4 | 0.480 | 0.042 |
| 6 | **0.474** | 0.042 |
| 8 | 0.501 | 0.042 |

窗口从 2 增加到 6 后，全局点准确性提升；窗口为 8 时由于累计 pose 误差超过时序上下文收益，Acc 反而下降。Abs Rel 基本不变，说明 window size 主要影响全局时序建模，而非当前帧局部深度。

### 7.9 效率对比图

论文 Figure 7 对比长序列在线推理的内存和延迟：

![DVGT-2 efficiency comparison](https://arxiv.org/html/2604.00813v3/x7.png)

> **图 7：在线推理效率对比。** VGGT/DVGT 的全序列计算导致内存和延迟快速增长，StreamVGGT 的 full-history cache 也随历史增长；DVGT-2 的固定 sliding window 使内存基本恒定，延迟稳定在约 260 ms/帧。图片来源：[论文 Figure 7](https://arxiv.org/html/2604.00813v3/x7.png)。

### 7.10 实验结论与证据边界

论文实验支持：

- 固定窗口可以显著降低在线几何重建成本；
- DVGT-2 的局部 ray depth 重建能力较强；
- dense geometry 可以直接支持端到端规划；
- 混合数据集训练有利于跨数据集和相机配置泛化；
- NAVSIM 微调后规划分数进一步提升。

需要谨慎解释：

- DVGT-2 并非在所有全局 geometry 指标上优于 batch 模型；
- 规划优势部分来自 NAVSIM 专项微调；
- 0.27 s/帧是在特定硬件、序列和窗口设置下测得；
- 低碰撞率不等于真实道路安全保证。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **提出 VGA 范式。** 将 dense 3D geometry 作为视觉到动作之间的核心表示。
2. **解决在线几何重建的复杂度问题。** 通过固定窗口和缓存避免历史重复计算。
3. **联合几何与规划。** 一个模型同时输出 pointmaps、ego-pose 和 trajectory。
4. **支持多数据集训练。** 通过随机 views、frames 和 aspect ratio 提高相机配置泛化。
5. **局部几何建模更适合在线决策。** 当前帧局部 pointmap 不需要等待完整序列。

### 8.2 论文自身暴露的局限

- 全局 pose 预测不如 batch/full-sequence 方法；
- 累计相对 pose 会导致全局点图漂移；
- 固定窗口丢失长期上下文；
- KITTI 和 DDAD 上轨迹规划存在明显 domain gap；
- 训练和微调资源规模很大；
- 模型约 1.8B 参数，部署成本仍高。

### 8.3 分析者识别出的潜在问题

#### 问题一：局部—全局 trade-off

局部 pointmap 提升 ray depth，但全局地图需要累计 pose。长期运行中：

$$
\text{pose error}_{global}
\approx
\sum_t\text{relative pose error}_t
$$

因此，固定窗口虽然降低计算量，却可能牺牲长期全局一致性。

#### 问题二：规划提升与几何提升的因果关系不完全隔离

模型同时增加了：

- 大规模混合数据训练；
- 强视觉 backbone；
- dense geometry supervision；
- trajectory diffusion head；
- NAVSIM fine-tuning。

因此无法仅凭最终分数判断规划收益完全来自 VGA 表示。

#### 问题三：数据分布高度不均衡

OpenScene 占训练数据比例约 77%。附录明确显示 KITTI、DDAD 上规划 L2 误差超过 2 m，作者将其归因于速度和轨迹分布 domain gap。

这说明跨数据集泛化并非均匀，主要结果可能受主导数据集影响。

#### 问题四：窗口大小是准确率与效率的关键超参数

$W=6$ 在几何准确性上优于 $W=4$，但论文主要效率结果使用 $W=4$。不同任务的最佳窗口可能不同，需要显式报告规划—几何—延迟三者的联合权衡。

#### 问题五：预测局部坐标会增加系统集成复杂度

下游规划直接使用当前局部坐标相对简单，但若需要全局地图或长期预测，必须额外维护 pose 累积和坐标变换，带来漂移和误差传播。

#### 问题六：实时性仍需结合控制周期评估

约 260 ms/帧约为 3.8 Hz，论文数据采样为 2 Hz。若目标是更高频率的车辆控制，仍需进一步优化或引入低延迟控制器。

#### 问题七：几何 supervision 依赖伪标签

训练几何标签由 MoGe-2 推理深度并过滤得到，并非全部来自真实精确 3D 测量。伪标签误差可能影响 pointmap 学习，尤其是遮挡和远距离区域。

## 九、通俗讲解

### 9.1 传统自动驾驶看什么

传统模型可能只看稀疏对象：

```text
前方有一辆车
左侧有一条车道线
右侧有道路边界
```

或者让 VLA 模型描述：

```text
前方车辆减速，道路即将转弯
```

这些信息有用，但不够精确地告诉车辆：

```text
每个像素在三维空间中的位置是什么？
```

### 9.2 DVGT-2 做什么

DVGT-2 给每个相机像素预测一个 3D 点：

```text
图像像素 → 三维空间坐标
```

因此模型可以获得密集的三维场景：

- 车辆表面在哪里；
- 道路平面在哪里；
- 障碍物离车多远；
- 多帧之间场景如何变化。

然后直接根据这些几何信息预测未来轨迹。

### 9.3 为什么不能每次重新处理所有历史帧

如果车辆已经行驶了 1000 帧，每来一帧都把 1000 帧重新输入模型：

```text
第 1 帧：处理 1 帧
第 2 帧：处理 2 帧
...
第 1000 帧：处理 1000 帧
```

计算量会越来越大。

DVGT-2 只保留最近几帧：

```text
当前帧 + 最近 W 帧缓存
```

最早的帧被丢弃，当前帧加入缓存，因此运行很长时间也不会无限变慢。

### 9.4 为什么预测相对位姿

如果所有点都要放到第一帧坐标系中，模型需要长期记住第一帧，且容易出现全局漂移。

DVGT-2 改为：

```text
当前帧局部几何
+ 当前帧相对上一帧移动了多少
```

这更适合在线处理，但长时间累积时会有漂移。

### 9.5 模型内部怎么工作

```text
多视角图像
    ↓
视觉 token
    ↓
与最近历史特征做时间注意力
    ↓
分别交给三个 head
    ├─ 3D geometry head：预测点云式几何
    ├─ pose head：预测自车相对位姿
    └─ trajectory head：预测未来轨迹
```

### 9.6 一句话理解

> DVGT-2 让自动驾驶模型不只识别“有什么物体”，而是在线恢复“整个三维世界长什么样”，再利用固定历史窗口把这些几何信息直接转化为驾驶轨迹。

## 十、综合评价与后续研究方向

### 10.1 综合评价

DVGT-2 的核心贡献可以概括为：

$$
\text{Dense Geometry}
+
\text{Temporal Causal Attention}
+
\text{Fixed Sliding Window}
+
\text{Joint Planning}
$$

论文提出 VGA 范式，将 dense 3D geometry 作为自动驾驶决策的核心中间表示，并通过 DVGT-2 解决高质量多帧几何模型难以在线运行的问题。

完整因果链为：

$$
\text{多视角图像}
\rightarrow
\text{视觉 token}
\rightarrow
\text{空间—时间几何 Transformer}
\rightarrow
\text{局部 3D pointmap + 相对 pose}
\rightarrow
\text{未来轨迹}
$$

方法最有价值的部分是流式设计：

- 当前帧与固定历史缓存交互；
- 缓存使用相对时间编码；
- 历史特征 FIFO 更新；
- 计算和显存不随驾驶时长增长。

实验上，DVGT-2 在多数据集几何重建中具有较强局部深度能力，并在 NAVSIM v2 达到 88.9/89.6 EPDMS，在 NAVSIM v1 达到 88.6/90.3 PDMS。但需要注意，全球几何和位姿存在累计漂移，KITTI/DDAD 规划存在明显域差距，且规划最优结果依赖 NAVSIM 专项微调。

更准确的评价是：

> DVGT-2 证明了密集几何可以作为端到端自动驾驶的有效中间表示，并通过固定窗口流式缓存将高成本几何重建推进到在线规划场景；但长期全局一致性、跨域轨迹泛化和大模型部署成本仍是主要限制。

### 10.2 后续研究方向

#### 方向一：漂移校正与全局一致性

结合 loop closure、可学习全局锚点、地图匹配或外部定位，抑制相对 pose 累积误差。

#### 方向二：自适应窗口

根据场景复杂度、运动速度和几何变化动态调整 $W$，在简单场景降低延迟，在复杂场景扩大上下文。

#### 方向三：高效模型压缩

对 1.8B 模型进行蒸馏、量化、稀疏化和 token pruning，提升实际车载部署可行性。

#### 方向四：几何—规划联合目标

不只分别监督 pointmap 和 trajectory，而是引入规划风险对几何表示的反馈，使模型学习对决策最有用的几何信息。

#### 方向五：多源数据域适应

针对 KITTI、DDAD 等不同速度、相机和轨迹分布，研究 domain-balanced sampling、数据重加权和场景条件化 diffusion anchors。

#### 方向六：几何不确定性建模

预测每个 3D 点的置信度和深度不确定性，将遮挡、远距离和动态物体的几何风险传递给规划头。

#### 方向七：动态世界建模

当前模型重点重建几何和自车位姿，后续可显式预测其他交通参与者的 4D motion 和未来 occupancy。

#### 方向八：更高频率在线推理

优化 attention、缓存读写、diffusion head 和硬件部署，将约 260 ms/帧进一步降低到满足实际控制周期的水平。

## 一句话结论

> DVGT-2 以密集 3D 几何替代稀疏感知或粗粒度语言，利用固定滑动窗口和时间因果缓存实现在线几何—轨迹联合预测，在保持较强重建能力的同时显著降低长序列推理成本，是 VGA 自动驾驶范式的一项系统性探索。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2604.00813](https://arxiv.org/abs/2604.00813)
- 论文 HTML：[https://arxiv.org/html/2604.00813v3](https://arxiv.org/html/2604.00813v3)
- 论文 PDF：[https://arxiv.org/pdf/2604.00813](https://arxiv.org/pdf/2604.00813)
- 代码：[https://github.com/wzzheng/DVGT](https://github.com/wzzheng/DVGT)
