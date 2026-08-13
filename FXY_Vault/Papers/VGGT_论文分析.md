# VGGT：论文分析

> 标题：*VGGT: Visual Geometry Grounded Transformer*  
> 来源：用户提供的《VGGT_CVPR25.pdf》  
> 对应 arXiv：[2503.11651](https://arxiv.org/abs/2503.11651)  
> 发表信息：CVPR 2025；用户提供 PDF 为 CVPR 2025 版本。  
> 作者：Jianyuan Wang、Minghao Chen、Nikita Karaev、Andrea Vedaldi、Christian Rupprecht、David Novotny  
> 代码：[github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | 多视角几何、三维重建、相机估计、深度估计、点跟踪 |
| 核心模型 | Visual Geometry Grounded Transformer（VGGT） |
| 输入 | 单张、少量或数百张 RGB 图像 |
| 输出 | 相机参数、深度图、point maps、点跟踪特征 |
| 核心架构 | DINO patchification + Alternating-Attention Transformer + 多任务预测头 |
| 主要思想 | 用一个大规模前馈 Transformer 直接联合预测多种三维属性 |
| 关键设计 | Frame-wise attention 与 Global attention 交替执行 |
| 训练目标 | 相机、深度、point map、tracking 多任务联合损失 |
| 主要优势 | 无需测试时 BA/全局对齐即可输出可用三维结果 |
| 主要速度 | H100 上约 0.2 秒级前馈重建；输入帧越多成本增加 |
| 应用 | SfM/MVS、相机估计、点云重建、点跟踪、新视角合成 |

## 二、极简全文核心总结

VGGT 提出一个大规模前馈视觉几何 Transformer，从一张到数百张图像直接联合预测相机参数、深度图、视点不变 point maps 和点跟踪特征。模型采用 DINO 图像 token、交替的帧内与全局 self-attention，以及相机、密集几何和 tracking 多任务监督。相比依赖全局对齐、三角化或 Bundle Adjustment 的方法，VGGT 在约 0.2 秒内取得具有竞争力甚至领先的相机、深度、点云和跟踪结果。

## 三、研究背景与研究意义

### 3.1 传统三维重建流程

经典 SfM/MVS 通常包含：

```text
图像匹配
    ↓
三角化
    ↓
相机估计
    ↓
Bundle Adjustment
    ↓
稠密重建
```

其优点是几何约束明确，但缺点是：

- 流程复杂；
- 需要大量迭代优化；
- 输入图像越多，计算量越大；
- 对匹配失败、弱纹理和重复纹理敏感。

### 3.2 近期神经几何模型的不足

DUSt3R、MASt3R 等模型可以直接预测两视图或局部点图，但通常：

- 一次只能处理少量图像；
- 多视图需要 pairwise 推理和后续融合；
- 仍需 global alignment 或其他优化；
- 图像数量增加时成本迅速上升。

VGGT 试图进一步回答：

> 能否让一个前馈网络一次性读取大量图像，并直接输出所有关键三维属性？

### 3.3 论文的核心观点

论文采用 neural-first 路线：

$$
\text{多视角图像}
\rightarrow
\text{统一 Transformer 推理}
\rightarrow
\text{相机 + 深度 + 点图 + 跟踪}
$$

论文认为，多种三维属性相互约束：

- 相机决定几何坐标变换；
- 深度和相机可以生成 point map；
- point map 反映场景结构；
- tracking 特征帮助跨视图对应；
- 联合学习可以形成互补监督。

## 四、核心方法、模型、公式与流程

### 4.1 论文整体架构图

![VGGT architecture overview](https://arxiv.org/html/2503.11651v1/x2.png)

> **图 2：VGGT 架构。** 输入图像先通过 DINO 转换为 patch tokens，并加入 camera tokens 和 register tokens；主干交替执行 frame-wise 与 global self-attention；camera head 预测相机参数，DPT head 预测深度、point maps 和 tracking features。图片来源：[论文 Figure 2](https://arxiv.org/html/2503.11651v1/x2.png)。

整体流程：

```text
N 张 RGB 图像
      ↓
DINO patchification
      ↓
Image tokens + Camera tokens + Register tokens
      ↓
Frame-wise Self-Attention
      ↓
Global Self-Attention
      ↓ 交替 L 次
Camera Head ─────────→ 相机内外参
DPT Head ────────────→ 深度图、Point Map、Tracking Feature
Tracking Head ───────→ 跨视图 2D point tracks
```

### 4.2 问题定义

输入为同一场景的 $N$ 张 RGB 图像：

$$
(I_i)_{i=1}^{N},
\qquad I_i\in\mathbb{R}^{3\times H\times W}
$$

VGGT 输出每一帧对应的：

$$
 f(I_i)_{i=1}^{N}
=
(g_i,D_i,P_i,T_i)_{i=1}^{N}
$$

其中：

- $g_i\in\mathbb{R}^{9}$：相机参数；
- $D_i\in\mathbb{R}^{H\times W}$：深度图；
- $P_i\in\mathbb{R}^{3\times H\times W}$：point map；
- $T_i\in\mathbb{R}^{C\times H\times W}$：用于点跟踪的特征图。

第一张图像作为世界参考坐标系。Point maps 定义在第一相机的世界坐标系中，因此不同视角的像素可以对应到统一三维空间。

### 4.3 相机参数表示

相机参数表示为：

$$
 g_i=[q_i,t_i,f_i]
$$

其中：

- $q_i\in\mathbb{R}^{4}$：旋转四元数；
- $t_i\in\mathbb{R}^{3}$：平移向量；
- $f_i\in\mathbb{R}^{2}$：水平和垂直方向的 field of view 参数。

论文假设主点位于图像中心。

第一帧作为参考坐标系，因此：

$$
q_1=[0,0,0,1],
\qquad
 t_1=[0,0,0]
$$

### 4.4 Depth Map 与 Point Map

对于图像像素 $y$：

- 深度图给出该像素沿相机视线方向的深度：

$$
D_i(y)\in\mathbb{R}_{+}
$$

- Point map 给出该像素对应的三维世界点：

$$
P_i(y)\in\mathbb{R}^{3}
$$

二者并非独立任务。给定相机参数和深度，可以通过反投影得到 point map；论文仍然同时训练两者，并在推理时发现由 depth + camera 重建 point map 往往比直接 point-map head 更准确。

### 4.5 Feature Backbone

VGGT 采用大规模 Transformer，尽量减少手工 3D inductive bias，让模型从大量 3D 标注数据中学习几何规律。

每张图像先通过 DINO patchify 为 token：

$$
 t_i^I\in\mathbb{R}^{K\times C}
$$

所有图像 token 拼接后输入主干：

$$
 t^I=\bigcup_{i=1}^{N}\{t_i^I\}
$$

另外为每帧加入：

- 一个 camera token；
- 四个 register tokens。

第一帧使用独立的可学习 camera/register tokens，以便模型识别参考帧并将所有输出置于第一相机坐标系。

### 4.6 Alternating-Attention

论文核心架构是不使用 cross-attention，而交替执行两种 self-attention：

```text
Frame-wise Self-Attention
          ↓
Global Self-Attention
          ↓
Frame-wise Self-Attention
          ↓
Global Self-Attention
          ↓
重复 L 次
```

#### Frame-wise Attention

每张图像内部独立计算 attention：

$$
\operatorname{Attn}_{frame}(t_i^I)
$$

作用：

- 保持每帧内部 token 的空间结构；
- 让图像内局部特征充分交互；
- 避免所有图像 token 一开始就混合造成激活统计不稳定。

#### Global Attention

所有图像的 token 一起计算 attention：

$$
\operatorname{Attn}_{global}(t_1^I,\ldots,t_N^I)
$$

作用：

- 建立跨视图对应关系；
- 融合多视角几何信息；
- 让不同图像共同推断相机和三维结构。

默认使用 $L=24$ 层 alternating attention。论文消融显示，交替注意力优于只使用 global self-attention 或 cross-attention。

### 4.7 Camera Head

经过主干后，camera tokens 进入额外的 self-attention 层和线性层：

$$
(\hat g_i)_{i=1}^{N}
=
\mathcal{H}_{cam}
((\hat t_i^g)_{i=1}^{N})
$$

相机 head 使用四个额外 self-attention 层，再预测内外参。

### 4.8 DPT Dense Prediction Head

图像 token 首先通过 DPT layer 转换为稠密特征图：

$$
\hat t_i^I
\rightarrow
F_i\in\mathbb{R}^{C'\times H\times W}
$$

之后使用卷积层预测：

- 深度图 $\hat D_i$；
- point map $\hat P_i$；
- tracking features $\hat T_i$；
- 深度和点图的不确定性 $\hat\Sigma_i^D,\hat\Sigma_i^P$。

### 4.9 Point Tracking Head

VGGT 主干不直接输出最终 tracks，而是输出每帧 dense tracking features $T_i$。

给定查询图像 $I_q$ 中的查询点 $y_j$：

1. 在查询帧特征 $T_q$ 上双线性采样查询特征；
2. 与其他帧特征图计算 correlation maps；
3. 使用 self-attention 处理 correlation；
4. 输出所有图像中的对应点。

形式上：

$$
T((y_j)_{j=1}^{M},(T_i)_{i=1}^{N})
=
((\hat y_{j,i})_{i=1}^{N})_{j=1}^{M}
$$

tracking head 使用 CoTracker2 结构，VGGT 与 tracking 网络端到端联合训练。

### 4.10 多任务训练损失

总损失为：

$$
\mathcal{L}
=
\mathcal{L}_{camera}
+
\mathcal{L}_{depth}
+
\mathcal{L}_{pmap}
+
\lambda\mathcal{L}_{track}
$$

论文设置：

$$
\lambda=0.05
$$

#### Camera loss

使用 Huber loss：

$$
\mathcal{L}_{camera}
=
\sum_{i=1}^{N}
\|\hat g_i-g_i\|_{\epsilon}
$$

#### Depth loss

深度损失使用预测的不确定性加权，并加入梯度项：

$$
\mathcal{L}_{depth}
=
\sum_i
\left[
\hat\Sigma_i^D\odot(\hat D_i-D_i)
+
\hat\Sigma_i^D\odot(\nabla\hat D_i-\nabla D_i)
-
\alpha\log\hat\Sigma_i^D
\right]
$$

#### Point-map loss

Point-map loss 与 depth loss 类似：

$$
\mathcal{L}_{pmap}
=
\sum_i
\left[
\hat\Sigma_i^P\odot(\hat P_i-P_i)
+
\hat\Sigma_i^P\odot(\nabla\hat P_i-\nabla P_i)
-
\alpha\log\hat\Sigma_i^P
\right]
$$

#### Tracking loss

$$
\mathcal{L}_{track}
=
\sum_{j=1}^{M}\sum_{i=1}^{N}
\|y_{j,i}-\hat y_{j,i}\|
$$

并加入 visibility binary cross-entropy loss。

### 4.11 Ground-truth 坐标归一化

三维重建存在尺度和坐标系歧义：对场景进行整体缩放或改变参考坐标系，图像本身不变。

论文采用 canonical normalization：

1. 将所有量转换到第一相机坐标系；
2. 计算所有 3D 点的平均欧氏距离；
3. 使用该尺度归一化相机平移、point map 和 depth；
4. 要求网络直接学习这种规范化，而不是推理后再额外归一化。

这能消除训练标签中不必要的尺度自由度。

### 4.12 训练数据与训练设置

训练数据覆盖：

- Co3Dv2；
- BlendMVS；
- DL3DV；
- MegaDepth；
- Kubric；
- WildRGB；
- ScanNet；
- HyperSim；
- Mapillary；
- Habitat；
- Replica；
- MVS-Synth；
- Virtual KITTI；
- PointOdyssey；
- Aria Synthetic Environments；
- Aria Digital Twin；
- 类 Objaverse 的合成资产数据。

主要设置：

- 24 层 alternating attention；
- feature dimension 1024；
- 16 attention heads；
- 约 1.2B 参数；
- 160K iterations；
- AdamW；
- peak learning rate $2\times10^{-4}$；
- warmup 8K iterations；
- 每 batch 随机采样 2–24 帧；
- 最大图像尺寸 518；
- 随机 aspect ratio 和图像增强；
- 64 张 A100 训练约 9 天。

## 五、核心创新点与传统方法对比

### 5.1 多任务一体化三维预测

VGGT 不为相机、深度、点图、跟踪分别训练专门网络，而是共享一个几何 backbone，通过多任务头统一预测。

### 5.2 Large Transformer 替代重后处理

与 DUSt3R/MASt3R 等需要 pairwise 推理和 global alignment 的方法相比，VGGT 直接处理一组图像并前馈输出结果。

### 5.3 Alternating-Attention

帧内注意力保持图像内部结构，全局注意力实现跨视图几何融合，在效率和多视图交互之间取得折中。

### 5.4 过完备预测

虽然相机、深度和 point map 可以相互推导，论文仍同时监督它们。理由是冗余任务带来互补学习信号，最终提升点图质量。

### 5.5 从多视图重建到通用几何 backbone

VGGT 特征可以迁移到：

- 动态点跟踪；
- 新视角合成；
- 两视图匹配；
- 下游三维任务。

### 5.6 对比总结

| 方法 | 输入规模 | 核心机制 | 是否需要后优化 | 典型速度 |
|---|---|---|---|---|
| COLMAP/SfM | 多图像 | 匹配、三角化、BA | 是 | 慢 |
| DUSt3R | 通常两图 | pairwise pointmap | 常需全局对齐 | 秒级 |
| MASt3R | 通常少量图像 | 匹配增强 + 对齐 | 常需后处理 | 秒级 |
| VGGSfM | 多视图 | 学习 + 可微 BA | 通常需要 | 较慢 |
| VGGT | 单图到数百图 | 前馈大 Transformer | 可选 BA | 约 0.2 秒起 |

## 六、理论分析与关键假设

### 6.1 关键理论/建模假设

#### 第一帧作为参考坐标系

所有相机、point maps 和深度都需要统一到第一帧坐标系中。这提供一致的输出定义，但要求模型识别第一帧特殊 token。

#### 场景主要为静态或弱动态

多视图几何和 pointmap 的联合预测隐含假设大部分场景结构在视图之间保持一致。论文表示对轻微非刚体运动有一定适应性，但大幅非刚体变形会失败。

#### 任务间共享表示有效

多任务学习假设相机、深度、点图和 track 的学习信号互补，而不是相互冲突。

#### 数据规模可以替代强几何 inductive bias

模型只使用较少的 3D 专用结构，主要依赖大 Transformer 和大量带 3D 标注的数据。

### 6.2 不确定性建模的含义

网络同时预测误差尺度 $\Sigma$，损失中包含 $-\log\Sigma$ 项。模型可以在难预测区域输出较大不确定性，但该不确定性仍需通过训练约束避免无限增大。

### 6.3 理论没有保证的内容

论文没有严格保证：

- 单次前馈输出在任意场景都替代 BA；
- 多任务输出之间始终严格几何一致；
- 数百张输入下注意力成本仍然很低；
- 训练数据之外的极端相机、畸变和动态场景都有效；
- point map、depth 和 camera 的独立预测一定满足闭式几何约束。

### 6.4 Point map 与 Depth + Camera 的差异

虽然 point map 可以由相机和深度生成，但论文发现推理时：

$$
\text{Depth + Camera}
\rightarrow
\text{Point Map}
$$

比直接使用 point-map head 更准确。

这说明拆分复杂任务为相机和深度两个子任务，可能比让单个 point-map head 直接学习所有几何结构更容易。

## 七、实验设计与结果分析

### 7.1 相机位姿估计

在 RealEstate10K 和 CO3Dv2 上随机选择 10 张图像，使用 AUC@30 评价相机旋转和平移准确率。

| 方法 | Re10K AUC@30 | CO3Dv2 AUC@30 | 时间 |
|---|---:|---:|---:|
| DUSt3R | 67.7 | 76.7 | ~7s |
| MASt3R | 76.4 | 81.8 | ~9s |
| VGGSfM v2 | 78.9 | 83.4 | ~10s |
| VGGT Feed-Forward | **85.3** | **88.2** | ~0.2s |
| VGGT + BA | 93.5 | 91.8 | ~1.8s |

结论：

- 前馈 VGGT 已超过主要对比方法；
- 加 BA 后仍能继续提升；
- VGGT 输出是高质量 BA 初始化，而不是完全消除优化价值。

### 7.2 Dense MVS：DTU

| 方法 | Known GT camera | Acc ↓ | Comp ↓ | Overall ↓ |
|---|---|---:|---:|---:|
| MVSNet | 是 | 0.396 | 0.527 | 0.462 |
| MASt3R | 是 | 0.403 | 0.344 | 0.374 |
| DUSt3R | 否 | 2.677 | 0.805 | 1.741 |
| VGGT | 否 | **0.389** | **0.374** | **0.382** |

VGGT 在未知相机条件下达到接近甚至优于部分已知 GT camera 的方法。

### 7.3 Point Map：ETH3D

| 方法 | Acc ↓ | Comp ↓ | Overall ↓ | 时间 |
|---|---:|---:|---:|---:|
| DUSt3R | 1.167 | 0.842 | 1.005 | ~7s |
| MASt3R | 0.968 | 0.684 | 0.826 | ~9s |
| VGGT，Point head | 0.901 | 0.518 | 0.709 | ~0.2s |
| VGGT，Depth + Camera | **0.873** | **0.482** | **0.677** | ~0.2s |

Depth + Camera 组合优于直接 point-map head，支持论文关于任务分解的观察。

### 7.4 两视图匹配：ScanNet-1500

| 方法 | AUC@5 | AUC@10 | AUC@20 |
|---|---:|---:|---:|
| SuperGlue | 16.2 | 33.8 | 51.8 |
| LoFTR | 22.1 | 40.8 | 57.6 |
| DKM | 29.4 | 50.7 | 68.3 |
| Roma | 31.8 | 53.4 | 70.9 |
| VGGT | **33.9** | **55.2** | **73.4** |

虽然 VGGT 的 tracking head 并非专门为两视图匹配设计，但仍超过对比方法。

### 7.5 Alternating-Attention 消融

ETH3D point map 结果：

| Backbone | Acc ↓ | Comp ↓ | Overall ↓ |
|---|---:|---:|---:|
| Cross-Attention | 1.287 | 0.835 | 1.061 |
| Global Self-Attention Only | 1.032 | 0.621 | 0.827 |
| Alternating-Attention | **0.901** | **0.518** | **0.709** |

说明仅使用 global self-attention 或 cross-attention 都不如帧内—全局交替结构。

### 7.6 多任务消融

| $L_{camera}$ | $L_{depth}$ | $L_{track}$ | Acc ↓ | Comp ↓ | Overall ↓ |
|---|---|---|---:|---:|---:|
| ✗ | ✓ | ✓ | 1.042 | 0.627 | 0.834 |
| ✓ | ✗ | ✓ | 0.920 | 0.534 | 0.727 |
| ✓ | ✓ | ✗ | 0.976 | 0.603 | 0.790 |
| ✓ | ✓ | ✓ | **0.901** | **0.518** | **0.709** |

联合训练所有任务效果最好。相机监督对 point map 帮助明显，深度带来的额外提升较小，但 tracking 监督仍有贡献。

### 7.7 动态点跟踪

将 VGGT 预训练特征接入 CoTracker，并在 Kubric 上微调：

| 方法 | Kinetics $\delta^{vis}_{avg}$ | RGB-S $\delta^{vis}_{avg}$ | DAVIS $\delta^{vis}_{avg}$ |
|---|---:|---:|---:|
| CoTracker | 64.3 | 78.9 | 76.1 |
| CoTracker + VGGT | **69.0** | **84.0** | **77.5** |

VGGT 特征即使不是为动态视频专门训练，也能提升 tracking 表现。

### 7.8 新视角合成

VGGT backbone 微调用于 feed-forward novel view synthesis，在 GSO 上：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| LGM | 21.44 | 0.832 | 0.122 |
| GS-LRM | 29.59 | 0.944 | 0.051 |
| LVSM | 31.71 | 0.957 | 0.027 |
| VGGT-NVS | 30.41 | 0.949 | 0.033 |

VGGT 输入图像不需要已知相机参数，且只使用 LVSM 约 20% 的训练数据，仍取得有竞争力的结果。

### 7.9 运行时和显存

H100、图像尺寸 $336\times518$：

| 输入帧数 | 时间 s | 显存 GB |
|---:|---:|---:|
| 1 | 0.04 | 1.88 |
| 2 | 0.05 | 2.07 |
| 4 | 0.07 | 2.45 |
| 8 | 0.11 | 3.23 |
| 20 | 0.31 | 5.58 |
| 50 | 1.04 | 11.41 |
| 100 | 3.12 | 21.15 |
| 200 | 8.75 | 40.63 |

注意：论文强调可以处理数百张图像，但全局 self-attention 的显存和时间仍会随 token 数量快速增长。其“高效”主要是相对于优化型多视图重建，而不是对任意数量输入都具有常数复杂度。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **统一多种视觉几何任务。** 一个模型同时预测 camera、depth、point map 和 track features。
2. **前馈替代重后处理。** 在约 0.2 秒内取得接近或超过需要 BA/global alignment 的方法。
3. **多任务互补监督。** 相机、深度和 tracking 共同提升几何表示。
4. **通用几何 backbone。** 可迁移到动态跟踪和新视角合成。
5. **数据规模驱动的几何学习。** 证明大 Transformer 加大规模 3D 数据可以学习较强几何能力。

### 8.2 论文承认的局限

论文明确指出：

- 不支持 fisheye 或 panoramic images；
- 极端输入旋转下重建性能下降；
- 对大幅非刚体变形场景处理较差；
- BA 仍能带来额外精度提升；
- 全局输入帧数较大时内存和时间显著增加；
- 微分 BA 训练成本约使每步变慢 4 倍，因此未纳入主训练。

### 8.3 分析者识别出的潜在问题

#### 问题一：前馈结果并非完全替代几何优化

VGGT + BA 在相机估计上优于 VGGT 单独前馈结果，说明模型更像高质量初始化器和快速近似求解器，而不是完全取代 BA。

#### 问题二：多任务输出存在几何不一致可能

相机、深度和 point map 分别预测，并没有在网络输出层强制满足严格投影关系。推理时 depth + camera 反而更好，说明直接 point-map head 可能存在一致性问题。

#### 问题三：全局 attention 的扩展性有限

输入 200 帧时显存达到约 40.63 GB。论文的“百帧级”能力仍受显存约束，不能视为无限长视频模型。

#### 问题四：数据分布和相机配置依赖

模型训练数据非常多样，但论文也承认不支持鱼眼和全景相机。对于工业车载相机、不同畸变、不同曝光和不同视场，仍可能需要微调。

#### 问题五：静态场景假设限制动态应用

大幅非刚体运动会破坏多视图几何一致性。动态车辆、行人和非刚体物体可能导致 point map、track 和 camera 推理互相干扰。

#### 问题六：训练成本极高

约 1.2B 参数、64 张 A100、9 天训练，意味着其性能依赖大规模算力和数据，不代表小模型或小数据条件下也能复现。

#### 问题七：指标和感知质量并不完全一致

点云 Accuracy/Completeness、相机 AUC 和 tracking metrics 衡量不同方面，单一 Overall 不能完整反映几何拓扑、尺度、动态一致性和实际下游可用性。

## 九、通俗讲解

### 9.1 传统三维重建怎么做

传统系统像一个工程流水线：

```text
先找图片之间的对应点
    ↓
再计算相机位置
    ↓
再三角化 3D 点
    ↓
再用 BA 反复优化
    ↓
得到 3D 场景
```

精确，但步骤多、速度慢。

### 9.2 VGGT 怎么做

VGGT 直接把多张图片送进一个大 Transformer：

```text
图片 1、图片 2、图片 3、...
              ↓
         VGGT Transformer
              ↓
相机 + 深度 + 3D 点图 + 跟踪特征
```

它通过大量 3D 数据训练，学会从图像之间的关系直接推断三维结构。

### 9.3 为什么要两种 attention

如果只看一张图片，模型能理解局部纹理，但无法充分知道不同视角之间的关系；如果所有图像 token 一直混合，计算和训练会变得困难。

所以 VGGT 交替使用：

```text
先在每张图片内部理解
再在所有图片之间交换几何信息
```

这就是 Alternating-Attention。

### 9.4 为什么同时预测多个结果

模型同时预测相机、深度和 point map，看起来有些重复，但这些任务互相帮助：

```text
相机帮助理解空间位置
深度帮助恢复距离
point map 帮助表达完整结构
tracking 帮助发现跨图像对应
```

共同训练后，模型的几何表示更强。

### 9.5 为什么不直接预测一张点云

一张点云需要同时解决很多问题。VGGT 将它拆成：

```text
图像 → 深度
图像 → 相机
深度 + 相机 → 3D 点
```

论文实验发现，这种分解后的 depth + camera 结果有时比专门 point-map head 更准确。

### 9.6 VGGT 是不是完全不需要 BA

不是。VGGT 单独前馈已经很强，但加入 BA 后仍能继续提高精度。

更准确的理解是：

> VGGT 用神经网络快速得到高质量几何结果；如果追求更高精度，仍可以把它作为 BA 的初始化。

## 十、综合评价与后续研究方向

### 10.1 综合评价

VGGT 的核心贡献可以概括为：

$$
\text{大规模 3D 数据}
+
\text{Alternating-Attention Transformer}
+
\text{相机/深度/点图/跟踪多任务}
+
\text{前馈几何推理}
$$

其完整因果链为：

$$
\text{多视角图像}
\rightarrow
\text{DINO patch tokens}
\rightarrow
\text{帧内注意力}
\rightarrow
\text{跨帧全局注意力}
\rightarrow
\text{相机与密集几何表示}
\rightarrow
\text{点图/深度/跟踪输出}
$$

论文的重要性在于，它把多视图三维重建从“先匹配、再优化、再融合”的传统流程，推进到“一个大模型直接预测多种三维属性”的方向。

实验表明，VGGT 在相机估计、MVS、点云重建、匹配和点跟踪上都具有较强表现，并且其特征可以作为下游视觉几何 backbone。

但应避免把它理解为完全取代所有几何优化：

- BA 仍有额外收益；
- 大输入数量带来显存和延迟增长；
- 动态、鱼眼、全景和极端旋转场景存在限制；
- 训练成本极高；
- 多任务独立输出可能不完全满足几何一致性。

更准确的评价是：

> VGGT 展示了大规模前馈 Transformer 学习通用视觉几何的可行性，并在速度与精度之间取得了很强的折中；它既可以作为无需后处理的快速重建器，也可以作为传统 BA/SfM 的高质量初始化和通用几何特征 backbone。

### 10.2 后续研究方向

#### 方向一：流式和长视频 VGGT

将全局 attention 改造为固定窗口、KV cache 或 memory token，使模型支持无限长度视频并降低显存增长。

#### 方向二：几何一致性约束

加入显式投影一致性：

$$
\hat P_i(y)
\approx
\Pi^{-1}(\hat g_i,y,\hat D_i(y))
$$

减少 camera、depth 和 point-map head 之间的不一致。

#### 方向三：动态场景几何

将静态 pointmap 扩展为 4D geometry，显式建模车辆、行人和非刚体物体的运动。

#### 方向四：更高效的注意力

研究 sparse attention、低秩 attention、token pruning、tensor parallelism 和 blockwise global attention，解决 100–200 帧显存增长问题。

#### 方向五：机器人和自动驾驶部署

将 VGGT 作为 DVGT-2 等驾驶模型的几何 backbone，研究相机、深度、点图和轨迹的联合训练，而不是只做离线重建。

#### 方向六：不确定性驱动的几何推理

将预测的 aleatoric uncertainty 传递给下游匹配、BA、规划和控制模块，实现风险敏感决策。

#### 方向七：弱监督和无监督训练

利用可微 BA、跨视图重投影、视频一致性和自监督 correspondence，降低对显式 3D 标注的依赖。

#### 方向八：跨相机与跨域泛化

加入鱼眼、全景、车载广角和不同相机内参，研究相机条件化和 domain-adaptive geometry foundation model。

## 一句话结论

> VGGT 用一个大规模、交替注意力的前馈 Transformer，直接从单张到数百张图像联合预测相机、深度、point map 和跟踪特征，在无需后处理的情况下取得强视觉几何性能；其核心突破是把多视图几何从迭代优化流程推进为可迁移的神经基础模型。

## 参考链接

- 论文 arXiv：[https://arxiv.org/abs/2503.11651](https://arxiv.org/abs/2503.11651)
- 论文 PDF：[https://arxiv.org/pdf/2503.11651](https://arxiv.org/pdf/2503.11651)
- 官方代码：[https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- 用户提供 PDF：`VGGT_CVPR25.pdf`
