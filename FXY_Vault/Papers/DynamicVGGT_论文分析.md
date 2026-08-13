# DynamicVGGT：论文分析

> 标题：*DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving*  
> 来源：[arXiv:2603.08254](https://arxiv.org/abs/2603.08254)  
> 版本：v1，2026 年 3 月 9 日提交  
> 作者：Zhuolin He、Jing Li、Guanghao Li、Xiaolei Chen、Jiacheng Tang、Siyang Zhang、Zhounan Jin、Feipeng Cai、Bin Li、Jian Pu、Jia Cai、Xiangyang Xue  
> 论文原文未说明正式发表的会议或期刊。  
> 该论文没有在 arXiv 摘要页明确给出代码链接。

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | 4D 场景重建、动态视觉几何、自动驾驶、3D Gaussian Splatting |
| 基础模型 | VGGT |
| 核心模型 | DynamicVGGT |
| 核心问题 | 静态 feed-forward 3D 模型难以建模自动驾驶中的动态物体和时序运动 |
| 核心表示 | Dynamic Point Maps（DPM） |
| 关键模块 | Motion-aware Temporal Attention、Future Point Head、Dynamic 3D Gaussian Splatting Head |
| 主要输出 | 当前/未来 point maps、动态 Gaussian、深度、场景流与新视角图像 |
| 主要数据集 | Waymo、Virtual KITTI、MVS-Synth；KITTI 用于评估 |
| 主要结果 | KITTI point-map Acc 0.901、NC 0.939；Waymo 3DGS 动态区域 PSNR 18.07、SSIM 0.376 |
| 训练方式 | 合成数据几何预训练 → 真实驾驶数据动态 Gaussian 微调 |
| 论文类型 | 方法论文、自动驾驶视觉几何研究 |

## 二、极简全文核心总结

DynamicVGGT 将 VGGT 从静态 3D 感知扩展到动态 4D 场景重建。模型在共享参考坐标系中联合预测当前和未来 point maps，通过 Motion-aware Temporal Attention 建模时序运动，再使用 Future Point Head 隐式学习点位移，并用带 scene-flow 监督的 Dynamic 3D Gaussian Splatting Head 显式预测 Gaussian 速度和渲染动态场景。两阶段训练使其在 KITTI、Waymo 上优于 VGGT 与 StreamVGGT。

## 三、研究背景与研究意义

### 3.1 静态视觉几何模型的局限

VGGT、DUSt3R 等 feed-forward 视觉几何模型主要在静态场景上学习：

```text
多视角图像 → 相机、深度、点图
```

它们隐含假设不同时间或视角之间的场景几何基本不变。但自动驾驶场景中存在：

- 车辆和行人运动；
- 自车运动；
- 道路结构和视角快速变化；
- 长时间依赖；
- 大量动态遮挡。

直接将静态模型用于驾驶视频，常见问题是：

- 动态目标位置不一致；
- point maps 跨帧不连贯；
- 点云稀疏、模糊或变形；
- 无法进一步进行动态场景渲染。

### 3.2 为什么需要 4D 场景表示

3D 重建描述“物体在哪里”，4D 重建还要描述“物体如何运动”：

$$
\text{4D Scene}
=
\text{3D Geometry}
+
\text{Temporal Motion}
$$

对于自动驾驶，动态 4D 表示可以支持：

- 动态物体重建；
- 新视角合成；
- 未来场景预测；
- 闭环仿真；
- 轨迹规划和风险评估。

### 3.3 现有动态重建的局限

传统动态 3DGS 方法通常依赖：

- 场景级优化；
- 稠密标注；
- 已知相机参数；
- 每个场景单独训练。

这些方法质量可能较高，但不适合大规模、实时、跨场景推理。

DynamicVGGT 的目标是：

> 用一个 feed-forward 模型直接从多视角驾驶图像学习动态几何和运动，不依赖外部相机外参或逐场景优化。

## 四、核心方法、模型、公式与流程

### 4.1 DynamicVGGT 总体框架图

![DynamicVGGT overall framework](https://arxiv.org/html/2603.08254v1/x2.png)

> **图 2：DynamicVGGT 训练框架。** 输入多帧多视角图像，首先使用预训练 DINOv2/VGGT 结构提取 patch 与 camera tokens；AA blocks 建模帧内空间几何，MTA blocks 通过 motion tokens 并行建模跨帧运动；时序特征同时输入 Future Point Head 和 Dynamic 3D Gaussian Head，分别学习未来点图和显式 Gaussian motion。图片来源：[论文 Figure 2](https://arxiv.org/html/2603.08254v1/x2.png)。

整体流程：

```text
多帧多视角图像
        ↓
VGGT / DINOv2 视觉编码
        ↓
Patch tokens + Camera tokens
        ├─ AA blocks：帧内空间几何
        └─ Motion-aware Temporal Attention
              ↓
        时序增强特征 T_A
              ├─ Future Point Head
              │     └─ 预测未来 point map
              └─ Dynamic 3DGS Head
                    ├─ Gaussian geometry
                    ├─ appearance
                    └─ Gaussian velocity
        ↓
Dynamic Point Maps + 动态 3DGS + 新视角渲染
```

### 4.2 Dynamic Point Map 定义

普通静态 point map 可表示为：

$$
P_{v,t}
=
\pi^{-1}
\left(
I_{v,t};K_{v,t},E_{v,t}
\right)
\in\mathbb{R}^{3\times H\times W}
$$

其中：

- $v$：camera index；
- $t$：frame index；
- $I_{v,t}$：第 $t$ 帧第 $v$ 个相机的图像；
- $K_{v,t}$：相机内参；
- $E_{v,t}$：相机外参；
- $P_{v,t}$：每个像素对应的三维点。

动态场景可以在共享参考坐标系中表示：

$$
P_{v,t}^{ref}
=
\mathcal{T}_{(v,t)\rightarrow ref}
\left(
P_{v,t}
\right)
$$

跨帧点运动为：

$$
\Delta P_{v,t}^{ref}
=
P_{v,t+\delta}^{ref}-P_{v,t}^{ref}
$$

DynamicVGGT 不依赖外部显式坐标变换，而是让模型在 VGGT 学到的 canonical/shared coordinate space 中预测当前和未来 point maps：

$$
\hat P_{v,t},\hat P_{v,t+\delta}
=
 f_\theta(\{I_{v,t}\})
$$

隐式运动为：

$$
\Delta \hat P_{v,t}
=
\hat P_{v,t+\delta}-\hat P_{v,t}
$$

### 4.3 动态任务组织图

![Dynamic point map task formulation](https://arxiv.org/html/2603.08254v1/x3.png)

> **图 3：动态任务定义。** Future Point Head 通过当前特征预测未来 point map，利用点图之间的一致性隐式学习运动；Dynamic 3DGS Head 以 Gaussian primitive 为单位预测速度，并通过 scene flow 进行显式运动监督。图片来源：[论文 Figure 3](https://arxiv.org/html/2603.08254v1/x3.png)。

两个互补任务：

```text
Future Point Head
    → 点图级位移约束
    → 隐式 motion learning

Dynamic 3DGS Head
    → Gaussian 速度约束
    → 显式 scene-flow motion learning
```

## 4.4 Motion-aware Temporal Attention（MTA）

### 4.4.1 设计动机

仅使用 point-map 位移监督不足以完整建模动态运动。StreamVGGT 等方法直接叠加时序 attention，可能导致：

- 与原 VGGT 空间建模互相干扰；
- 训练初期不稳定；
- 静态几何先验被破坏。

DynamicVGGT 引入 Motion-aware Temporal Attention，将 motion tokens 作为显式时序先验，与 VGGT 的空间 AA 分支并行。

### 4.4.2 MTA 输入

AA blocks 输出聚合特征：

$$
\tilde F_{v,t}
=[F_{v,t}^{c};F_{v,t}^{p}]
$$

其中：

- $F_{v,t}^{c}$：camera token；
- $F_{v,t}^{p}$：patch tokens。

MTA 移除 camera token，将 patch tokens 与可学习 motion tokens $M_{v,t}^{(l)}$ 拼接：

$$
F_{m,v,t}^{(l)}
=
\operatorname{Concat}
\left(
M_{v,t}^{(l)},F_{v,t}^{p(l)}
\right)
$$

第 $l>1$ 层还融合前一层 patch 特征：

$$
F_{m,v,t}^{(l)}
=
\operatorname{Concat}
\left(
M_{v,t}^{(l)},
F_{v,t}^{p(l)}+F_{v,t}^{p(l-1)}
\right)
$$

### 4.4.3 时间注意力

对于每个 view 和每个 patch 位置，MTA 沿时间维计算 attention：

$$
A_{t,t'}^{(l)}
=
\operatorname{Softmax}
\left(
\frac{Q_t^{attn(l)}K_{t'}^{attn(l)\top}}{\sqrt d}
+B_{t,t'}^{time}
\right)
$$

其中 $B_{t,t'}^{time}$ 是基于 rotary position embedding 的时间位置偏置。

时间聚合为：

$$
\tilde F_{m,v,t}^{(l)}
=
\sum_{t'=1}^{\tau}
A_{t,t'}^{(l)}V_{t'}^{attn(l)}
$$

然后使用 LayerNorm、MLP 和 residual connection：

$$
F_{m,v,t}^{(l+1)}
=
\operatorname{MLP}^{(l)}
\left(
\operatorname{LayerNorm}
\left(
\tilde F_{m,v,t}^{(l)}
\right)
\right)
+
F_{m,v,t}^{(l)}
$$

最终时序增强特征为：

$$
T_{A,v,t}=F_{m,v,t}^{(L)}
$$

### 4.4.4 MTA 的作用

MTA 不直接输出运动，而是让每个当前帧 patch 能够访问同一 camera stream 的其他时刻特征：

```text
同一空间位置的 patch
      ↓ 跨时间 attention
观察它在前后帧中的变化
      ↓
形成时序连续的几何特征
```

## 4.5 Future Point Head（FPH）

FPH 根据当前时刻的时序特征预测未来点图：

$$
\hat P_{v,t+\delta}^{fut}
=
\operatorname{DPT}_p(T_{A,v,t})
$$

它的关键思想是：

- 输入当前帧特征；
- 预测同一 camera stream 的未来帧 point map；
- 通过预测点图和真实未来点图之间的约束学习运动；
- 不需要单独显式生成逐点 motion vector 才能获得运动信息。

### 4.5.1 Temporal Consistency Loss

定义有效点集合 $\mathcal{N}$，时间一致性损失为：

$$
\mathcal{L}_{temp}
=
\frac{1}{|\mathcal{N}|}
\sum_{i\in\mathcal{N}}
\left\|
\left(
 p_{v,t+\delta}^{(i)}-p_{v,t}^{(i)}
\right)
-
\left(
 \hat p_{v,t+\delta}^{(i)}-\hat p_{v,t}^{(i)}
\right)
\right\|_1
$$

该损失约束的是点级位移：

$$
\Delta p^{gt}\approx\Delta \hat p
$$

它属于隐式运动监督，因为模型不一定显式输出独立的 flow head，而是通过未来点图的一致性学习 motion。

## 4.6 Dynamic 3D Gaussian Splatting Head

### 4.6.1 设计动机

FPH 主要处理 point-map 层面的运动一致性。为了获得更高质量的动态场景渲染，论文进一步引入 Dynamic 3DGS Head，以 Gaussian primitive 为基本动态表示。

每个 Gaussian primitive 表示为：

$$
\mathcal{G}_i
=
\{\mu_i,\sigma_i,r_i,c_i,\nu_i\}
$$

其中：

- $\mu_i$：Gaussian 中心；
- $\sigma_i$：尺度；
- $r_i$：旋转；
- $c_i$：颜色/外观；
- $\nu_i$：速度向量。

### 4.6.2 外观与几何特征融合

Gaussian head 同时使用：

1. 输入图像的 RGB appearance features；
2. MTA 输出的 geometry/motion features。

图像外观特征：

$$
F_{v,t}^{app}=\operatorname{Conv}(I_{v,t})
$$

几何特征和 Gaussian depth：

$$
F_{g,v,t},D_{g,v,t}
=
\operatorname{DPT}_g(T_{A,v,t})
$$

融合特征：

$$
G_{v,t}
=
F_{v,t}^{app}+F_{g,v,t}
$$

使用预测 depth 和 VGGT camera branch 输出重建 point map：

$$
P_{v,t}^{g}
=\operatorname{Unproject}
(D_{g,v,t},\hat g_{v,t})
$$

以此初始化 Gaussian 中心 $\mu_i$。

### 4.6.3 Motion Tokens 与 Gaussian 速度

MTA 中的可学习 motion tokens 被用来解码一组 velocity bases：

$$
\nu_b\in\mathbb{R}^{3}
$$

每个 Gaussian 通过 motion representation 得到速度 $\nu_i$。

论文在短时间片段内采用 constant velocity 假设：

$$
\mu_{i,t+\delta}
=
\mu_{i,t}+\delta\cdot\nu_{i,t}
$$

该设计将动态 Gaussian 从静态位置扩展为随时间变化的 primitive。

### 4.6.4 为什么要融合 appearance features

论文观察到，如果冻结 VGGT 的 AA blocks，几何推理能力较强，但外观建模能力不足，Gaussian rendering 质量下降。因此加入直接从 RGB 图像提取的 appearance branch，补充：

- 颜色；
- 纹理；
- 细节；
- 渲染相关外观特征。

## 4.7 两阶段训练目标

### 4.7.1 Stage 1：几何与时序预训练

第一阶段使用高质量合成驾驶数据，目标为：

$$
\mathcal{L}_{stage1}
=
\mathcal{L}_{cam}
+
\mathcal{L}_{depth}
+
\mathcal{L}_{point}(t)
+
\mathcal{L}_{point}(t+\delta)
+
\lambda_{temp}\mathcal{L}_{temp}
$$

主要学习：

- 当前帧几何；
- 未来帧几何；
- 跨帧点位移；
- 相机参数；
- 动态场景的几何先验。

### 4.7.2 Stage 2：Dynamic 3DGS 微调

第二阶段在真实驾驶数据上加入 Gaussian rendering：

$$
\mathcal{L}_{stage2}
=
\mathcal{L}_{stage1}
+
\mathcal{L}_{3DGS}
$$

其中：

$$
\mathcal{L}_{3DGS}
=
\mathcal{L}_{rgb}
+
\lambda_{gs}\mathcal{L}_{gsdepth}
+
\lambda_{dist}\mathcal{L}_{distill}
+
\lambda_{flow}\mathcal{L}_{flow}
$$

### 4.7.3 RGB Rendering Loss

将动态 Gaussian 渲染图像与真实 RGB 图像比较：

$$
\mathcal{L}_{rgb}
=
\operatorname{MSE}
( I_{v,t},\hat I_{v,t})
$$

### 4.7.4 Depth Distillation

真实自动驾驶数据中的 LiDAR 通常稀疏且空间分布不均。直接使用 sparse LiDAR 监督会导致：

- 深度图不平滑；
- 点云粗糙；
- Gaussian 优化不稳定。

论文使用 Stage 1 point-map branch 产生的深度作为 teacher：

$$
\mathcal{L}_{distill}
=
\left\|
D_{g,v,t}
-
\operatorname{sg}(D_{v,t}^{pm})
\right\|_1
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。

### 4.7.5 Scene Flow Loss

显式 Gaussian 运动使用 scene-flow 监督：

$$
\mathcal{L}_{flow}
=
\operatorname{MSE}
(s_{v,t},\hat s_{v,t})
$$

与 $\mathcal{L}_{temp}$ 的区别：

| 损失 | 作用层级 | 监督对象 |
|---|---|---|
| $\mathcal{L}_{temp}$ | Point-map 层面 | 点图跨帧位移 |
| $\mathcal{L}_{flow}$ | Gaussian 层面 | Gaussian 速度/scene flow |

二者互补，而不是完全重复。

## 4.8 DynamicVGGT 训练框架中的关键模块图

![DynamicVGGT training framework](https://arxiv.org/html/2603.08254v1/x2.png)

> 该图位于论文方法章节，重点展示 AA 空间分支与 MTA 时间分支并行，之后共同服务 FPH 和 DGSHead。它是理解 DynamicVGGT 代码/模型拆分的关键图。

## 五、核心创新点与传统方法对比

### 5.1 从静态 VGGT 到动态 DPM

VGGT 主要学习：

$$
\text{Image}
ightarrow\text{Static Point Map}
$$

DynamicVGGT 学习：

$$
\text{Image Clip}
\rightarrow
\text{Current Point Map}
+
\text{Future Point Map}
+
\text{Motion}
$$

### 5.2 Motion-aware Temporal Attention

MTA 不是简单将时间 attention 叠加到 VGGT 后面，而是：

- 使用独立 motion tokens；
- 以 patch 位置为单位沿时间聚合；
- 与原始 AA 空间分支并行；
- 保留 VGGT 的静态几何先验。

### 5.3 Future Point Head

FPH 通过预测未来 point map，让模型从几何一致性中隐式学习运动，不依赖完全显式的运动标注才能工作。

### 5.4 Dynamic 3DGS Head

DGSHead 将动态建模从 point-map 层扩展到 Gaussian primitive 层：

```text
几何中心 + 尺度 + 旋转 + 颜色 + 速度
```

可以直接用于动态场景渲染和新视角合成。

### 5.5 两阶段课程训练

先用合成数据学习稳定几何和运动，再用真实数据和 3DGS 目标微调，缓解真实自动驾驶数据深度稀疏造成的性能下降。

### 5.6 对比总结

| 方法 | 场景 | 表示 | 动态建模 | 是否逐场景优化 |
|---|---|---|---|---|
| VGGT | 静态为主 | Point maps/depth/camera | 弱 | 否 |
| StreamVGGT | 时序/室内为主 | Streaming point maps | 有限 | 否 |
| STORM | 动态驾驶 | 3DGS | 强 | 依赖相机/几何条件 |
| DrivingForward | 动态驾驶 | 3DGS | 强 | Feed-forward |
| DynamicVGGT | 动态驾驶 | DPM + Dynamic 3DGS | 隐式 + 显式 | 否 |

## 六、理论分析与关键假设

### 6.1 Dynamic Point Map 的核心假设

论文假设当前和未来 point maps 可以在共享 learned canonical coordinate 中表达。这样点图差异可以作为 motion signal：

$$
\Delta \hat P
=
\hat P_{future}-\hat P_{current}
$$

它避免显式依赖外部相机外参对所有帧进行统一坐标对齐。

### 6.2 MTA 的关键假设

- 同一 camera/view 的对应 patch 在跨帧中具有可比性；
- motion tokens 可以学习具有普适性的时序先验；
- 时间 attention 的局部聚合足以表达短期运动连续性。

### 6.3 Constant Velocity 假设

Dynamic 3DGS 使用：

$$
\mu_{i,t+\delta}
=
\mu_{i,t}+\delta\nu_{i,t}
$$

这在短片段内近似合理，但对：

- 加速和减速；
- 转向；
- 遮挡恢复；
- 非刚体形变；
- 复杂交互运动

可能不足。

### 6.4 论文没有保证的内容

方法不保证：

- DPM 的跨帧对应一定是物理真实对应；
- 常速度模型适用于所有动态物体；
- point-map consistency 一定能解决遮挡和新出现点；
- Gaussian velocity 一定具有长期预测能力；
- feed-forward 结果在所有真实驾驶场景中超过逐场景优化。

### 6.5 局部几何与动态外观的差距

DGSHead 需要同时拟合：

- 几何；
- 颜色；
- 动态位置；
- 视角变化。

即使 point map 很准确，RGB rendering 仍可能受到 appearance feature、遮挡和运动建模误差影响。因此几何指标与渲染指标需要分开解读。

## 七、实验设计与结果分析

### 7.1 数据和实现设置

训练数据：

- Waymo Open Dataset；
- Virtual KITTI；
- MVS-Synth。

训练策略：

- Stage 1：Virtual KITTI + MVS-Synth，几何和时序预训练；
- Stage 2：Waymo + Virtual KITTI，Dynamic 3DGS 微调。

模型配置：

- 12 个 MTA layers；
- 约 1.4B 参数；
- 从预训练 VGGT 初始化；
- 约 800M 参数参与微调；
- 时间偏移 $\delta$ 随机取 1–3；
- 每批处理 18 张图像；
- 图像最长边不超过 518。

学习率：

- Stage 1：峰值 $1\times10^{-6}$；
- Stage 2：峰值 $5\times10^{-5}$；
- 两阶段均使用 warmup + cosine decay。

损失权重：

$$
\lambda_{temp}=0.01,
\quad
\lambda_{gs}=\lambda_{dist}=0.1,
\quad
\lambda_{flow}=0.01
$$

### 7.2 KITTI 和 Waymo Point Map

| 方法 | KITTI Acc ↓ | KITTI Comp ↓ | KITTI NC ↑ | Waymo Acc ↓ | Waymo Comp ↓ | Waymo NC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| VGGT | 1.489 | 0.690 | 0.918 | 4.635 | 2.667 | 0.561 |
| StreamVGGT | 1.078 | 0.495 | 0.899 | 4.598 | 2.626 | 0.564 |
| DynamicVGGT | **0.901** | **0.584** | **0.939** | **4.021** | **2.390** | **0.562** |

KITTI：

- Acc：1.489 → 0.901；
- NC：0.918 → 0.939；
- 相比 VGGT，几何更平滑、时间更一致。

Waymo：

- Acc：4.635 → 4.021；
- Completeness：2.667 → 2.390；
- NC：0.561 → 0.562。

### 7.3 Dynamic 3DGS 渲染

Waymo 动态区域：

| 方法 | Supervision | PSNR ↑ | SSIM ↑ |
|---|---|---:|---:|
| 3DGS | Full | 17.13 | 0.267 |
| DeformableGS | Full | 17.10 | 0.266 |
| GS-LRM | Camera | 20.02 | 0.520 |
| STORM | Camera | 21.26 | 0.535 |
| DynamicVGGT | Image-only | 18.07 | 0.376 |

完整图像：

| 方法 | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| 3DGS | 25.13 | 0.741 |
| DeformableGS | 25.29 | 0.761 |
| GS-LRM | 25.18 | 0.753 |
| STORM | 25.03 | 0.750 |
| DynamicVGGT | 24.07 | 0.676 |

需要谨慎：DynamicVGGT 使用 image-only，不使用相机参数和逐场景优化；但其 PSNR/SSIM 低于使用更强几何条件的 STORM。这说明它的主要优势是泛化和免标定，而不是所有渲染指标都达到最佳。

### 7.4 深度估计

| 方法 | KITTI Mono Abs Rel ↓ | KITTI Mono $\delta<1.25$ ↑ | NYU Abs Rel ↓ | KITTI MVS Abs Rel ↓ | KITTI MVS $\delta<1.25$ ↑ |
|---|---:|---:|---:|---:|---:|
| DUSt3R | 0.109 | 0.873 | 0.081 | 0.143 | 0.814 |
| MASt3R | 0.077 | 0.948 | 0.110 | 0.115 | 0.848 |
| MonST3R | 0.098 | 0.895 | 0.094 | 0.107 | 0.884 |
| VGGT | 0.082 | 0.938 | 0.059 | 0.062 | 0.969 |
| StreamVGGT | 0.082 | 0.947 | 0.057 | 0.173 | 0.721 |
| DynamicVGGT | **0.070** | 0.940 | 0.064 | **0.051** | **0.976** |

DynamicVGGT 在 KITTI 单目 Abs Rel 和 KITTI MVS 上表现最好；NYU-v2 的 Abs Rel 略高于 VGGT/StreamVGGT，说明动态驾驶训练并不保证在室内数据上全面提升。

### 7.5 消融实验

| 变体 | KITTI Acc ↓ | KITTI Comp ↓ | KITTI NC ↑ | Waymo Acc ↓ | Waymo Comp ↓ | Waymo NC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline VGGT | 1.489 | 0.690 | 0.918 | 4.635 | 2.667 | 0.561 |
| + TA & FPH（Stage 1） | 0.927 | 0.600 | 0.915 | 4.330 | 2.939 | 0.561 |
| + DGSHead（Stage 2） | **0.901** | **0.584** | **0.939** | **4.021** | **2.390** | **0.562** |

观察：

- TA + FPH 主要带来精度改善；
- Stage 2 DGSHead 进一步提升精度、完整性和 normal consistency；
- Waymo Completeness 在加入 Stage 1 后短暂变差，再由 DGSHead 恢复并超过 baseline；
- 模块收益并非所有指标单调改善。

### 7.6 可视化结果

![DynamicVGGT point-map reconstruction](https://arxiv.org/html/2603.08254v1/x5.png)

> **图 5：Point map 重建。** DynamicVGGT 生成更密集、更平滑且时序更一致的 point maps，在大视角变化和动态场景中仍能保持较好结构。图片来源：[论文 Figure 5](https://arxiv.org/html/2603.08254v1/x5.png)。

![DynamicVGGT scene reconstruction](https://arxiv.org/html/2603.08254v1/x6.png)

> **图 6：场景重建与新视角合成。** 给定若干输入帧，模型重建对应动态场景，并合成后续视角/时间的图像，展示其 4D 几何和外观建模能力。图片来源：[论文 Figure 6](https://arxiv.org/html/2603.08254v1/x6.png)。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **将 VGGT 扩展到动态 4D。** 从静态 point maps 进一步学习点级运动。
2. **隐式和显式运动互补。** FPH 约束 point-map displacement，DGSHead 监督 Gaussian velocity。
3. **统一几何、运动和外观。** 同一 feed-forward 模型输出 point maps 并支持动态 Gaussian rendering。
4. **减少对外部相机参数的依赖。** 主要使用图像输入，适合真实驾驶数据。
5. **两阶段训练缓解稀疏深度问题。** 先用合成高质量几何学习，再用真实数据做动态渲染。

### 8.2 论文暴露的局限

- Dynamic 3DGS 渲染结果仍低于使用 camera/geometry 条件的部分方法；
- constant-velocity 仅适合短时间片段；
- 主要在 KITTI、Waymo 上验证；
- 动态场景中的遮挡和大幅非刚体运动仍然困难；
- 模型参数量约 1.4B；
- 训练依赖合成数据和预训练 VGGT。

### 8.3 分析者识别出的潜在问题

#### 问题一：Dynamic Point Map 的对应关系可能不严格物理一致

预测的当前和未来点图来自共享网络，但并不等价于显式追踪每个真实物理点。对于遮挡、出视野和新出现区域，差分：

$$
\hat P_{future}-\hat P_{current}
$$

可能混合物体运动、视角变化和点对应变化。

#### 问题二：Constant Velocity 限制运动表达

Gaussian 中心采用线性更新：

$$
\mu_{t+\delta}=\mu_t+\delta\nu_t
$$

它难以描述转弯、加速、刹车、非刚体变形和交互行为。较长时间预测中误差会快速累积。

#### 问题三：DGS 渲染指标不占优

DynamicVGGT 在动态区域 PSNR/SSIM 低于 STORM，说明免相机、免场景优化和高渲染质量之间存在明显 trade-off。论文更准确的贡献是通用前馈动态重建，而非达到最佳渲染质量。

#### 问题四：Stage 1/Stage 2 的收益归因不完全隔离

模型同时改变：

- temporal attention；
- future point prediction；
- DGS head；
- depth distillation；
- scene-flow supervision；
- 训练数据域。

当前消融验证了组合收益，但尚不能完全分离每个损失和模块的独立贡献。

#### 问题五：动态场景中的深度蒸馏教师可能有偏差

Stage 1 point-map branch 作为 Gaussian depth teacher，虽然比稀疏 LiDAR 更平滑，但它本身是模型预测结果，不是真实深度，可能将几何误差传递给 DGSHead。

#### 问题六：自动驾驶数据域泛化有限

论文在 NYU-v2 上没有全面超过 VGGT，说明动态驾驶数据训练可能改善驾驶域，但不保证通用室内或非驾驶场景的几何泛化。

#### 问题七：计算规模限制实际部署

约 1.4B 参数、动态 Gaussian 渲染和多任务输出使其更接近大模型研究框架，而非直接可部署的车端模型。

## 九、通俗讲解

### 9.1 VGGT 的问题

VGGT 很擅长回答：

```text
这组图片中的三维世界是什么样？
```

但它主要把世界看成静态的。如果一辆车在不同帧中移动，VGGT 可能把它当成不同位置的静态几何，而不是明确建模它的运动。

### 9.2 DynamicVGGT 的核心想法

DynamicVGGT 不仅预测当前画面中的三维点，还预测未来画面中的三维点：

```text
当前 point map
        ↓
未来 point map
```

两者的差异就提供了运动线索：

```text
未来位置 - 当前位​​置 = 点的运动
```

### 9.3 MTA 在做什么

MTA 可以理解为“让同一个像素位置查看其他时间”：

```text
第 1 帧的这个位置
第 2 帧的这个位置
第 3 帧的这个位置
          ↓ 时间 attention
判断这个位置发生了什么变化
```

motion tokens 相当于专门用来记录运动模式的记忆单元。

### 9.4 FPH 和 DGSHead 的区别

FPH 更像是在点图层面学习：

```text
这个点未来应该移动到哪里？
```

DGSHead 更像是在三维物体粒子层面学习：

```text
这个 Gaussian 粒子在哪里、是什么颜色、以什么速度移动？
```

所以：

```text
FPH：隐式点图运动
DGSHead：显式 Gaussian 运动
```

### 9.5 为什么分两阶段训练

真实驾驶数据中的 LiDAR 很稀疏，直接用它训练稠密 Gaussian 容易产生粗糙结果。论文先用合成数据学习比较干净的几何和时序关系，再用真实数据训练动态渲染，并用第一阶段的深度作为更平滑的 teacher。

### 9.6 一句话理解

> DynamicVGGT 是“让 VGGT 学会物体怎么动”：先通过未来点图差分学习隐式运动，再让 Gaussian 显式携带速度，从而实现动态 4D 场景重建。

## 十、综合评价与后续研究方向

### 10.1 综合评价

DynamicVGGT 的核心贡献可以概括为：

$$
\text{VGGT 静态几何}
+
\text{MTA 时序建模}
+
\text{Future Point Prediction}
+
\text{Dynamic 3DGS}
$$

完整因果链为：

$$
\text{多帧多视角图像}
\rightarrow
\text{VGGT 空间特征}
\rightarrow
\text{Motion-aware Temporal Attention}
\rightarrow
\begin{cases}
\text{未来 point map}\
\text{Gaussian geometry + velocity}
\end{cases}
\rightarrow
\text{4D 场景重建与渲染}
$$

论文最大的价值是把静态 feed-forward 视觉几何模型推进到动态驾驶场景，并通过两个层次的运动建模避免只依赖单一 motion head：

- FPH 从点图一致性中隐式学习运动；
- DGSHead 用 scene flow 显式监督 Gaussian 速度。

实验显示，DynamicVGGT 在 KITTI 和 Waymo 的 point-map 重建上优于 VGGT/StreamVGGT，在 KITTI 和 KITTI MVS 深度估计上取得较强结果，并能生成动态场景和后续视角的图像。

但需要正确理解其贡献边界：

- 动态区域渲染质量仍不如使用更强几何条件或逐场景优化的方法；
- Gaussian 运动采用短期 constant-velocity 假设；
- DPM 的跨帧对应并非严格物理追踪；
- 训练和模型规模较大；
- 真实复杂动态场景中的长期一致性仍未完全解决。

更准确的评价是：

> DynamicVGGT 提出了一条将静态视觉几何 backbone 扩展到动态 4D 重建的清晰路径，通过 MTA、未来点预测和动态 Gaussian 速度建模，在自动驾驶数据上实现了较强的 feed-forward 动态几何能力，但距离高保真、长时域、轻量化的真实世界 4D 重建仍有明显距离。

### 10.2 后续研究方向

#### 方向一：非恒速运动模型

将 constant velocity 扩展为 acceleration、曲线运动或基于 transformer 的时间动态函数：

$$
\mu_{t+\delta}
=
\mu_t
+\delta\nu_t
+\frac{1}{2}\delta^2 a_t
$$

#### 方向二：显式点对应与遮挡处理

结合 tracking、scene flow 和 visibility 预测，区分真实点运动与视角变化、新出现点和遮挡恢复。

#### 方向三：更强动态 Gaussian 表示

为每个 Gaussian 建模时间变化的：

- 位置；
- 尺度；
- 旋转；
- 颜色；
- 不透明度。

而非只预测速度。

#### 方向四：长时域 4D 重建

引入 temporal memory、windowed motion tokens、动态对象级 memory 和全局漂移校正，支持更长视频。

#### 方向五：几何—规划联合学习

将动态点图和 Gaussian motion 直接服务于自动驾驶轨迹规划、碰撞预测和闭环控制。

#### 方向六：降低真实数据标注成本

结合自监督重投影、可微渲染、伪 scene flow、LiDAR 稀疏约束和跨帧一致性，减少对高质量动态标签的依赖。

#### 方向七：模型压缩与部署

研究 DynamicVGGT 的蒸馏、量化、稀疏 attention、轻量 motion head 和分辨率自适应，以满足车端算力和延迟要求。

#### 方向八：跨域泛化

在鱼眼、不同相机配置、不同天气、室内动态和机器人数据上验证 motion-aware geometry 的普适性。

## 一句话结论

> DynamicVGGT 通过“未来点图一致性 + Motion-aware Temporal Attention + Gaussian 速度监督”，将 VGGT 从静态三维感知扩展到自动驾驶动态四维重建，在较低标注依赖下同时学习几何和运动，是 feed-forward 4D 视觉几何的一项有价值探索。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2603.08254](https://arxiv.org/abs/2603.08254)
- 论文 HTML：[https://arxiv.org/html/2603.08254v1](https://arxiv.org/html/2603.08254v1)
- 论文 PDF：[https://arxiv.org/pdf/2603.08254](https://arxiv.org/pdf/2603.08254)
