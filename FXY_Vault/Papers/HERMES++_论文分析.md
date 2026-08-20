# HERMES++：论文分析

> 标题：*HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation*  
> 来源：[arXiv:2604.28196](https://arxiv.org/abs/2604.28196)  
> 版本：v1，2026 年 4 月 30 日提交  
> 作者：Xin Zhou、Dingkang Liang、Xiwu Chen、Feiyang Tan、Dingyuan Zhang、Hengshuang Zhao、Xiang Bai  
> 延伸关系：论文说明这是 ICCV 2025 论文 HERMES 的扩展版本。  
> 论文原文未说明本扩展版本的正式发表信息。  
> 代码：摘要称模型和代码将公开，但当前 arXiv 页面没有提供正式代码仓库链接。

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | Driving World Model、3D 场景理解、未来点云预测、多模态 LLM |
| 核心问题 | 现有世界模型偏重未来生成，LLM 偏重语义理解，二者缺少统一几何表示 |
| 核心模型 | HERMES++ |
| 输入 | 多视角相机图像、文本指令、ego motion |
| 中间表示 | BEV tokens、world queries、未来 BEV features |
| 输出 | 文本回答、当前/未来 3D point clouds |
| LLM | InternVL2 系列，实验包括 1.8B 和 3.8B |
| 关键模块 | BEV Visual Tokenizer、BEV-to-Point Render、World Queries、Current-to-Future Link、Joint Geometric Optimization |
| 训练阶段 | 几何预训练 → 视觉语言对齐/精调 → 联合理解与未来几何预测 |
| 主要数据 | nuScenes、OmniDrive-nuScenes、NuScenes-QA、DriveLM、NuInteract |
| 主要几何指标 | 0–3 秒未来场景 Chamfer Distance，越低越好 |
| 主要理解指标 | METEOR、ROUGE-L、CIDEr、VQA accuracy |

## 二、极简全文核心总结

HERMES++ 将驾驶场景的 3D 理解与未来几何生成统一到一个 BEV-LLM 框架中。多视角图像先压缩为 BEV tokens，LLM 通过 world queries 聚合视觉、文本和世界知识；Current-to-Future Link 再将语义上下文传播到未来 BEV，最后通过共享 BEV-to-Point Render 预测未来点云。联合几何优化同时约束渲染深度、局部特征和全局 Gram 结构，使模型兼顾语言理解与几何预测。

## 三、研究背景与研究意义

### 3.1 Driving World Model

驾驶世界模型希望学习环境状态的演化：

$$
\mathcal{O}_t,\mathcal{A}_t
\rightarrow
\mathcal{O}_{t+1}
$$

通常包含：

$$
\mathcal{Z}_t=E(\mathcal{O}_t),
\qquad
\mathcal{Z}_{t+1}=M(\mathcal{Z}_t,\mathcal{A}_t),
\qquad
\hat{\mathcal{O}}_{t+1}=D(\mathcal{Z}_{t+1})
$$

现有方法重点是未来图像、占用或点云生成，但往往缺少语言级场景理解。

### 3.2 LLM 驾驶模型的局限

LLM 可以回答：

- 当前道路上有哪些车辆？
- 前方车辆是否可能变道？
- 当前场景是否存在风险？

但纯语言模型通常难以直接预测：

- 物体在未来 1–3 秒的位置；
- 3D 几何结构如何变化；
- 点云或 BEV 几何如何连续演化。

因此存在：

$$
\text{semantic reasoning}
\not\Rightarrow
\text{physical simulation}
$$

### 3.3 HERMES++ 的核心观点

论文试图让语言理解与几何生成共享信息：

```text
多视角图像
    ↓
BEV 几何表示
    ↓
LLM 语义理解 + World Queries
    ↓
Current-to-Future Link
    ↓
未来 BEV
    ↓
未来点云
```

BEV 被选为接口，因为它同时具备：

- 多视角融合能力；
- 清晰的空间拓扑；
- 可压缩为 LLM token；
- 可解码为未来 3D 几何。

## 四、核心方法、模型、公式与流程

### 4.1 HERMES++ 总体框架图

![HERMES++ pipeline](https://arxiv.org/html/2604.28196v1/pipeline.png)

> **图 2：HERMES++ 流程。** BEV tokens、用户指令和 world queries 输入 LLM；world queries 从语言理解分支吸收语义上下文；Current-to-Future Link 将当前 BEV 传播到未来时刻；共享 Render 解码未来点云；Joint Geometric Optimization 同时约束输出几何和隐式几何特征。图片来源：[论文 Figure 2](https://arxiv.org/html/2604.28196v1/pipeline.png)。

```text
Multi-view images
        ↓
Vision encoder + BEVFormer-style spatial cross-attention
        ↓
Current BEV feature F_bev
        ├─ Downsample + flatten → BEV tokens
        ├─ BEV-to-Point Render → current point cloud
        └─ LLM input with instructions + world queries
                          ↓
               semantic text + enriched world queries
                          ↓
              Current-to-Future Link
              ├─ world-query cross-attention
              ├─ textual injection
              └─ ego modulation
                          ↓
                   future BEV features
                          ↓
                 shared BEV-to-Point Render
                          ↓
                  future point clouds
```

### 4.2 BEV Visual Tokenizer

给定 $N$ 个相机的图像：

$$
\{I_t^i\}_{i=1}^{N}
$$

视觉编码器提取多尺度透视特征，再通过空间 cross-attention 生成当前 BEV：

$$
F_{bev}^t\in\mathbb{R}^{w\times h\times c}
$$

BEV query 位于预设鸟瞰网格。对于网格位置 $(x,y)$，聚合多个相机和高度 anchor 的特征：

$$
B(x,y)
=
\sum_{i=1}^{N}\sum_{z\in\mathcal{H}}
\operatorname{DA}
\left(
Q(x,y),F_i,\pi_i(x,y,z)
\right)
$$

其中：

- $Q(x,y)$：BEV 网格 query；
- $F_i$：第 $i$ 个相机特征；
- $\pi_i$：3D 位置投影到图像平面的映射；
- $\operatorname{DA}$：multi-scale deformable attention；
- $\mathcal{H}$：高度 anchor 集合。

原始 BEV token 数量较大，因此下采样 4 倍：

$$
F_{down}^t\in
\mathbb{R}^{\frac{w}{4}\times\frac{h}{4}\times 4c}
$$

再展平和线性投影到 LLM 隐空间：

$$
F_t
=\phi(\operatorname{Flatten}(F_{down}^t))
\in\mathbb{R}^{L_{BEV}\times C}
$$

其中：

$$
L_{BEV}=\frac{w}{4}\times\frac{h}{4}
$$

### 4.3 BEV-to-Point Render

BEV 本身缺少高度维度。Render 先将 BEV 特征上采样并扩展为体素特征：

$$
\hat V^t
\in
\mathbb{R}^{w\times h\times z\times c'}
$$

然后将其视为隐式 SDF 场。对 LiDAR ray $r_k$，沿射线采样：

$$
 p_i=o+d_i t_k
$$

局部体素特征通过 trilinear interpolation 得到 $f_i$，SDF MLP 输出：

$$
 s_i=\phi_{SDF}(p_i,f_i)
$$

渲染深度：

$$
\tilde d(r_k)
=\sum_{i=1}^{n}w_i d_i
$$

$$
 w_i=T_i\alpha_i,
 \qquad
 T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
$$

论文使用基于 SDF 的 occupancy/opacity 转换：

$$
\alpha_i
=
\max\left(
\frac{\sigma_\tau(s_i)-\sigma_\tau(s_{i+1})}
{\sigma_\tau(s_i)},0
\right)
$$

最终将渲染深度反投影为空间点，得到当前或未来 point cloud。

### 4.4 Language-based Scene Understanding

LLM 接收三类输入：

1. BEV visual tokens；
2. 用户文本指令；
3. world queries。

文本 token 记为：

$$
T\in\mathbb{R}^{L_{text}\times C}
$$

LLM 使用自回归 next-token prediction 进行场景理解。语言损失为：

$$
\mathcal{L}_{lang}
=
-
\sum_{i=1}^{L_{text}}
\log P
\left(
T_i\mid F_t,T_{<i};\Theta
\right)
$$

关键不是只让 LLM 输出文字，而是利用 LLM 的 hidden states 生成后续几何预测所需的 semantic context。

### 4.5 World Queries

#### 4.5.1 设计动机

普通 LLM 输出文本，语义信息和未来几何分支之间可能断开。World queries 被插入 LLM 输入序列，作为可学习的 latent carriers：

```text
BEV tokens + text tokens + world queries
                    ↓ LLM
         enriched world queries
                    ↓
          future geometry branch
```

#### 4.5.2 Query 构造

设每个未来时刻有 $n$ 个 spatial world queries，共预测 $\Delta t$ 个未来时间点。基础 query 由 $F_{down}^t$ 自适应池化得到：

$$
Q\in\mathbb{R}^{n\times4c}
$$

对每个未来时刻的 ego motion 生成 embedding：

$$
e_{t+i}=\operatorname{MLP}(\operatorname{EgoMotion}_{t+i})
$$

加入可学习的 frame embedding $FE_i$：

$$
Q_w
=
\phi
\left(
\operatorname{Concat}_{i=1}^{\Delta t}
\left(Q\oplus e_{t+i}\right)
\oplus FE
\right)
$$

其中 $\oplus$ 表示带广播的逐元素加法。

World queries 通过 LLM 的 causal attention 访问其前面的视觉和文本 token，因而可以聚合：

- 当前场景视觉信息；
- 用户指令语义；
- LLM 预训练获得的通用知识；
- 对未来时间点的 ego-motion 条件。

### 4.6 Current-to-Future Link

World queries 较稀疏，不能直接生成完整未来 BEV。因此 Current-to-Future Link 从当前 BEV $B_t$ 出发，利用 world queries 和文本语义生成未来特征。

#### 4.6.1 Textual Injection

从 LLM 文本 token 中池化得到：

$$
\hat T\in\mathbb{R}^{k\times C}
$$

第 $i$ 个未来时刻的 Link block 使用：

$$
X_{cross}^{(l)}
=
X^{(l)}
+
\operatorname{CrossAttn}
\left(
\operatorname{LN}(X^{(l)}),
[Q_{w,i}^{\epsilon};\hat T]
\right)
$$

这里 world queries 和文本 embedding 共同作为 key/value。

#### 4.6.2 Ego Modulation

未来 ego motion 通过 MLP 产生调制参数 $\gamma,\beta$：

$$
\operatorname{EM}(x)
=(\gamma+1)\odot\operatorname{LN}(x)+\beta
$$

$\gamma,eta$ 初始为零，从而训练初期近似恒等映射，降低新模块破坏预训练特征的风险。

#### 4.6.3 输出

经过多层 cross-attention、self-attention、FFN 和 EM 后，将特征 reshape/upsample 成未来 BEV：

$$
\{B_{t+i}\}_{i=1}^{\Delta t}
$$

再通过共享 Render 输出未来点云：

$$
\{B_{t+i}\}
\rightarrow
\{P_{t+i}\}
$$

### 4.7 Joint Geometric Optimization

#### 4.7.1 显式渲染约束

对当前和未来时刻的 ray depth 进行 L1 监督：

$$
\mathcal{L}_{render}
=
\sum_{i=0}^{\Delta t}
\lambda_i
\frac{1}{N_i}
\sum_{k=1}^{N_i}
\left|
 d(r_k)-\tilde d(r_k)
\right|
$$

论文设置 frame-wise 权重：

$$
\lambda_i=1+0.5i
$$

这会提高长期未来预测的几何监督权重。

#### 4.7.2 几何特征先验

论文先训练一个自监督 point cloud reconstruction network，得到 geometry-aware feature extractor。主训练阶段冻结该 extractor，提取目标几何特征：

$$
V_t\in\mathbb{R}^{w\times h\times z\times c'}
$$

预测特征为 $\hat V_t$。

#### 4.7.3 Cosine Feature Loss

对每个 voxel 位置施加局部方向一致性：

$$
\mathcal{L}_{cos}
=
1-
\frac{1}{whz}
\sum_{i,j,k}
\frac{
\hat V_t(i,j,k)\cdot V_t(i,j,k)
}
{
\|\hat V_t(i,j,k)\|_2\|V_t(i,j,k)\|_2
}
$$

#### 4.7.4 Gram Loss

沿不同空间轴投影特征，得到 $d\in\{HW,HZ,WZ\}$ 的 feature maps $V_t^d$。Gram matrix：

$$
G_t^d=V_t^d(V_t^d)^\top
$$

全局结构损失：

$$
\mathcal{L}_{gram}
=
\frac{1}{3}
\sum_{d}
\left\|
G_t^d-\hat G_t^d
\right\|_F^2
$$

Cosine loss 更关注局部对应和方向，Gram loss 更关注空间位置之间的全局关系。

#### 4.7.5 总生成目标

$$
\mathcal{L}_{gen}
=10\mathcal{L}_{render}
+\mathcal{L}_{cos}
+\mathcal{L}_{gram}
$$

总目标：

$$
\mathcal{L}_{total}
=\mathcal{L}_{lang}+\mathcal{L}_{gen}
$$

## 五、核心创新点与传统方法对比

### 5.1 统一语义理解与几何生成

过去方法通常分为两类：

- world model：擅长预测未来视觉/几何，但语言理解弱；
- LLM driving model：擅长问答和语义推理，但未来几何预测弱。

HERMES++ 用共享 BEV、world queries 和 Current-to-Future Link 连接二者。

### 5.2 BEV 作为 LLM 接口

直接将多视角图像 token 输入 LLM 容易导致空间拓扑被打散。BEV 在压缩 token 的同时保留：

- 俯视空间关系；
- 车辆和道路的相对位置；
- 多相机融合后的统一坐标；
- 未来几何解码所需的空间结构。

### 5.3 LLM-enhanced World Queries

World queries 是语义理解到几何生成的桥梁，而不是普通的视觉 query。它们从 LLM 的 causal attention 中吸收图像、文本和预训练世界知识。

### 5.4 Current-to-Future Link

该模块将稀疏语义 query 转换为密集未来 BEV，并加入文本注入和 ego modulation，实现：

$$
\text{semantic context}+
\text{current geometry}+
\text{ego motion}
\rightarrow
\text{future geometry}
$$

### 5.5 Joint Geometric Optimization

只监督渲染结果可能出现 latent geometry ambiguity。HERMES++ 同时使用：

- 显式深度/渲染约束；
- 局部 cosine geometry prior；
- 全局 Gram structural prior。

## 六、理论分析与关键假设

### 6.1 BEV 是有效统一接口的假设

模型假设 BEV 比直接多视角 token 更适合作为 LLM 输入和 point cloud decoder 输入。该假设依赖：

- BEV 投影质量；
- 相机标定准确性；
- BEV 分辨率足以保留细节；
- 下采样不会破坏关键小目标。

### 6.2 World Queries 能传递 LLM 世界知识

论文认为 world queries 通过 LLM causal attention 聚合语义和通用知识，但这不等于 LLM 的抽象知识一定能转化为准确 3D 几何。实际效果依赖：

- query 插入位置；
- attention mask；
- 多模态对齐质量；
- 几何监督强度。

### 6.3 隐式特征先验的假设

冻结 geometry extractor 提供的 feature prior 被认为包含有效空间结构。若 extractor 本身存在偏差，cosine/Gram regularization 也可能把偏差传给 HERMES++。

### 6.4 未来几何预测的条件假设

Current-to-Future Link 使用当前 BEV、world queries、文本和 ego motion 预测未来点云，隐含假设这些信息足以描述未来环境。对于不可观测事件、强交互、多主体决策和遮挡区域，预测仍具有不可避免的不确定性。

### 6.5 论文没有证明的内容

- 语言理解一定能改善所有未来几何预测；
- BEV 在所有相机配置和视场下优于多视角 token；
- 3D point cloud 预测等价于完整 driving world model；
- 生成性能和语言问答性能之间存在必然正相关；
- 未来 3 秒之外仍能保持可靠几何一致性。

## 七、实验设计与结果分析

### 7.1 数据集和指标

#### nuScenes

用于多视角图像、同步点云、当前/未来几何训练和评估。几何指标是双向 Chamfer Distance：

$$
CD(P,\hat P)
=\frac{1}{|P|}\sum_{p\in P}\min_{\hat p\in\hat P}\|p-\hat p\|_2
+\frac{1}{|\hat P|}\sum_{\hat p\in\hat P}\min_{p\in P}\|\hat p-p\|_2
$$

论文在 ROI 内评估：

$$
 x,y\in[-51.2,51.2]\text{ m},
\qquad
 z\in[-3,5]\text{ m}
$$

#### OmniDrive-nuScenes

用于场景描述、VQA 和统一 instruction tuning。理解指标：

- METEOR；
- ROUGE-L；
- CIDEr。

#### NuScenes-QA

约 460K QA pairs，使用 top-1 accuracy。

#### DriveLM

Graph VQA，评估感知、预测、规划之间的逻辑链，使用官方 hybrid metrics。

#### NuInteract

约 1.5M 语言驾驶标注，用于视觉—语言初始对齐。

### 7.2 训练配置

| 阶段 | 主要内容 | Epoch | Batch | LR |
|---|---|---:|---:|---:|
| Stage 1 | 几何预训练与视觉 tokenizer/Render | 12/6 | 32 | $2\times10^{-4}$ |
| Stage 2 | 视觉语言对齐与 refinement | 3/6 | 128 | $2\times10^{-4}/4\times10^{-4}$ |
| Stage 3 | 理解 + 未来几何联合训练 | 36 | 128 | $4\times10^{-4}$ |

实现细节：

- OpenCLIP ConvNeXt-L 作为视觉 backbone；
- InternVL2 作为语言模型；
- BEV 网格 $180\times180$，通道数 256；
- Render 高度维 $z=32$，输出通道 $c'=32$；
- 预测未来 $\Delta t=3$ 秒；
- 使用 LoRA 微调 LLM；
- 训练使用 AdamW + cosine scheduler。

### 7.3 统一模型对比

#### 未来点云生成

| 方法 | 0s | 1s | 2s | 3s |
|---|---:|---:|---:|---:|
| ViDAR | 1.12 | 1.38 | 1.73 | — |
| DriveX | 0.66 | 0.86 | — | 1.10 |
| Hermes++，1.8B | **0.53** | **0.71** | **0.86** | **1.01** |
| Hermes++，3.8B | **0.51** | **0.68** | **0.82** | **0.97** |

Chamfer Distance 越低越好。论文报告 3 秒时，HERMES++ 相比 ViDAR 的 CD 降低 41.6%。

#### 场景理解

| 方法 | METEOR | ROUGE | CIDEr |
|---|---:|---:|---:|
| Omni-L | 0.376 | 0.321 | 0.732 |
| OmniDrive-2D | 0.383 | 0.325 | 0.671 |
| Hermes++，1.8B | 0.385 | 0.327 | 0.749 |
| Hermes++，3.8B | **0.389** | **0.331** | **0.772** |

HERMES++ 在没有额外 3D box/lane auxiliary supervision 的情况下取得竞争力结果。需要注意，不同模型的输入模态、训练数据和辅助任务不同，表格比较不完全同质。

### 7.4 BEV 输入消融

论文比较 BEV 输入与直接多视角 image tokens：

- 两者理解性能接近；
- BEV 在未来几何生成上明显更优；
- BEV 方法将 3 秒 CD 从 2.012 降至 1.436，约改善 26.8%；
- 直接多视角 token 在 LLM 中更容易发生空间结构坍缩。

BEV 下采样：

| 配置 | 3 秒 CD | CIDEr |
|---|---:|---:|
| Downsample ×8 | 1.781 | 0.681 |
| Downsample ×4 | **1.436** | 0.720 |
| Direct Query ×4 | 2.012 | 0.723 |

×4 在几何和理解之间取得较好折中；×8 信息瓶颈明显。

### 7.5 Joint Geometric Optimization 消融

| $\mathcal{L}_{cos}$ | $\mathcal{L}_{gram}$ | 3 秒 CD |
|---|---|---:|
| — | — | 1.637 |
| ✓ | — | 1.441 |
| — | ✓ | 1.544 |
| ✓ | ✓ | **1.436** |

Cosine loss 主要改善局部特征对应，Gram loss 改善全局结构关系，联合使用效果最好。

### 7.6 Current-to-Future Link 消融

| 配置 | 3 秒 CD | CIDEr |
|---|---:|---:|
| w/o Link | 2.377 | 0.433 |
| Simple Link | 1.542 | 0.718 |
| + Textual Injection | 1.506 | 0.717 |
| + Ego Modulation | 1.442 | 0.711 |
| + More blocks | **1.436** | **0.720** |

Current-to-Future Link 是未来生成的核心组件。单纯复制当前 BEV 加 ego motion 无法建模动态对象演化。

### 7.7 证据边界

实验支持：

- BEV 比直接多视角 token 更适合几何预测；
- world query、Current-to-Future Link 和几何正则化有效；
- HERMES++ 可以在一个框架中同时完成理解与未来点云生成；
- LLM 参数规模增加带来一定收益。

实验不能完全证明：

- HERMES++ 已经覆盖所有 driving world model 功能；
- 文本问答中的语义理解一定导致未来预测提升；
- 在真实闭环驾驶控制中一定优于专门 planner；
- 3 秒点云预测可直接代表长期世界建模能力。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **统一任务接口：** 用 BEV 连接多视角视觉、LLM 和 3D geometry。
2. **语义指导几何预测：** world queries 使理解分支的信息能进入未来生成分支。
3. **显式—隐式联合几何约束：** 同时约束深度输出和 latent feature structure。
4. **减少任务孤岛：** 理解和生成共享 visual tokenizer、LLM 和 Render。
5. **提升未来几何预测：** 相比 specialist generation model，3 秒 CD 更低。

### 8.2 论文和系统局限

- 当前输入以多视角图像为主，几何输出依赖训练数据和 Render；
- 未来点云预测不等同于完整可交互世界模型；
- LLM、BEV tokenizer 和 3D Render 训练复杂、算力成本高；
- 训练阶段依赖多个数据集和多阶段流程；
- 对未来不可观测事件和多主体交互的建模仍有限；
- arXiv 页面尚未提供代码链接，代码复现信息不完整。

### 8.3 分析者识别出的潜在问题

#### 问题一：语义理解与几何生成可能存在目标冲突

语言模型倾向于学习语义和文本流畅性，几何生成要求精确的空间结构。共享参数和联合训练可能出现：

```text
语言 loss 下降
但几何结构未必同步改善
```

#### 问题二：World Queries 的信息瓶颈

每个未来时间点只有有限 query token，而未来 BEV 是密集空间。Current-to-Future Link 需要从稀疏语义 query 恢复大规模空间细节，可能造成细粒度几何丢失。

#### 问题三：隐式几何先验的偏差传播

如果冻结的 geometry extractor 在某些物体、天气或稀疏区域上有偏差，cosine 和 Gram loss 可能把错误先验强制传给预测分支。

#### 问题四：Chamfer Distance 的局限

Chamfer Distance 主要衡量点集接近程度，但不能完整衡量：

- 点的语义类别；
- 物体 identity；
- 时间连续性；
- 碰撞可行性；
- 物理动力学一致性。

#### 问题五：BEV 压缩的细节损失

×4 下采样效果最好，但小目标、交通灯和远处物体可能仍被压缩。×8 的性能下降说明 token efficiency 与空间细节之间存在明显 trade-off。

#### 问题六：多数据集结果的可比性

理解和生成使用不同数据集、不同辅助标注与训练阶段。 specialist 对比也可能有不同输入模态和监督强度，因此不能只看表格中的绝对数值。

#### 问题七：世界知识不等于场景事实

LLM 的预训练知识可以帮助常识推理，但也可能产生 hallucination。几何生成必须优先服从当前 BEV 观测和物理约束，而不能让文本先验覆盖真实场景证据。

## 九、通俗讲解

### 9.1 普通世界模型和普通 LLM 的分工

普通世界模型擅长：

```text
现在的场景 → 未来场景
```

普通 LLM 擅长：

```text
场景 + 问题 → 文字回答
```

前者几何强、语言弱；后者语言强、几何未来预测弱。

### 9.2 HERMES++ 如何统一

HERMES++ 先把相机图像转换成 BEV：

```text
多视角图像 → 鸟瞰视图特征
```

BEV 既能给 LLM 看，也能还原成点云：

```text
BEV → LLM → 文字理解
BEV → Render → 3D 点云
```

### 9.3 World Queries 是什么

可以把 world queries 理解成“未来场景记忆槽”：

```text
当前图像 + 用户问题 + world queries
                  ↓ LLM
world queries 携带场景语义
                  ↓
未来几秒的几何预测
```

它们不是最终文字，而是把 LLM 理解结果带给几何生成分支的中间载体。

### 9.4 Current-to-Future Link 是什么

它负责把当前密集 BEV 推进未来：

```text
当前 BEV
  + 未来时间
  + ego motion
  + world queries
  + 文本语义
        ↓
未来 BEV
        ↓
未来点云
```

如果没有这个 Link，简单复制当前 BEV 很难预测车辆、行人和其他动态对象的变化。

### 9.5 为什么需要 Joint Geometric Optimization

只要求最终点云看起来接近 GT，内部 BEV 特征可能仍然不是真正几何结构。HERMES++ 增加：

- cosine loss：逐 voxel 对齐特征方向；
- Gram loss：对齐全局空间关系。

```text
输出几何对齐
+ 内部局部几何对齐
+ 内部全局结构对齐
        ↓
更稳定的 3D latent representation
```

### 9.6 一句话理解

> HERMES++ 让 LLM 负责理解“场景发生了什么”，让 BEV 和几何 Render 负责预测“未来 3D 世界如何变化”，并用 world queries 和 Current-to-Future Link 把两者连接起来。

## 十、综合评价与后续研究方向

### 10.1 综合评价

HERMES++ 的核心因果链为：

$$
\text{Multi-view images}
\rightarrow
\text{BEV tokenizer}
\rightarrow
\text{LLM understanding + world queries}
\rightarrow
\text{Current-to-Future Link}
\rightarrow
\text{future BEV}
\rightarrow
\text{shared BEV-to-Point Render}
\rightarrow
\text{future point clouds}
$$

它的主要价值不是简单地把 LLM 和 world model 并排放在一起，而是构造了共享几何接口和信息传递路径：

1. BEV 将多视角空间信息压缩为 LLM 可处理的 token；
2. World queries 从理解分支吸收视觉、文本和世界知识；
3. Current-to-Future Link 将稀疏语义上下文扩展到密集未来 BEV；
4. 共享 Render 将当前/未来 BEV 统一解码为点云；
5. Joint Geometric Optimization 约束显式输出和隐式特征结构。

实验显示，HERMES++ 在未来 3D 点云预测和场景理解上都具有竞争力：

- 1.8B 模型 3 秒 CD 为 1.01；
- 3.8B 模型 3 秒 CD 为 0.97；
- 1.8B 模型 CIDEr 为 0.749；
- 3.8B 模型 CIDEr 为 0.772；
- BEV、Current-to-Future Link 和联合几何优化均有消融收益。

但应避免把它直接称为已经完成的“通用驾驶世界模型”：

- 主要预测点云而不是完整可交互环境；
- 没有充分展示长期 rollout 和闭环控制；
- 未来多主体行为仍受观测不确定性限制；
- LLM 语义先验可能引入 hallucination；
- 训练系统复杂且代码尚未公开。

更准确的评价是：

> HERMES++ 提出了一种将 BEV 几何表示、LLM 语义理解和未来点云预测统一起来的驾驶世界模型框架，通过 world queries、Current-to-Future Link 和联合几何优化缩小了语言理解与物理模拟之间的鸿沟，但距离可交互、长时域和闭环可控的完整驾驶世界模型仍有明显距离。

### 10.2 后续研究方向

1. **长时域世界模型：** 从 3 秒点云预测扩展到多步 rollout，并抑制误差累积。
2. **动作条件生成：** 将转向、加速度、规划轨迹和导航指令作为显式 action condition。
3. **多主体交互建模：** 为车辆、行人和骑行者建立对象级 memory 与行为预测。
4. **可交互 4D 表示：** 从 point cloud 扩展到 occupancy、Gaussian、场景图和可编辑对象。
5. **几何—语言双向反馈：** 让几何预测反过来校正 LLM 的场景回答，减少 hallucination。
6. **不确定性估计：** 输出未来点云分布、置信度和多模态场景假设，而不是单一预测。
7. **更高效 token 化：** 研究对象感知 BEV token、稀疏 world queries 和自适应分辨率。
8. **真实闭环评估：** 在规划、控制和仿真闭环中验证未来几何预测是否真正改善驾驶决策。
9. **几何监督替代方案：** 比较冻结 extractor、对比学习、场景流和显式 3D consistency 的收益。
10. **模型规模与能力关系：** 系统研究 LLM 参数量、BEV 分辨率、query 数量和生成质量之间的 scaling law。

## 一句话结论

> HERMES++ 以 BEV 为统一接口，用 LLM 理解场景、world queries 传递语义、Current-to-Future Link 推演未来，并通过可微 Render 生成点云，构建了一个同时面向 3D 场景理解和未来几何生成的驾驶世界模型。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2604.28196](https://arxiv.org/abs/2604.28196)
- 论文 HTML：[https://arxiv.org/html/2604.28196v1](https://arxiv.org/html/2604.28196v1)
- 论文 PDF：[https://arxiv.org/pdf/2604.28196](https://arxiv.org/pdf/2604.28196)
- 关联 ICCV 2025 版本：论文 arXiv 页面注明为 HERMES 的扩展版本
