# DiffusionDriveV2：论文分析

> 来源：[arXiv:2512.07745](https://arxiv.org/abs/2512.07745)  
> 论文版本：v1，2025 年 12 月 8 日提交。当前为 arXiv 预印本，论文原文未说明正式发表的会议或期刊。  
> 代码：[github.com/hustvl/DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 标题 | **DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving** |
| 中文理解 | DiffusionDriveV2：强化学习约束的截断扩散端到端自动驾驶建模 |
| 作者 | Jialv Zou、Shaoyu Chen、Bencheng Liao、Zhiyu Zheng、Yuehao Song、Lefei Zhang、Qian Zhang、Wenyu Liu、Xinggang Wang |
| 研究方向 | 端到端自动驾驶、轨迹生成、扩散模型、强化学习 |
| 基础方法 | DiffusionDrive |
| 核心问题 | 轨迹多样性与轨迹质量难以同时保证 |
| 核心方法 | Scale-Adaptive Multiplicative Noise、Intra-Anchor GRPO、Inter-Anchor Truncated GRPO |
| 输入 | 多摄像头图像、LiDAR BEV 表示 |
| 输出 | 多模态未来驾驶轨迹及最终选择轨迹 |
| 数据集 | NAVSIM v1、NAVSIM v2 |
| 主要结果 | NAVSIM v1 PDMS 91.2；NAVSIM v2 EPDMS 85.5 |
| 骨干网络 | ResNet-34 |
| 推理扩散步数 | 2 步 |

## 二、极简全文核心总结

DiffusionDriveV2 针对端到端自动驾驶扩散模型中“轨迹多样但质量不稳定”或“质量稳定但模式坍缩”的矛盾，引入强化学习约束截断扩散生成器。方法使用尺度自适应乘性噪声进行平滑探索，并在同一驾驶意图内部使用 GRPO，在不同意图之间使用截断优势进行全局约束，最终结合模式选择器选出目标轨迹。在 NAVSIM v1/v2 上分别达到 91.2 PDMS 和 85.5 EPDMS。

## 三、研究背景与研究意义

### 3.1 端到端自动驾驶的轨迹生成问题

端到端自动驾驶希望直接学习：

$$
\text{传感器输入}\rightarrow\text{驾驶轨迹}
$$

轨迹通常表示为未来若干时刻的车辆位置：

$$
\tau=\{(x_n,y_n)\}_{n=1}^{N_f}
$$

其中，$N_f$ 是规划时域，$(x_n,y_n)$ 是第 $n$ 个未来 waypoint。

在复杂交通场景中，合理行为往往不止一种，例如直行或超车、左转或右转、保守跟车或主动变道。因此，规划模型需要生成多个具有不同驾驶意图的候选轨迹。

### 3.2 单模态方法的问题

传统回归式端到端模型通常只生成一条轨迹，无法表达驾驶行为的不确定性，也不能同时保留多个合理方案。面对交叉路口、变道、超车等场景时，一旦唯一预测失败，系统缺少备选方案。

### 3.3 扩散模型的问题：Mode Collapse

扩散模型理论上能够生成多种轨迹，但直接用于轨迹生成时可能出现模式坍缩：

```text
多种潜在驾驶意图 → 模型只生成一种高概率轨迹
```

结果是轨迹质量可能稳定，但缺乏行为多样性，难以覆盖超车、转弯、直行等不同策略。

### 3.4 DiffusionDrive 的改进与局限

DiffusionDrive 使用多个预定义 trajectory anchors 构造 Gaussian Mixture Model：

- 每个 anchor 对应一种驾驶意图；
- 不同 anchor 划分不同轨迹子空间；
- 可以生成直行、转弯、超车等多模态轨迹。

但它依赖 imitation learning。每个训练场景通常只有一条专家轨迹，因此与专家轨迹最接近的 anchor 得到正样本监督，其他 anchor 缺少明确的安全性和质量监督。于是可能产生：

```text
高质量轨迹 + 大量碰撞或越界轨迹
```

最终必须依赖下游 selector 过滤低质量候选轨迹。

### 3.5 论文要解决的核心矛盾

论文把问题概括为：

$$
\text{多样性}\quad\text{vs.}\quad\text{一致的高质量}
$$

| 方法 | 多样性 | 轨迹质量 |
|---|---:|---:|
| Vanilla Diffusion | 低 | 相对稳定 |
| DiffusionDrive | 高 | 质量不稳定，存在碰撞轨迹 |
| DiffusionDriveV2 | 较高 | 更稳定、更安全 |

论文的目标不是简单消除多样性，而是对所有模式施加质量约束，同时保留不同驾驶意图。

## 四、核心方法、模型、公式与整体流程

### 4.1 论文方案整体框架图

论文 Figure 2 展示了 DiffusionDriveV2 的总体架构：

![DiffusionDriveV2 overall architecture](https://arxiv.org/html/2512.07745v1/x2.png)

> **图 2：DiffusionDriveV2 整体架构。** 不同颜色的轨迹表示不同的 anchor 驾驶意图；实线表示高质量轨迹，虚线表示低质量轨迹。模型首先利用截断扩散解码器生成多模态候选轨迹，然后通过乘性探索噪声扩大邻近动作空间，再使用 Anchored Truncated GRPO 优化轨迹质量，最后由 mode selector 选择最符合目标的轨迹。图片来源：[论文 Figure 2](https://arxiv.org/html/2512.07745v1/x2.png)。

整体流程：

```text
多摄像头图像 + LiDAR BEV
          ↓
感知网络提取场景特征 z
          ↓
多个 trajectory anchors
          ↓
Anchor-conditioned truncated diffusion decoder
          ↓
生成不同驾驶意图的候选轨迹
          ↓
尺度自适应乘性高斯噪声
          ↓
轨迹探索
          ↓
奖励函数 / 闭环驾驶评价
          ↓
Intra-Anchor GRPO
          ↓
Inter-Anchor Truncated GRPO
          ↓
更新截断扩散策略
          ↓
Mode Selector
          ↓
选择最终目标轨迹
```

训练和推理的区别：

| 阶段 | 行为 |
|---|---|
| 强化学习训练 | 使用随机扩散和探索噪声，产生多样轨迹 |
| 轨迹优化 | 使用 Intra-Anchor GRPO 和 Inter-Anchor Truncated GRPO |
| 模式选择器训练 | 对候选轨迹进行粗到细排序 |
| 推理 | 使用确定性扩散设置，2 步生成候选轨迹，再由 selector 选择 |

### 4.2 Anchor-based Gaussian Mixture Model

DiffusionDrive 将轨迹分布表示为多个 anchor 条件下的高斯分布：

$$
p(\tau^k\mid \mathbf{a}_k,z)
=
\mathcal{N}
\left(
\tau^k
\mid
\mathbf{a}_k+\mu_k(z),
\Sigma_k(z)
\right)
$$

其中：

- $\mathbf{a}_k$：第 $k$ 个 trajectory anchor；
- $z$：场景特征；
- $\mu_k(z)$：相对于 anchor 的场景相关偏移；
- $\Sigma_k(z)$：轨迹不确定性；
- $\tau^k$：anchor $k$ 对应的轨迹。

整体轨迹分布为：

$$
p(\tau\mid z)
=
\sum_{k=1}^{N_{\text{anchor}}}
 s(\mathbf{a}_k\mid z)
 p(\tau^k\mid\mathbf{a}_k,z)
$$

其中 $s(\mathbf{a}_k\mid z)$ 是场景条件下选择第 $k$ 个驾驶意图的混合权重。

直观上：

```text
Anchor 1：直行
Anchor 2：左转
Anchor 3：右转
Anchor 4：超车
...
```

每个 anchor 负责一个局部行为分布。

### 4.3 截断扩散生成器

标准扩散模型通常需要较多去噪步骤。DiffusionDrive 使用截断扩散，只保留较少步骤：

$$
\tau_t^k
=
\sqrt{\bar{\alpha}_t}\mathbf{a}_k
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

其中 $t\in[1,T_{\text{trunc}}]$，$T_{\text{trunc}}\ll T$，$\bar\alpha_t$ 是噪声调度参数。

DiffusionDriveV2 继承 DiffusionDrive 的截断扩散生成器，并使用其 imitation learning 预训练权重作为 cold start。

### 4.4 将去噪过程建模为 MDP

论文将每个 anchor 下的扩散去噪过程建模为 MDP。第 $t$ 步策略为高斯分布：

$$
\pi_\theta
\left(
\tau_{t-1}^k
\mid
\tau_t^k,z,\mathbf{a}_k
\right)
=
\mathcal{N}
\left(
\tau_{t-1}^k;
\mu_\theta(\tau_t^k,t,z,\mathbf{a}_k),
\eta(1-\alpha_t)I
\right)
$$

其中 $\mu_\theta$ 是模型预测均值，$\eta$ 是扩散采样随机性参数，$\alpha_t$ 是预定义噪声调度，$\pi_\theta$ 是当前扩散策略。

由于策略是显式高斯分布，可以计算 $\log\pi_\theta$，因此可以使用 REINFORCE 或 GRPO 进行策略梯度更新。

### 4.5 尺度自适应乘性探索噪声

#### 4.5.1 加性噪声的问题

轨迹中近处 waypoint 和远处 waypoint 的空间尺度不同。如果对每个 waypoint 直接添加相同尺度的高斯噪声：

$$
\tau'=\tau+\epsilon_{\text{add}}
$$

容易产生局部折线、轨迹不平滑、车辆运动不连贯等问题。

#### 4.5.2 乘性噪声

论文提出只使用两个全局乘性噪声：纵向噪声和横向噪声：

$$
\tau'=(1+\epsilon_{\text{mul}})\tau
$$

其中：

$$
\epsilon_{\text{mul}}
=(\epsilon_{\text{long}},\epsilon_{\text{lat}})
$$

它不是对每个 waypoint 独立扰动，而是在整体轨迹层面进行缩放，因而更能保持连续性和平滑性。

```text
加性噪声：每个点各自乱动 → 轨迹变锯齿
乘性噪声：整条轨迹平滑变形 → 保留驾驶结构
```

训练阶段设置 $\eta=1$ 引入随机性；验证和推理阶段设置 $\eta=0$ 使用确定性采样。

### 4.6 Intra-Anchor GRPO

#### 4.6.1 直接使用 GRPO 的问题

普通 GRPO 通常将一组样本放在一起计算相对优势。但在 DiffusionDriveV2 中，不同 anchor 代表不同驾驶意图：

```text
直行轨迹 vs 左转轨迹
```

这两个轨迹不是简单的优劣关系。如果直接跨 anchor 比较，常见的“直行”可能因奖励更高而压制“左转”模式，最终造成模式坍缩。

#### 4.6.2 只在同一个 anchor 内比较

对于第 $k$ 个 anchor，生成 $G$ 条轨迹：

$$
\{\tau_0^{k,i}\}_{i=1}^{G}
$$

每条轨迹获得最终奖励：

$$
 r_{k,i}=R(\tau_0^{k,i})
$$

在同一 anchor 内计算组相对优势：

$$
 A_{k,i}
=
\frac{
 r_{k,i}-\operatorname{mean}(\{r_{k,1},\ldots,r_{k,G}\})
}{
\operatorname{std}(\{r_{k,1},\ldots,r_{k,G}\})
}
$$

这样直行只与其他直行候选比较，左转只与其他左转候选比较，不会因驾驶意图不同而错误比较。

#### 4.6.3 Intra-Anchor RL 损失

$$
\mathcal{L}_{RL}
=
-\frac{1}{N_{\text{anchor}}}
\sum_k
\frac{1}{G}\sum_i
\frac{1}{T}\sum_t
\gamma^{t-1}
\log\pi_\theta
\left(\tau_{t-1}^{k,i}\mid\tau_t^{k,i}\right)
A_{k,i}
$$

其中 $\gamma^{t-1}$ 是去噪折扣系数，$A_{k,i}$ 是同一 anchor 内的相对优势。论文设置 $\gamma=0.8$，降低较早、噪声较大的去噪步骤对训练的影响。

#### 4.6.4 与 imitation learning 联合

为防止 RL 过度偏离原始驾驶能力，论文将 RL 损失与 imitation learning 损失结合：

$$
\mathcal{L}=\mathcal{L}_{RL}+\lambda\mathcal{L}_{IL}
$$

其中 $\mathcal{L}_{RL}$ 负责强化更高质量、更安全的轨迹，$\mathcal{L}_{IL}$ 防止模型忘记基本驾驶能力。

### 4.7 Inter-Anchor Truncated GRPO

#### 4.7.1 Intra-Anchor 的新问题

只在单个 anchor 内比较可以避免模式坍缩，但也会丢失全局可比性。例如某个 anchor 内最好的轨迹只是安全但一般，而另一个 anchor 内最好的轨迹发生碰撞；如果只进行组内比较，碰撞轨迹可能仍得到正优势。

#### 4.7.2 截断优势

论文定义：

$$
 A^{\text{trunc}}_{k,i}
=
\begin{cases}
-1, & \text{如果轨迹发生碰撞}\\
\max(0,A_{k,i}), & \text{其他情况}
\end{cases}
$$

含义是：碰撞轨迹无论属于哪个 anchor 都施加强惩罚；所有负优势被截断为 0；只有相对表现较好的非碰撞轨迹得到正向强化。

这是一种折中：

$$
\text{同 anchor 内相对优化}
+
\text{跨 anchor 的绝对安全约束}
$$

### 4.8 Mode Selector

强化学习生成多个候选轨迹后，仍需要最终选择器。其输入包括候选轨迹坐标、BEV 特征、agent queries 和 map queries：

```text
候选轨迹坐标
    ↓
与 BEV 特征进行 deformable cross-attention
    ↓
与 agent/map queries 进行 cross-attention
    ↓
MLP 预测轨迹分数
    ↓
选择最符合目标的轨迹
```

论文采用两阶段 coarse-to-fine selector：

1. 粗粒度 selector 先选出 Top-k；
2. 细粒度 selector 对 Top-k 轨迹进一步排序。

训练损失包括 BCE loss 和 Margin-Rank loss：

$$
\mathcal{L}_{rank}
=
\frac{1}{N}\sum_{i,j}
\max
\left(
0,
-\operatorname{sign}(s_i-s_j)(\hat{s}_i-\hat{s}_j)+m
\right)
$$

其中 $s_i,s_j$ 是真实轨迹质量分数，$\hat{s}_i,\hat{s}_j$ 是预测分数，$m$ 是正的 margin。

## 五、核心创新点与传统方法对比

### 5.1 用 RL 约束所有轨迹模式

DiffusionDrive 的 imitation learning 主要监督与专家轨迹对应的正模式。DiffusionDriveV2 使用轨迹级奖励约束所有 anchor 和所有候选轨迹：

```text
正模式：继续优化
负模式：惩罚碰撞和低质量行为
潜在更优轨迹：通过探索发现
```

### 5.2 Intra-Anchor GRPO

不在不同驾驶意图之间直接比较，而是在同一 anchor 内进行 GRPO，避免“直行”压制“转弯”。

### 5.3 Inter-Anchor Truncated GRPO

通过截断优势实现有限的跨模式约束：碰撞轨迹统一惩罚，非碰撞负优势不再继续惩罚，从而保留不同模式并提供全局安全性信号。

### 5.4 尺度自适应乘性噪声

相比逐 waypoint 添加独立噪声，乘性噪声在整体轨迹层面进行变形，更适合车辆轨迹探索。

### 5.5 保持 2 步推理效率

论文保留截断扩散模型的高效特性：

$$
\text{多模态生成}+
\text{强化学习优化}+
\text{2 步推理}
$$

### 5.6 方法对比

| 方法 | 轨迹多样性 | 质量一致性 | 是否需要 selector | 主要问题 |
|---|---:|---:|---:|---|
| Vanilla Diffusion | 低 | 较高 | 通常需要 | Mode collapse |
| DiffusionDrive | 高 | 较低 | 需要 | 大量负模式缺少监督 |
| DiffusionDriveV2 | 高 | 高 | 需要 | 训练成本和奖励依赖较高 |
| 回归式规划器 | 低 | 稳定 | 不一定 | 无法表示多种意图 |

## 六、理论分析与关键假设

### 6.1 论文的理论逻辑

论文的核心不是给出严格收敛定理，而是提出针对模式结构的策略优化原则：

1. 不同 anchor 代表不同意图；
2. 不同意图不应直接进行普通组内比较；
3. 同一意图内可以进行相对优势估计；
4. 碰撞属于跨意图都应惩罚的绝对失败；
5. 非碰撞轨迹主要通过正优势进行强化。

因此，方法对应的优势结构是：

$$
\text{局部相对比较}+
\text{全局绝对安全约束}
$$

### 6.2 关键假设

- **Anchor 能够对应稳定的驾驶意图。** 例如直行、左转、右转和超车。
- **同一 anchor 内的轨迹具有可比较性。** 只有同一 anchor 中的候选轨迹确实属于相近行为模式，组内优势才有合理含义。
- **碰撞检测是可靠的。** 碰撞轨迹直接赋值为 $-1$，误报和漏报会直接影响策略更新。
- **闭环奖励能够反映轨迹质量。** 如果仿真器奖励与真实驾驶安全性不一致，策略可能过拟合模拟器目标。
- **imitation learning 能提供稳定的基础策略。** RL 从 DiffusionDrive 预训练权重开始。

### 6.3 方法没有保证的内容

论文方法不能严格保证：

- 所有生成轨迹都无碰撞；
- 所有驾驶意图都被均匀覆盖；
- anchor 一定对应唯一驾驶意图；
- 训练奖励提升一定对应真实道路性能提升；
- RL 训练一定不破坏原始驾驶能力；
- 2 步截断扩散在所有场景下都足够；
- selector 在分布外场景中仍然可靠。

### 6.4 理论与工程边界

论文的“解决多样性—质量矛盾”主要是实验性结论，而不是普适定理。它成立依赖奖励设计、碰撞检测、anchor 划分、探索噪声、RL/IL 损失权重、selector 质量和 NAVSIM 闭环评估环境。

## 七、实验设计与结果分析

### 7.1 数据集与输入

论文使用 NAVSIM v1 和 NAVSIM v2。输入包括 8 个摄像头组成的 360°视场、5 个 LiDAR 合并得到的点云、三张裁剪和缩放后的前视图像，以及光栅化 LiDAR BEV 表示。

实现设置：

- ResNet-34；
- 与 DiffusionDrive 相同规模的扩散解码器；
- DiffusionDrive 预训练权重；
- RL 训练 10 个 epoch；
- 8 张 NVIDIA L20；
- batch size 为 512；
- AdamW 学习率为 $2\times10^{-4}$。

### 7.2 NAVSIM v1 主结果

| 方法 | Backbone | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| DiffusionDrive | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| DIVER | ResNet-34 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| DriveSuprim | ResNet-34 | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| DiffusionDriveV2 | ResNet-34 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | **91.2** |

论文报告：

- 相比 DiffusionDrive，PDMS 提升 3.1；
- EP 提升 5.3；
- 相比 DIVER，PDMS 提升 2.9；
- 使用 ResNet-34 仍超过部分使用更大 V2-99 backbone 的方法。

这里的“新纪录”是论文在其 NAVSIM v1、ResNet-34 对齐实验设置下的报告，不应脱离具体 backbone 和评测协议泛化。

### 7.3 NAVSIM v2 主结果

| 方法 | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | **85.5** |

### 7.4 多样性与质量分析

论文定义轨迹多样性指标：

$$
D^{raw}_{iv,n}
=
\frac{2}{M(M-1)}
\sum_{i=1}^{M-1}
\sum_{j=i+1}^{M}
\left\|\mathbf{p}_n^i-\mathbf{p}_n^j\right\|_2
$$

然后归一化：

$$
D_{iv,n}
=
\min
\left(
1,
\frac{D^{raw}_{iv,n}}
{\epsilon+\frac{1}{M}\sum_{m=1}^{M}\|\mathbf{p}_n^m\|_2}
\right)
$$

结果：

| 方法 | Diversity | PDMS@1 | PDMS@5 | PDMS@10 |
|---|---:|---:|---:|---:|
| Transfuser | 0.1 | 85.7 | 85.7 | 85.7 |
| DiffusionDrive | 42.3 | 93.5 | 84.3 | 75.3 |
| DiffusionDriveV2 | 30.3 | 94.9 | 91.1 | 84.4 |

解释：Transfuser 几乎只生成一条确定性轨迹；DiffusionDrive 多样性最高但 Top-10 质量下降明显；DiffusionDriveV2 的 Diversity 虽低于 DiffusionDrive，但 PDMS@5 和 PDMS@10 显著更高，说明它牺牲了一部分无约束的轨迹差异，换取更稳定的候选轨迹质量。

### 7.5 探索噪声消融

| 噪声类型 | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| Additive | 98.1 | 97.2 | 94.4 | 99.9 | 85.3 | 89.7 |
| Multiplicative | 98.1 | 97.6 | 94.5 | 99.9 | 85.7 | **90.1** |

乘性噪声相较加性噪声将 PDMS 从 89.7 提升到 90.1，支持其更适合保持轨迹结构的主张。

### 7.6 Intra-Anchor GRPO 消融

| Intra-Anchor GRPO | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| 不使用 | 97.9 | 97.3 | 93.8 | 99.5 | 84.9 | 89.2 |
| 使用 | 98.1 | 97.6 | 94.5 | 99.9 | 85.7 | **90.1** |

说明在 anchor 划分的多模态轨迹空间中，组内优势估计很重要。

### 7.7 Inter-Anchor Truncated GRPO 消融

| Inter-Anchor Truncated GRPO | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| 不使用 | 97.7 | 97.3 | 93.6 | 99.9 | 85.7 | 89.5 |
| 使用 | 98.1 | 97.6 | 94.5 | 99.9 | 85.7 | **90.1** |

说明仅有 Intra-Anchor GRPO 仍缺少全局安全性约束；增加跨 anchor 截断优势后，性能进一步提升。

### 7.8 Mode Selector 消融

| 模型 | 设计 | PDMS |
|---|---|---:|
| $M_0$ | 基础 selector | 89.7 |
| $M_1$ | $M_0$ + coarse-to-fine | 89.9 |
| $M_2$ | $M_1$ + rank loss | 90.1 |

coarse-to-fine selector 带来约 0.2 PDMS 提升，rank loss 再带来约 0.2 PDMS 提升。即便给 DiffusionDrive 使用同样的 selector，其 PDMS 为 89.1，仍低于 DiffusionDriveV2 的 91.2，说明性能提升主要来自 RL 轨迹生成，而不仅是 selector 变复杂。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **解决扩散轨迹生成中的质量—多样性矛盾。** 不是简单追求更多轨迹，而是让多模态轨迹整体更可靠。
2. **将 GRPO 迁移到截断扩散模型。** 识别并处理不同驾驶意图之间不能直接比较的问题。
3. **探索噪声符合轨迹结构。** 乘性噪声避免 waypoint 级独立噪声造成轨迹折断。
4. **提升候选轨迹整体质量。** 不只提升 Top-1，也关注 Top-5、Top-10，减少对 selector 的过度依赖。

### 8.2 论文自身暴露的限制

- 训练依赖闭环仿真器和奖励指标；
- 仍然需要 mode selector；
- 依赖预训练 DiffusionDrive 作为 cold start；
- anchor 数量和质量影响多模态能力；
- 强化学习训练成本高于单纯 imitation learning；
- 论文主要在 NAVSIM 上验证。

### 8.3 分析者识别出的潜在问题

#### 多样性下降可能被误解为质量提升

DiffusionDriveV2 的 Diversity 为 30.3，低于 DiffusionDrive 的 42.3。论文将其解释为去除了大量低质量模式，但也可能意味着部分有价值的行为多样性被 RL 压缩。需要进一步评估每个 anchor 的有效性、不同驾驶意图的覆盖率、长尾行为是否被削弱，以及极端场景下是否过度保守。

#### Anchor 可能限制行为空间

Anchor 由专家行为聚类得到，模型的多模态能力受 anchor vocabulary 限制。若真实场景出现训练数据未覆盖的新行为，模型可能无法生成合适的候选轨迹。

#### 碰撞惩罚过于离散

论文对碰撞轨迹直接设置 $A^{\text{trunc}}=-1$，无法区分轻微擦碰、高速正面碰撞、即将碰撞但尚未碰撞，以及仅仅舒适性较差的轨迹。更连续的安全奖励可能提供更细粒度的学习信号。

#### 闭环指标可能掩盖具体失败模式

PDMS 和 EPDMS 是综合指标。分数提升不能说明每类交通场景都提升、长尾风险都降低、分布外场景更安全，或真实道路部署风险降低。

#### Mode Selector 仍是潜在单点故障

论文批评 DiffusionDrive 过度依赖 selector，但 DiffusionDriveV2 仍然保留 selector。其在 OOD 场景中的可靠性仍需单独评估。

#### RL 和 IL 损失的权重敏感

联合目标为：

$$
\mathcal{L}=\mathcal{L}_{RL}+\lambda\mathcal{L}_{IL}
$$

$\lambda$ 太大时 RL 改进能力受限；$\lambda$ 太小时模型可能偏离原始驾驶分布。论文没有在主文中充分展示 $\lambda$ 的敏感性分析。

#### 真实部署差距

NAVSIM 是闭环仿真评测。现实部署还会受到感知误差、车辆动力学误差、传感器延迟、控制器误差、交通参与者不可预测行为，以及仿真器奖励与现实安全目标不一致等影响。

## 九、通俗讲解

可以把 DiffusionDriveV2 理解成“让自动驾驶模型先提出多种驾驶方案，再通过强化学习把危险方案淘汰掉”。

### 9.1 普通模型的问题

前方是路口时，模型可能需要考虑：

```text
方案 A：直行
方案 B：左转
方案 C：右转
方案 D：超车
```

普通扩散模型可能只输出直行，这叫模式坍缩。

DiffusionDrive 通过多个 anchor 强迫模型生成多种方案，因此能够得到直行、左转、右转和超车，但其中有些轨迹可能会撞车、压线或驶出可行驶区域，因为 imitation learning 只知道专家轨迹是好的，不知道其他模式如何安全行驶。

### 9.2 DiffusionDriveV2 如何改进

它给每个 anchor 生成多条候选轨迹。例如对“左转”这个 anchor 生成：

```text
左转轨迹 1：安全但保守
左转轨迹 2：平滑且高效
左转轨迹 3：发生碰撞
左转轨迹 4：偏离车道
```

然后只在“左转组”内部比较，不会把“左转”直接拿去和“直行”比较。

### 9.3 为什么还需要跨 anchor 约束

如果完全只在组内比较，某个 anchor 中最好的轨迹即使很差，也可能获得正奖励。因此论文增加规则：

```text
发生碰撞 → 无论属于哪个 anchor，都强惩罚
其他低于组内平均的轨迹 → 不再继续强惩罚
组内表现好的轨迹 → 强化
```

这样既保留不同驾驶意图，又能统一约束绝对危险行为。

### 9.4 为什么使用乘性噪声

如果给轨迹上的每个点单独加噪声，轨迹会变成锯齿。乘性噪声则对整条轨迹进行平滑缩放和偏移，更符合车辆运动规律。

### 9.5 最终输出

经过 RL 优化后，模型仍然生成多种方案，但候选轨迹整体更可靠。最后由 mode selector 根据场景目标选择最合适的一条。

## 十、综合评价与后续研究方向

### 10.1 综合评价

DiffusionDriveV2 的核心贡献可以概括为：

$$
\text{Anchor 多模态生成}
+
\text{结构化探索}
+
\text{组内 GRPO}
+
\text{跨模式安全约束}
$$

它没有简单地把普通 GRPO 直接套到扩散轨迹模型上，而是识别出关键问题：不同 anchor 代表不同驾驶意图，因此不能直接用同一个组内基线比较所有轨迹。

针对这个问题，论文设计了 Intra-Anchor GRPO、Inter-Anchor Truncated GRPO、乘性探索噪声和 Mode Selector。

实验结果显示：

- NAVSIM v1：91.2 PDMS；
- NAVSIM v2：85.5 EPDMS；
- PDMS@1：94.9；
- PDMS@10：84.4；
- 相比 DiffusionDrive，轨迹整体质量明显提升；
- 相比单模态模型，仍保留一定多样性；
- 推理仍只需 2 个扩散步骤。

总体评价：

> 这是一个针对“多模态轨迹生成中的质量不稳定”问题设计得较完整的方法。它将强化学习的探索—约束机制与 anchor 化的扩散轨迹生成结合起来，并避免了跨驾驶意图的错误优势比较。

但它仍然依赖 anchor 设计、闭环仿真奖励、碰撞检测、RL/IL 权重、selector 和预训练模型质量。因此，它更准确地解决了 NAVSIM 评测下的质量—多样性折中，而不是从理论上彻底解决所有自动驾驶多模态规划问题。

### 10.2 后续研究方向

#### 方向一：自适应 Anchor

研究场景条件下动态生成 anchor、可学习 anchor、新行为自动发现，以及根据交通参与者和道路结构动态调整 anchor 数量。

#### 方向二：更细粒度的安全奖励

将碰撞惩罚从离散的 $-1$ 扩展为连续风险函数：

$$
R_{\text{safety}}
=f(\text{TTC},\text{距离},\text{相对速度},\text{碰撞概率})
$$

#### 方向三：多目标强化学习

自动驾驶同时需要优化安全、进度、舒适性、车道遵循、交通规则和行为多样性，可以研究多目标 GRPO 或 Pareto 优化。

#### 方向四：减少对 Mode Selector 的依赖

可以让生成器本身直接输出质量校准后的轨迹分布，或将 selector 与生成器联合训练，降低 selector 成为单点故障的风险。

#### 方向五：分布外场景泛化

在未见道路结构、未见交通规则、极端天气、稀有交通参与者行为、未见驾驶意图和仿真到真实环境迁移中验证方法。

#### 方向六：真实闭环部署

进一步研究车辆动力学约束、轨迹可执行性、控制延迟、感知不确定性、实车安全验证和与传统安全模块的协同。

#### 方向七：提高 RL 训练效率

研究离线轨迹数据预训练、world model rollout、价值模型辅助优势估计、更低方差的 GRPO、轨迹级 reward caching 和分阶段 curriculum RL。

## 一句话结论

> DiffusionDriveV2 通过“同意图内 GRPO + 跨意图碰撞截断惩罚 + 平滑乘性探索噪声”，在保留多模态驾驶行为的同时提升候选轨迹整体质量，是对 DiffusionDrive 中“多样但不安全”问题的一种针对性强化学习解决方案。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2512.07745](https://arxiv.org/abs/2512.07745)
- 论文 HTML：[https://arxiv.org/html/2512.07745v1](https://arxiv.org/html/2512.07745v1)
- 论文 PDF：[https://arxiv.org/pdf/2512.07745](https://arxiv.org/pdf/2512.07745)
- 代码：[https://github.com/hustvl/DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)
