# ResAD：论文分析

> 标题：*ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving*  
> 来源：[arXiv:2510.08562](https://arxiv.org/abs/2510.08562)  
> 版本：v2，2025 年 11 月 22 日修订；论文于 2025 年 10 月 9 日首次提交  
> 作者：Zhiyu Zheng、Shaoyu Chen、Haoran Yin、Xinbang Zhang、Jialv Zou、Xinggang Wang、Qian Zhang、Lefei Zhang  
> 单位：武汉大学、地平线机器人、华中科技大学  
> 论文原文未说明正式发表的会议或期刊。  
> 代码：论文摘要称代码将发布，但当前论文页面未提供正式代码仓库链接。

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | 端到端自动驾驶、扩散式轨迹规划、多模态规划 |
| 核心问题 | 直接预测完整轨迹带来的时空尺度不均衡与错误相关性 |
| 核心方法 | Normalized Residual Trajectory Modeling（NRTM） |
| 关键模块 | Trajectory Residual Modeling、Point-wise Residual Normalization、Inertial Reference Perturbation、Trajectory Ranker |
| 基础模型 | Transfuser-style camera-LiDAR encoder + diffusion decoder |
| 物理先验 | 基于当前速度的恒速惯性参考轨迹 |
| 输出方式 | 生成多条候选轨迹，再由 ranker 选择最终轨迹 |
| NAVSIM v1 | ResNet-34：88.8 PDMS；V2-99：90.6 PDMS |
| NAVSIM v2 | ResNet-34：85.5 EPDMS |
| 推理采样 | DDIM，2 个去噪步骤 |
| 训练模式数 | $K_{train}=20$ |
| 测试模式数 | $K_{infer}=200$ |

## 二、极简全文核心总结

ResAD 不直接预测完整未来轨迹，而是先根据车辆当前速度生成恒速惯性参考，再学习真实轨迹相对该参考的残差。为避免远期大尺度误差主导训练，模型对残差进行逐点归一化；同时扰动初始速度生成多个惯性参考，以获得上下文相关的多模态轨迹。实验表明，ResAD 在 NAVSIM v1/v2 上取得较强结果，并在仅 2 步 DDIM 去噪下实现高效规划。

## 三、研究背景与研究意义

### 3.1 端到端自动驾驶

端到端自动驾驶希望直接学习：

$$
\text{传感器观测}
\rightarrow
\text{未来自车轨迹}
\rightarrow
\text{控制指令}
$$

相比感知—预测—规划模块化系统，端到端方法可以减少中间模块的信息损失和误差累积。但直接预测轨迹仍有两个问题。

### 3.2 问题一：错误相关性

直接从高维图像、LiDAR 特征预测完整轨迹时，模型可能学习表面相关性，而不是安全驾驶逻辑。例如：

- 模型看到前车刹车灯，于是学习“跟随刹车”；
- 但模型没有真正理解前方红灯；
- 在前车闯红灯时，模型可能错误跟随。

论文的观点是：让网络从零学习完整轨迹，优化负担过重，容易利用 shortcut。

### 3.3 问题二：规划时域不平衡

近处 waypoint 通常更重要、更确定；远处 waypoint 不确定性更高。直接使用完整轨迹误差时，远期较大的坐标误差可能产生更大的损失：

$$
\text{远期大误差}
\rightarrow
\text{更强梯度}
\rightarrow
\text{模型忽略近场安全细节}
$$

这就是论文所说的 planning horizon dilemma。

### 3.4 论文的解决思路

将问题从：

> 未来完整轨迹是什么？

改写成：

> 相对于车辆自然惯性运动，场景要求车辆做什么修正？

即：

$$
\text{完整轨迹}
=
\text{惯性参考}
+
\text{学习到的残差}
$$

## 四、核心方法、模型、公式与流程

### 4.1 ResAD 整体框架图

![ResAD framework](https://arxiv.org/html/2510.08562v2/x2.png)

> **图 2：ResAD 框架。** 多视角图像和 LiDAR 经过 Transfuser-style encoder 得到场景特征；模型根据车辆当前状态生成惯性参考，并通过 Inertial Reference Perturbation 得到多个参考假设；Diffusion Decoder 在参考轨迹条件下预测归一化残差；最终由 Trajectory Ranker 从候选轨迹中选出结果。图片来源：[论文 Figure 2](https://arxiv.org/html/2510.08562v2/x2.png)。

整体流程：

```text
Camera + LiDAR + Ego State
          ↓
Transfuser-style Encoder
          ↓
Scene Features
          ↓
当前速度 v0
          ↓
恒速 Inertial Reference
          ↓
Inertial Reference Perturbation
          ↓
K 个候选惯性参考
          ↓
计算 GT 相对参考的 residual
          ↓
Point-wise Residual Normalization
          ↓
Diffusion Decoder
          ↓
归一化残差 → 反归一化
          ↓
参考轨迹 + 残差
          ↓
K 条候选轨迹
          ↓
Trajectory Ranker
          ↓
最终轨迹
```

### 4.2 扩散模型预备知识

ResAD 使用 vanilla diffusion。前向加噪过程可写为：

$$
q(z_t\mid z_0)
=
\mathcal{N}\left(
 z_t;
 \sqrt{\bar\alpha_t}z_0,
 (1-\bar\alpha_t)I
\right)
$$

其中：

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s
$$

$\beta_t$ 控制每一步加入的噪声量。训练时，模型根据带噪样本、场景条件和 diffusion timestep 预测干净的数据样本；推理时从高斯噪声开始逐步去噪。

需要注意：论文使用 DDPM 训练、DDIM 推理；训练扩散步数为 $T=1000$，但测试只使用 2 个 DDIM 去噪步骤。

### 4.3 Trajectory Residual Modeling（TRM）

#### 4.3.1 惯性参考轨迹

当前自车速度为：

$$
\mathbf{v}_0=(v_{x,0},v_{y,0})
$$

当前位置为：

$$
\mathbf{p}_0=(x_0,y_0)
$$

在恒速模型下，未来 $t_i$ 时刻的位置为：

$$
\mathbf{p}_{t_i}^{ref}
=
\mathbf{p}_0+
\mathbf{v}_0\Delta t_i
$$

于是得到惯性参考轨迹：

$$
\tau_{ref}
=
\left\{
\mathbf{p}_{t_i}^{ref}
\right\}_{i=1}^{N}
$$

它表示在没有主动控制修正时，车辆按照当前速度自然前进的轨迹。

#### 4.3.2 轨迹残差

给定专家/真实轨迹 $\tau_{gt}$，残差为：

$$
\mathbf{r}
=
\tau_{gt}-\tau_{ref}
$$

逐点写为：

$$
\mathbf{r}_{t_i}
=
\mathbf{p}_{t_i}^{gt}-\mathbf{p}_{t_i}^{ref}
$$

模型不再直接学习完整轨迹，而是学习场景所需的控制偏离：

```text
直行巡航       → 残差较小
遇到红灯       → 残差体现减速/停车趋势
急转弯         → 残差体现横向偏转
前方障碍物     → 残差体现绕行
并线/汇入      → 残差体现横向调整
```

#### 4.3.3 方法的实际含义

惯性参考并不是最终轨迹，也不是严格物理仿真。它只是一个由当前状态确定的 baseline。网络仍需根据：

- 交通灯；
- 道路曲率；
- 障碍物；
- 让行规则；
- 导航方向；
- 其他交通参与者

预测必要的 residual。

### 4.4 Point-wise Residual Normalization（PRNorm）

#### 4.4.1 动机

即使改成 residual，残差在不同时间点仍可能存在尺度差异。论文按 $x/y$ 分量，在整个训练集、全部轨迹和全部时间步上统计极值：

$$
 r_d^{min}
=
\min_{j,t}r_{j,t,d},
\qquad
 r_d^{max}
=
\max_{j,t}r_{j,t,d}
$$

其中 $d\in\{x,y\}$。

#### 4.4.2 归一化公式

将每个残差分量映射到 $[-\gamma,\gamma]$：

$$
\tilde r_{t,d}
=
2\gamma
\frac{r_{t,d}-r_d^{min}}
{r_d^{max}-r_d^{min}+\epsilon_0}
-\gamma
$$

其中：

- $\tilde r_{t,d}$：归一化残差；
- $\gamma>0$：输出范围控制超参数；
- $\epsilon_0$：避免分母为零的稳定项。

反归一化为：

$$
 r_{t,d}
=
\left(
\frac{\tilde r_{t,d}+\gamma}{2\gamma}
\right)
\left(r_d^{max}-r_d^{min}+\epsilon_0\right)
+r_d^{min}
$$

#### 4.4.3 PRNorm 的作用

PRNorm 改变了各时间点和坐标分量对优化目标的相对贡献：

```text
原始残差：远期大幅误差可能主导 loss
PRNorm：各分量映射到相近数值范围
结果：近场的小幅安全修正更容易获得有效梯度
```

它不是简单的标准化 z-score，而是基于训练集极值的 component-wise min-max scaling。

### 4.5 Inertial Reference Perturbation（IRP）

#### 4.5.1 生成多个惯性参考

从零均值高斯分布采样速度扰动：

$$
\delta\mathbf{v}_k
\sim
\mathcal{N}(0,\Sigma)
$$

其中：

$$
\Sigma
=
\operatorname{diag}
(\sigma_{v_x}^2,\sigma_{v_y}^2)
$$

构造第 $k$ 个扰动速度：

$$
\mathbf{v}_{0,k}'
=
\mathbf{v}_0+\delta\mathbf{v}_k
$$

再用恒速模型得到多个参考轨迹：

$$
\tau_{ref,k}
=\operatorname{InertialReference}
(\mathbf{p}_0,\mathbf{v}_{0,k}')
$$

对应残差为：

$$
\mathbf{r}_k
=\tau_{gt}-\tau_{ref,k}
$$

#### 4.5.2 IRP 的双重作用

1. **多模态生成：** 不同参考轨迹对应不同初始意图，模型可以产生不同候选行为。
2. **传感器鲁棒性：** 初始速度扰动模拟 GPS/IMU 等 ego-state 误差。

与固定 trajectory vocabulary 的方法相比，IRP 直接围绕当前状态和场景生成候选参考，不需要对大量无关轨迹进行统一评估。

### 4.6 Diffusion Decoder 的训练和推理

对第 $k$ 个归一化残差添加噪声：

$$
\mathbf{z}_k^{(i)}
=
\sqrt{\bar\alpha_i}\tilde{\mathbf{r}}_k
+
\sqrt{1-\bar\alpha_i}\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)
$$

Diffusion Decoder 接收：

- noisy normalized residual；
- encoder query features；
- diffusion timestep embedding；
- 对应惯性参考的 positional encoding。

输出去噪后的残差：

$$
\{\hat{\mathbf{r}}_k\}_{k=1}^{K}
=
f_\theta
\left(
\{\mathbf{z}_k^{(i)}\}_{k=1}^{K},c
\right)
$$

训练损失为：

$$
\mathcal{L}_{diff}
=
\sum_{k=1}^{K}
\mathcal{L}_{rec}
(\hat{\mathbf{r}}_k,\mathbf{r}_k)
$$

论文指出 $\mathcal{L}_{rec}$ 可以使用 L1 或 MSE。论文正文没有在该处明确固定一种形式，因此不能将其简单写成唯一的 MSE 目标。

推理时：

1. 从 $K_{infer}$ 组高斯噪声开始；
2. 进行 2 步 DDIM 去噪；
3. 得到归一化 residual；
4. 反归一化；
5. 加回对应的扰动惯性参考；
6. 得到 $K_{infer}$ 条候选轨迹。

最终轨迹为：

$$
\hat\tau_k
=
\tau_{ref,k}+\operatorname{PRNorm}^{-1}
(\hat{\tilde r}_k)
$$

### 4.7 Multimodal Trajectory Ranker

Diffusion Decoder 生成多条候选轨迹，ranker 负责选择最终输出。

候选轨迹先经过位置编码：

$$
\mathcal{V}=\operatorname{PosEmb}(v_k)
$$

再与 encoder 的环境表示进行 Transformer 交互：

$$
\mathcal{V}'
=
\operatorname{Transformer}
(Q=\mathcal{V},K=E_{env},V=E_{env})+E
$$

其中 $E$ 是 ego status embedding。

之后使用多个 MLP heads 预测每条候选轨迹的 metric scores，例如：

- no-at-fault collision；
- drivable area compliance；
- time to collision；
- ego progress；
- comfort；
- NAVSIM v2 扩展指标。

训练 ranker 时，论文结合两类监督：

1. 基于轨迹与 GT 距离构造的软目标；
2. rule-based planner 生成的 metric score。

论文给出的轨迹距离软目标为：

$$
 y_i
=
\frac{
\exp\left(-\|\tau_{gt}-\hat\tau_i\|_2^2\right)
}
{
\sum_{j=1}^{K}
\exp\left(-\|\tau_{gt}-\hat\tau_j\|_2^2\right)
}
$$

正文公式中的 ranker loss 排版存在符号混乱，但语义是：

- 用轨迹距离软标签进行候选选择监督；
- 用 BCE 预测规则教师的各项 metric；
- 推理时按 NAVSIM 指标权重聚合 metric，选择最高分轨迹。

因此，ranker 不是简单按照欧氏距离选择最近 GT 的轨迹，而是学习兼顾安全、道路合规、进度和舒适性的选择策略。

## 五、核心创新点与传统方法对比

### 5.1 Trajectory Residual Modeling

将完整轨迹分解为：

$$
\tau_{gt}
=
\tau_{ref}+r
$$

模型只学习相对于物理 baseline 的必要修正，降低学习难度。

### 5.2 Point-wise Residual Normalization

通过逐坐标、逐分量的 min-max normalization，使残差不同时间点具有更平衡的数值范围，缓解远期误差主导优化的问题。

### 5.3 Inertial Reference Perturbation

不使用固定的全局 trajectory vocabulary，而是对当前速度进行随机扰动，生成当前场景相关的多个参考假设。

### 5.4 Residual Diffusion + Ranker

Diffusion Decoder 负责产生多模态候选，Trajectory Ranker 负责从候选中选择安全且符合规则的轨迹。两者角色分离：

```text
生成器：尽量产生多样、合理的候选
排序器：选择当前场景最优候选
```

### 5.5 与其他规划范式对比

| 方法 | 候选生成 | 主要表示 | 多模态来源 | 主要问题 |
|---|---|---|---|---|
| Transfuser | 直接回归 | 完整轨迹 | 通常单模态 | 难表达多种驾驶意图 |
| Hydra-MDP | 静态词典 | trajectory anchors | 固定词典 | 大量候选与场景无关 |
| DiffusionDrive | 扩散生成 | 轨迹/噪声 | 扩散采样 | 可能产生无效候选 |
| SparseDriveV2 | Path × Velocity 评分 | 因子化词典 | 组合词典 | 仍受静态词典限制 |
| ResAD | 残差扩散 | inertial reference + residual | 参考扰动 + diffusion | 依赖恒速先验和归一化统计 |

## 六、理论分析与关键假设

### 6.1 惯性参考的物理假设

ResAD 使用恒速运动作为默认先验：

$$
\mathbf{p}_t=\mathbf{p}_0+\mathbf{v}_0t
$$

该假设在短时间、稳定巡航中较合理，但在以下场景中偏差较大：

- 急刹车；
- 急加速；
- 急转弯；
- 坡道和复杂车辆动力学；
- 强制变道和避障。

此时模型必须学习较大的 residual。

### 6.2 PRNorm 是数据集级统计变换

PRNorm 的上下界来自训练集全局极值，因此：

- 它依赖训练数据分布；
- 极端 outlier 会扩大映射范围；
- 分布外 residual 可能超出 $[-\gamma,\gamma]$；
- 归一化本身不保证每个时间点具有相同物理不确定性。

它主要解决数值尺度不平衡，不等价于严格的 uncertainty weighting。

### 6.3 IRP 的多模态假设

IRP 仅扰动初始速度，而不是直接扰动转向、目标点或车辆动力学参数。因此其多模态能力依赖：

- 不同速度参考能否覆盖不同驾驶意图；
- Diffusion Decoder 能否从 residual 学出方向性偏离；
- ranker 能否删除不安全候选。

对于明显不同的路线决策，仅靠速度扰动可能不够。

### 6.4 Ranker 的监督假设

ranker 使用规则教师和 GT 轨迹距离监督，隐含假设这些监督可以代表最终驾驶质量。但：

- GT 轨迹不一定是唯一安全解；
- 规则教师不一定覆盖全部交互情景；
- NAVSIM 指标与真实驾驶舒适性不完全等价；
- ranker 可能学习 benchmark-specific scoring。

### 6.5 论文没有证明的内容

论文结果不能严格证明：

- residual modeling 在所有自动驾驶模型上都优于直接轨迹预测；
- 恒速 inertial reference 适用于所有驾驶场景；
- 2 步 DDIM 在分布外场景中仍然可靠；
- PRNorm 一定比 z-score、learned normalization 或 uncertainty weighting 更优；
- IRP 生成的候选覆盖真实世界全部关键驾驶模式。

## 七、实验设计与结果分析

### 7.1 数据集和评测协议

论文使用 NAVSIM v1 和 NAVSIM v2：

- navtrain：1192 个场景；
- navtest：136 个场景；
- 数据基于真实 NuPlan 数据；
- 传感器和标注以 2 Hz 采样。

NAVSIM v1 的 PDMS：

$$
\operatorname{PDMS}
=
\operatorname{NC}
\times\operatorname{DAC}
\times
\frac{5\operatorname{TTC}+2\operatorname{C}+5\operatorname{EP}}{12}
$$

NAVSIM v2 的 EPDMS 在论文中写为：

$$
\operatorname{EPDMS}
=
\operatorname{NC}\times\operatorname{DAC}\times\operatorname{DDC}\times\operatorname{TL}
\times
\frac{5\operatorname{TTC}+2\operatorname{C}+5\operatorname{EP}+5\operatorname{LK}+5\operatorname{EC}}{22}
$$

其中：

- NC：No At-Fault Collisions；
- DAC：Drivable Area Compliance；
- DDC：Driving Direction Compliance；
- TL：Traffic Light Compliance；
- TTC：Time to Collision；
- C：Comfort；
- EP：Ego Progress；
- LK：Lane Keeping；
- EC：Extended Comfort。

### 7.2 NAVSIM v1 主结果

| 方法 | Backbone | NC | DAC | EP | TTC | C | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Transfuser | ResNet-34 | 97.7 | 92.8 | 79.2 | 92.8 | 100 | 84.0 |
| DiffusionDrive | ResNet-34 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| WoTE | ResNet-34 | 98.5 | 96.8 | 81.9 | 94.9 | 99.9 | 88.3 |
| ResAD | ResNet-34 | 98.0 | 97.5 | 83.3 | 94.1 | 100 | **88.8** |
| Hydra-MDP | V2-99 | 98.0 | 97.8 | 86.5 | 93.9 | 100 | 90.3 |
| GoalFlow | V2-99 | 98.4 | 98.3 | 85.0 | 94.6 | 100 | 90.3 |
| ResAD | V2-99 | 98.9 | 97.8 | 87.0 | 94.9 | 100 | **90.6** |

在相同 ResNet-34 backbone 下，ResAD 的 PDMS 88.8 高于 DiffusionDrive 的 88.1 和 WoTE 的 88.3。V2-99 结果说明更强 backbone 仍能进一步提升 ResAD。

### 7.3 NAVSIM v2 主结果

| 方法 | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| ResAD | 97.8 | 97.2 | 99.5 | 99.8 | 88.2 | 96.9 | 97.0 | 98.4 | 88.2 | **85.5** |

ResAD 的优势主要体现在：

- DAC：97.2，高于 DiffusionDrive 的 95.9；
- EP：88.2，高于 DiffusionDrive 的 87.5；
- LK：97.0，高于 DiffusionDrive 的 96.8；
- EPDMS：85.5，高于 DiffusionDrive 的 84.5。

但 ResAD 的 NC、TTC 并不是所有指标都最优，因此应理解为综合评分领先，而不是每一个安全子指标都领先。

### 7.4 组件消融

| 模型 | 增加组件 | NC | DAC | EP | TTC | C | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| $\mathcal{M}_0$ | Base Model | 97.8 | 94.2 | 78.1 | 93.4 | 100 | 84.9 |
| $\mathcal{M}_1$ | + Ranker | 98.3 | 94.3 | 77.8 | 94.6 | 100 | 85.1 |
| $\mathcal{M}_2$ | + TRM | 97.4 | 96.6 | 80.3 | 93.2 | 100 | 86.3 |
| $\mathcal{M}_3$ | + PRNorm | 97.6 | 96.7 | 81.4 | 93.3 | 100 | 87.2 |
| $\mathcal{M}_4$ | + IRP | 98.0 | 97.5 | 83.4 | 94.1 | 100 | **88.8** |

结论：

- Ranker：PDMS 84.9 → 85.1，收益较小但安全指标有所改善；
- TRM：PDMS 85.1 → 86.3，是主要性能增益来源；
- PRNorm：PDMS 86.3 → 87.2，改善优化平衡和 EP；
- IRP：PDMS 87.2 → 88.8，贡献最大，改善候选多样性和上下文适应性。

### 7.5 跨模型迁移

论文将 NRTM 应用于 Transfuser 和 TransfuserDP：

| 模型 | 配置 | PDMS |
|---|---|---:|
| Transfuser | 原始 | 84.0 |
| Transfuser | + TRM | 85.2 |
| Transfuser | + TRM + PRNorm | 85.6 |
| TransfuserDP | 原始 | 84.5 |
| TransfuserDP | + TRM | 85.5 |
| TransfuserDP | + TRM + PRNorm | 85.8 |

该实验支持论文的 plug-and-play 论断：残差建模和 PRNorm 不只适用于 ResAD 的完整扩散结构。

### 7.6 效率与多模态质量

论文在 RTX 4090 上报告：

| 方法 | PDMS | $K_{infer}$ | $\mathcal{P}_m$ | 总时间 | 参数量 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| TransfuserDP | 84.6 | 20 | 84.4 | 130.0 ms | 101M | 7 |
| DiffusionDrive | 88.1 | 20 | 60.3 | 7.6 ms | 60M | 45 |
| ResAD w/o Ranker | — | 20 | 86.1 | 7.7 ms | 62M | 45 |
| ResAD | 88.3 | 20 | 86.1 | 11.4 ms | 68M | 37 |

$\mathcal{P}_m$ 用于衡量候选轨迹的平均质量。ResAD 的 ranker 增加了约 6M 参数和约 3.7 ms 延迟，但带来了最终候选选择能力。

### 7.7 Test-time scaling

论文研究 $K_{infer}$ 对 PDMS 的影响：

- DiffusionDrive 在 $K_{infer}<K_{train}$ 时性能明显下降；
- 当 $K_{infer}>20$ 时，DiffusionDrive 的性能大致饱和；
- ResAD 的性能随 $K_{infer}$ 增加持续提升。

解释是 IRP 将额外推理计算限制在物理上更相关的参考区域，使增加候选数更可能产生有效轨迹，而不是大量无效样本。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **改变轨迹学习目标。** 从直接预测完整轨迹转向预测相对于物理先验的必要修正。
2. **缓解规划时域不平衡。** PRNorm 减少远期大数值误差对梯度的支配。
3. **生成场景相关多模态候选。** IRP 比固定轨迹词典更贴近当前 ego state。
4. **与现有模型兼容。** TRM/PRNorm 可以迁移到 MLP planner 和 diffusion planner。
5. **支持少步扩散规划。** 论文报告只用 2 步 DDIM 即取得较强性能。

### 8.2 论文承认或实验暴露的局限

- 惯性参考使用恒速模型，复杂动态下可能产生大残差；
- IRP 主要扰动速度，不能显式覆盖所有转向/路线意图；
- ranker 增加推理耗时和参数量；
- 论文的主要验证集中于 NAVSIM，真实道路泛化证据有限；
- 代码在论文页面仅声明将发布，当前未提供正式仓库链接；
- PRNorm 依赖训练集 min/max 统计，对分布外残差的处理需要额外验证。

### 8.3 分析者识别出的潜在问题

#### 问题一：恒速参考不一定总是强先验

对于急转弯、急刹车和复杂车辆动力学，惯性参考可能离真实可行轨迹很远，模型反而需要学习大幅 residual。此时“预测 residual 更简单”的优势可能减弱。

#### 问题二：全局 min-max 容易受极值影响

PRNorm 使用训练集全局极值。如果少数极端轨迹扩大区间，普通样本会被压缩到较小范围；如果测试场景超出范围，反归一化可能产生不可控误差。percentile clipping 或 robust scaling 值得比较。

#### 问题三：逐点归一化不等于安全加权

PRNorm 平衡数值尺度，但并未显式使用碰撞风险、距离或时间价值函数。它可能提高近场梯度的相对作用，却不保证得到的优化目标与真实安全优先级完全一致。

#### 问题四：IRP 可能扩大候选残差难度

速度扰动越大，参考轨迹越偏离真实轨迹，residual 也越大。IRP 需要在候选多样性和 residual 可学习性之间选择合适协方差 $\Sigma$。

#### 问题五：Ranker 可能产生评测指标过拟合

ranker 直接学习 NAVSIM 的规则指标，最终性能可能依赖 benchmark 的 metric 定义。迁移到真实控制系统时，需要验证其选择是否仍符合车辆动力学和乘坐舒适性。

#### 问题六：多模态质量需要区分“多样”与“有效”

增加 $K_{infer}$ 可能提高找到高分轨迹的概率，但如果候选之间只是速度微扰，而没有覆盖真正不同的路线意图，那么多模态仍然有限。

## 九、通俗讲解

### 9.1 传统方法在学什么

传统规划模型像是在背答案：

```text
图片 + LiDAR → 未来 8 个位置点
```

它需要同时学会：

- 车辆自然前进；
- 什么时候刹车；
- 什么时候转弯；
- 什么时候让行；
- 什么时候绕开障碍物。

任务很重。

### 9.2 ResAD 怎么拆任务

ResAD 先根据当前速度画一条“如果什么都不做，车辆会怎么走”的参考线：

```text
当前状态 → 恒速惯性参考
```

然后只学习真实驾驶和参考线之间的差异：

```text
真实轨迹 - 惯性参考 = 需要学习的修正
```

例如：

- 正常巡航：修正很小；
- 红灯停车：修正主要体现在纵向减速；
- 左转：修正主要体现在横向方向；
- 避障：修正体现为绕行。

### 9.3 PRNorm 在做什么

远期点坐标和误差通常更大，容易压过近处的安全细节。PRNorm 把不同维度的 residual 映射到相近范围：

```text
原始 residual：远期误差数值大
        ↓
min-max 归一化
        ↓
各点的训练贡献更平衡
```

### 9.4 IRP 在做什么

只用一条惯性参考只能得到一种初始意图。IRP 对当前速度加一点随机扰动：

```text
原始速度
  ├─ 稍微快一点 → 参考轨迹 1
  ├─ 稍微慢一点 → 参考轨迹 2
  ├─ 稍微偏左   → 参考轨迹 3
  └─ 稍微偏右   → 参考轨迹 4
```

扩散模型分别修正这些参考，得到多条候选轨迹。

### 9.5 Ranker 在做什么

扩散模型负责“提出方案”，ranker 负责“选方案”：

```text
候选 1：安全但进度低
候选 2：进度高但可能压线
候选 3：安全、合规、进度适中
              ↓
         ranker 选择候选 3
```

### 9.6 一句话理解

> ResAD 不让模型从零画完整轨迹，而是先画一条物理上自然的惯性参考，再让模型学习交通规则和障碍物导致的必要偏离，并通过逐点归一化和参考扰动获得稳定的多模态规划。

## 十、综合评价与后续研究方向

### 10.1 综合评价

ResAD 的核心链条是：

$$
\text{当前 ego state}
\rightarrow
\text{Inertial Reference}
\rightarrow
\text{IRP 多参考}
\rightarrow
\text{Residual Modeling}
\rightarrow
\text{PRNorm}
\rightarrow
\text{Diffusion Denoising}
\rightarrow
\text{Trajectory Ranker}
$$

论文的主要贡献不是提出一个更复杂的 diffusion decoder，而是重新定义了 diffusion model 的预测对象：

$$
\text{预测完整轨迹}
\quad\Longrightarrow\quad
\text{预测相对惯性参考的修正}
$$

这一改变带来三个直接效果：

1. 将车辆自然运动交给物理先验；
2. 将模型容量集中到红灯、障碍物、转弯、让行等上下文决策；
3. 通过 PRNorm 缓解远期 waypoint 数值过大造成的优化失衡。

实验结果支持以下结论：

- TRM 是主要增益来源；
- PRNorm 改善优化稳定性和性能；
- IRP 对多模态质量和最终 PDMS 提升最大；
- ResAD 可以在 2 步 DDIM 下获得较强 NAVSIM 结果；
- NRTM 可以迁移到 Transfuser 和 TransfuserDP。

但不应将 ResAD 理解为完全解决了端到端规划中的不确定性：

- 惯性参考是恒速近似；
- 速度扰动不一定覆盖所有高层路线意图；
- ranker 依赖规则教师和 benchmark metric；
- PRNorm 的全局极值统计存在分布外风险；
- 主要证据来自 NAVSIM，真实闭环泛化仍需验证。

更准确的结论是：

> ResAD 通过“物理惯性参考 + 归一化残差 + 参考扰动 + 候选排序”降低了扩散式端到端轨迹规划的学习难度，在保持多模态生成能力的同时减少了对固定轨迹词典的依赖，是一种具有较强可解释性和迁移潜力的轨迹表示重构方法。

### 10.2 后续研究方向

#### 方向一：更强车辆动力学先验

将恒速模型扩展为考虑加速度、转向角和车辆动力学的可微 reference：

$$
\mathbf{p}_{t+\Delta t}
=
\operatorname{VehicleDynamics}
(\mathbf{s}_t,\mathbf{u}_t,\Delta t)
$$

#### 方向二：可学习或鲁棒的归一化

比较：

- percentile min-max；
- robust scaling；
- per-timestep normalization；
- uncertainty-aware normalization；
- learned residual scaling。

#### 方向三：多维 IRP

除了速度扰动，还可扰动：

- acceleration；
- yaw rate；
- target goal；
- route command；
- lane-change intent。

从而覆盖更丰富的多模态行为。

#### 方向四：安全约束 ranker

在 ranker 中加入显式碰撞、道路边界、车辆动力学和舒适性约束，减少对 benchmark-specific metric 的依赖。

#### 方向五：长时域与滚动规划

结合 receding-horizon planning，将短时 residual 预测和长时目标规划结合，避免恒速参考在长时间域内快速失真。

#### 方向六：候选覆盖率评估

增加：

- oracle trajectory recall；
- candidate diversity；
- collision-free candidate ratio；
- 高层意图覆盖率；
- 不同 $K_{infer}$ 下的有效候选增长率。

#### 方向七：真实闭环验证

在 CARLA、Bench2Drive 和真实车辆上评估：

- 传感器噪声；
- 交通参与者交互；
- 车辆动力学误差；
- 长尾交通规则；
- 分布外道路和天气。

## 一句话结论

> ResAD 用恒速惯性轨迹承担“车辆自然怎么走”，用归一化 residual 学习“为什么需要偏离”，再通过惯性参考扰动和 ranker 实现高效、上下文相关的多模态端到端自动驾驶规划。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2510.08562](https://arxiv.org/abs/2510.08562)
- 论文 HTML：[https://arxiv.org/html/2510.08562v2](https://arxiv.org/html/2510.08562v2)
- 论文 PDF：[https://arxiv.org/pdf/2510.08562](https://arxiv.org/pdf/2510.08562)
- 项目主页：[https://duckyee728.github.io/ResAD](https://duckyee728.github.io/ResAD)
