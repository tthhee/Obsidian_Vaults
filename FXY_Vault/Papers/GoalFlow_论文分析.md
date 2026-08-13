# GoalFlow：论文分析

> 标题：*GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving*  
> 来源：[arXiv:2503.05689](https://arxiv.org/abs/2503.05689)  
> 版本：v6，2025 年 10 月 1 日修订；首次提交于 2025 年 3 月 7 日  
> 作者：Zebin Xing、Xingyu Zhang、Yang Hu、Bo Jiang、Tong He、Qian Zhang、Xiaoxiao Long、Wei Yin  
> 论文原文未说明正式发表的会议或期刊。  
> 官方代码：[github.com/YvanYin/GoalFlow](https://github.com/YvanYin/GoalFlow)

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | 端到端自动驾驶、多模态轨迹生成、Flow Matching |
| 核心问题 | 无约束生成容易产生发散或越界轨迹，固定引导点与真实目标不一致时质量下降 |
| 核心方法 | Goal-driven Flow Matching |
| 主要模块 | Transfuser 感知、Goal Point Construction、Rectified Flow、Trajectory Scorer |
| 目标点词典 | 从训练轨迹终点聚类，通常 4096/8192 个候选 |
| 轨迹生成 | 以场景 BEV 和目标点为条件的 Rectified Flow |
| 主要优势 | 目标点约束生成区域，单步推理仍有较好性能 |
| NAVSIM v1 | PDMS 90.3；使用 GT endpoint 的上限实验为 92.1 |
| 单步推理 | PDMS 88.9；5 步为 90.3 |
| 官方代码 | 已公开，主要位于 `navsim/agents/goalflow/` |

## 二、极简全文核心总结

GoalFlow 将多模态轨迹规划拆成“目标点选择”和“目标点条件下的轨迹生成”两步。模型从轨迹终点聚类得到目标点词典，结合距离与可行驶区域评分选择场景相关目标点，再用 Rectified Flow 从高斯噪声生成多条目标约束轨迹，最后通过轨迹评分器选出结果。目标点降低了生成发散，Flow Matching 降低了采样步数，在 NAVSIM 上取得 PDMS 90.3。

## 三、研究背景与研究意义

### 3.1 多模态驾驶轨迹

同一驾驶场景往往存在多条合理轨迹：

- 直行、左转、右转；
- 减速让行或继续通行；
- 不同程度的横向偏移；
- 不同速度通过同一空间区域。

因此，模型需要学习条件分布：

$$
 p(\tau\mid I,L,E)
$$

其中 $I$ 是图像，$L$ 是 LiDAR，$E$ 是 ego status，$\tau$ 是未来轨迹。

### 3.2 无约束生成的问题

直接使用 diffusion/flow 生成完整轨迹时，模型可能产生：

- 远离道路的轨迹；
- 穿出 drivable area 的轨迹；
- 多个模式之间边界模糊；
- 目标点与场景语义不匹配；
- 需要额外后处理或逐条模拟筛选。

GoalFlow 的核心拆解是：

$$
 p(\tau\mid scene)
=
 p(g\mid scene)\,p(\tau\mid g,scene)
$$

先选目标点 $g$，再生成到达该目标点的轨迹。

## 四、核心方法、模型、公式与流程

### 4.1 整体框架图

![GoalFlow framework](https://arxiv.org/html/2503.05689v6/main_fig.png)

> **图 2：GoalFlow 总体架构。** 感知模块将图像和 LiDAR 融合为 BEV；目标点构造模块从目标点词典中选择目标点；轨迹规划模块使用 Rectified Flow 生成多条轨迹；最终由 trajectory scorer 选择输出。图片来源：[论文 Figure 2](https://arxiv.org/html/2503.05689v6/main_fig.png)。

```text
Camera + LiDAR + Ego status
            ↓
Transfuser-style perception
            ↓
         BEV feature
            ├──────────────┐
            ↓              ↓
Goal Point Vocabulary   Trajectory Flow Model
            ↓              ↑
Distance/DAC scorer ─ Goal condition
            ↓
      selected goal point
            ↓
128/256 candidate trajectories
            ↓
Trajectory scorer + shadow check
            ↓
      final trajectory
```

### 4.2 感知模块

论文沿用 Transfuser 风格的多模态融合：

- forward/left/right camera 视图拼接；
- LiDAR 形成点云张量；
- 图像和 LiDAR 使用独立 backbone；
- 多层 Transformer 融合；
- 输出 BEV feature $F_{bev}$。

形式化表示：

$$
F_{bev}=\Phi(I,L,E)
$$

论文还通过 HD map 和 3D bounding box 辅助任务增强 BEV 表示。

### 4.3 Goal Point Vocabulary

从训练集轨迹终点提取：

$$
 p_i=(x_i,y_i,\theta_i)
$$

使用聚类获得目标点词典：

$$
\mathbb{V}=\{g_i\}_{i=1}^{N}
$$

每个目标点不仅包含位置 $(x,y)$，还包含终点朝向 $\theta$。论文通常设置 $N=4096$ 或 $8192$，以获得更细粒度的目标位置覆盖。

目标点与固定完整轨迹词典不同：目标点只提供终点约束，具体中间轨迹仍由生成模型根据场景决定。

### 4.4 Goal Point Scorer

目标点需要满足：

1. 接近真实轨迹终点；
2. 位于可行驶区域内。

#### 4.4.1 Distance Score

对目标点 $g_i$ 和 GT goal $g_{gt}$ 计算距离软标签：

$$
\delta_i^{dis}
=
\frac{\exp(-\|g_i-g_{gt}\|_2)}
{\sum_j\exp(-\|g_j-g_{gt}\|_2)}
$$

论文 HTML 中该式写作平方距离/非平方距离存在排版版本差异；正文公式显示为 $\|\cdot\|_2$，实现理解应以仓库对应配置和 checkpoint 为准。其核心是：距离越近，目标概率越高。

#### 4.4.2 Drivable Area Compliance

对每个候选目标点构造一个 shadow vehicle，根据 $(x_i,y_i,\theta_i)$ 放置车辆框。设四个角点为 $p_j$，可行驶区域为多边形 $\mathbb{D}$：

$$
\delta_i^{dac}
=
\begin{cases}
1, & \forall j,\ p_j\in\mathbb{D}\\
0, & \text{otherwise}
\end{cases}
$$

这比只检查目标点中心更严格，因为车辆完整 footprint 都必须落在可行驶区域内。

#### 4.4.3 目标点最终分数

论文给出：

$$
\hat\delta_i^{final}
=
w_1\log\hat\delta_i^{dis}
+w_2\log\hat\delta_i^{dac}
$$

实现上通常需要对概率加极小值，避免 $\log 0$。候选得分最高的目标点用于引导轨迹生成。

### 4.5 Rectified Flow / Flow Matching

设噪声分布为 $\pi_0$，目标轨迹分布为 $\pi_1$：

$$
 x_0\sim\pi_0,
\qquad
 x_1\sim\pi_1
$$

采用直线插值：

$$
 x_t=(1-t)x_0+tx_1,
\qquad t\in[0,1]
$$

目标速度为：

$$
 v_t=x_1-x_0
$$

网络 $v_\theta(x_t,t,c)$ 预测该速度场，训练目标为：

$$
\mathcal{L}_{FM}
=
\mathbb{E}
\left[
\left\|
 v_\theta(x_t,t,c)-(x_1-x_0)
\right\|_2^2
\right]
$$

GoalFlow 中 $x_1$ 是归一化轨迹，条件 $c$ 包括：

- BEV scene feature；
- ego status；
- selected goal point；
- time embedding。

### 4.6 轨迹生成

论文将 GT trajectory 归一化：

$$
\tau_{norm}=\mathcal{H}(\tau_{gt})
$$

从高斯噪声开始，模型沿学习到的速度场积分：

$$
\hat\tau_{norm}
\approx
x_0+
\sum_i
\Delta t_i\,
\hat v_\theta(x_{t_i},t_i,c)
$$

最后反归一化：

$$
\hat\tau=\mathcal{H}^{-1}(\hat\tau_{norm})
$$

论文采用 Rectified Flow，推理步数可显著低于标准 DDPM。实验中：

- 20 步：PDMS 89.9；
- 10 步：PDMS 90.1；
- 5 步：PDMS 90.3；
- 1 步：PDMS 88.9。

### 4.7 Trajectory Scorer

生成多条轨迹后，使用目标点和 ego progress 进行筛选。论文形式为：

$$
 f(\hat\tau_i)
=
-\lambda_1\Phi(f_{dis}(\hat\tau_i))
+\lambda_2\Phi(f_{pg}(\hat\tau_i))
$$

其中：

- $f_{dis}$：轨迹终点与目标点的距离；
- $f_{pg}$：轨迹带来的 ego progress；
- $\Phi$：min-max normalization；
- $\lambda_1,\lambda_2$：权重。

论文还引入 shadow trajectory：生成时屏蔽目标点条件，得到一条无目标点引导的参考轨迹。如果目标点引导的主轨迹与 shadow trajectory 偏差过大，则认为目标点可能不可靠，并使用 shadow trajectory 作为输出。

## 五、核心代码讲解（官方仓库）

> 官方仓库：[YvanYin/GoalFlow](https://github.com/YvanYin/GoalFlow)。以下内容根据公开 `main` 分支源码核对，区分论文描述与源码实现。

### 5.1 代码目录与调用关系

| 文件 | 作用 |
|---|---|
| [`goalflow_model_traj.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/goalflow_model_traj.py) | 主模型、BEV 编码、flow training/sampling、目标点条件 |
| [`goalflow_agent_traj.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/goalflow_agent_traj.py) | NAVSIM agent 封装与推理接口 |
| [`goalflow_model_navi.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/goalflow_model_navi.py) | 导航/目标点 scorer 相关模型 |
| [`goalflow_loss.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/goalflow_loss.py) | 训练 loss |
| [`goalflow_config.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/goalflow_config.py) | 数据、模型、采样配置 |
| [`diffusion_es.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/diffusion_es.py) | sinusoidal timestep、attention、采样辅助模块 |
| [`utils.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/utils.py) | 位置编码与轨迹工具 |
| [`resnet_backbone.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/resnet_backbone.py) | ResNet backbone |
| [`v99_backbone.py`](https://github.com/YvanYin/GoalFlow/blob/main/navsim/agents/goalflow/v99_backbone.py) | V2-99 backbone |

源码调用链：

```text
GoalFlowAgent
    ↓
GoalFlowTrajModel.forward()
    ├─ V299Backbone(camera_feature, lidar_feature)
    ├─ BEV downscale + Transformer decoder
    ├─ goal point / navigation condition
    ├─ normalize_xy_rotation(gt_trajs)
    ├─ get_train_tuple() + denoise()       # training
    └─ flow integration + denormalize       # inference
          ↓
      candidate trajectories
          ↓
      nearest/progress scorer + shadow fusion
          ↓
      final trajectory
```

### 5.2 `GoalFlowTrajModel` 主模型

源码初始化核心模块：

```python
self._backbone = V299Backbone(config)
self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
self._tf_decoder = nn.TransformerDecoder(...)
```

模型还建立：

- BEV semantic head；
- key/value embedding；
- query embedding；
- agent head；
- trajectory encoder；
- timestep `SinusoidalPosEmb`；
- `RotaryPositionEncoding`；
- 多层 `ParallelAttentionLayer`；
- `decoder_mlp` 输出 30 维轨迹表示。

源码中 30 维对应 10 个 waypoint 的 $(x,y,\theta)$ 展平表示；具体 waypoint 数量受配置和预处理影响，不能只根据论文中的 8 个评测时间步推断代码内部所有张量形状。

### 5.3 BEV 特征流

源码 forward 首先读取：

```python
camera_feature = features["camera_feature"]
lidar_feature = features["lidar_feature"]
status_feature = features["status_feature"]
```

然后：

```python
bev_feature_upscale, bev_feature, _ = self._backbone(
    camera_feature,
    lidar_feature,
)
bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
bev_feature = bev_feature.permute(0, 2, 1)
```

这对应论文的：

$$
(I,L,E)\rightarrow F_{bev}
$$

BEV token 与 ego status token 拼接，并加入可学习位置 embedding：

```python
keyval = torch.concatenate(
    [bev_feature, status_encoding[:, None]], dim=1
)
keyval += self._keyval_embedding.weight[None, ...]
query = self._query_embedding.weight[None, ...].repeat(
    batch_size, 1, 1
)
query_out = self._tf_decoder(query, keyval)
```

### 5.4 目标点条件的代码实现

源码通过配置控制三种情况：

- `has_navi`：使用真实导航/目标条件；
- `has_student_navi`：从预计算的 distance/DAC score 文件中选择目标点；
- 否则只使用场景特征，不显式加入目标点。

在 `has_student_navi` 分支中，源码读取：

```python
im_score = np.load(...)
dac_score = np.load(...)
cluster_points = np.load(config.voc_path)
```

并构造最终目标点分数：

```python
final_scores = (
    0.1 * torch.log(torch.softmax(im_scores, dim=-1) + 1e-7)
    + theta * torch.log(torch.sigmoid(dac_scores) + 1e-7)
)
```

如果启用 progress 相关权重，还会加入目标点距离的归一化项。然后：

```python
topk_indices = torch.topk(final_scores, config.topk).indices
navi = torch.gather(
    cluster_points_tensor,
    dim=1,
    index=topk_indices_expanded,
).mean(1)[..., :2].unsqueeze(1)
```

这说明论文的“目标点选择”在代码中可能表现为：

- 选择单个 argmax 目标点；或
- 选择 Top-K 后取平均目标点；

具体取决于配置分支，不能把所有实验配置都简化成单一 argmax。

### 5.5 轨迹归一化与 flow training

源码调用：

```python
normal_trajs = self.normalize_xy_rotation(
    gt_trajs_,
    N=gt_trajs_.shape[-2],
    times=10,
)
```

训练阶段创建噪声：

```python
noise = torch.randn(
    size=(batch_size, 12, 30),
    device=normal_trajs.device,
    dtype=dtype,
) * config.train_scale
```

随后通过：

```python
noisy_traj_points, t, target = get_train_tuple(
    z0=noise,
    z1=normal_trajs,
)
```

获得 flow interpolation 的训练样本，并调用：

```python
pred = self.denoise(
    noisy_traj_points,
    timesteps,
    global_feature,
)
```

论文公式的直线插值在代码中对应 `get_train_tuple`；网络输出的 30 维向量对应轨迹 token 的速度/位移预测。

### 5.6 Flow inference：一步或多步积分

推理阶段先构造：

```python
trajs = noise
```

然后在采样时间序列中反复调用 `denoise`，更新轨迹：

```python
net_output = self.denoise(
    trajs, t_curr, global_feature
)
trajs = trajs.detach().clone() \\
    + net_output * (step / config.infer_steps)
```

最后反归一化：

```python
diffusion_output = self.denormalize_xy_rotation(
    trajs,
    N=gt_trajs.shape[-2],
    times=10,
)
pred_trajs = diffusion_output.reshape(
    batch_size,
    config.anchor_size,
    -1,
    3,
)
```

虽然代码变量名仍使用 `diffusion_output`，但论文方法实际使用的是 Rectified Flow/Flow Matching 风格的速度场积分。变量命名不应被误读为 DDPM 采样。

### 5.7 Classifier-free guidance 与 shadow trajectory

源码根据配置进行条件和无条件 denoise：

```python
net_output_nonavi = self.denoise(
    trajs, t_curr, global_feature,
    navi_dropout=True,
)
net_output_cond = self.denoise(
    trajs, t_curr, global_feature,
)
net_output = (
    (1 - config.cond_weight) * net_output_nonavi
    + config.cond_weight * net_output_cond
)
```

在 `fusion` 分支中，代码分别生成：

- 带 navigation condition 的主轨迹；
- `navi_dropout=True` 的 shadow/non-navigation 轨迹。

然后比较主轨迹与目标点的距离；如果目标条件造成过大偏差，则使用 shadow 结果，否则进行加权融合。这与论文描述的“目标点不可靠时使用 shadow trajectory”一致。

### 5.8 代码级伪代码

```python
def forward(features, targets):
    bev = encode_camera_lidar(features)
    scene = encode_bev_and_ego(bev, features["status_feature"])

    if use_goal_condition:
        goal = select_goal_from_vocabulary(
            distance_scores,
            dac_scores,
            progress_weight,
        )
        condition = encode_scene_and_goal(scene, goal)
    else:
        condition = encode_scene(scene)

    if training:
        clean = normalize(targets["trajectory"])
        noise = randn_like(clean) * train_scale
        noisy, t, target = get_train_tuple(noise, clean)
        prediction = denoise(noisy, t, condition)
        return flow_loss(prediction, target)

    candidates = []
    for _ in range(anchor_size):
        traj = randn_trajectory()
        for t in sampling_schedule:
            velocity = denoise(traj, t, condition)
            traj = traj + velocity * step_size
        candidates.append(denormalize(traj))

    candidates = shadow_fusion(candidates, scene, goal)
    return select_by_goal_distance_and_progress(candidates)
```

### 5.9 论文方法与源码对应关系

| 论文概念 | 源码对应 |
|---|---|
| Perception Module | `V299Backbone`、`_bev_downscale`、`_tf_decoder` |
| BEV feature | `bev_feature` |
| Goal Point Vocabulary | `cluster_points=np.load(config.voc_path)` |
| Distance/DAC score | `im_score`、`dac_score` 及其融合逻辑 |
| Goal condition | `navi_feature`、`encode_navi_features` |
| Flow Matching | `get_train_tuple`、`denoise`、trajectory Euler-like update |
| timestep embedding | `SinusoidalPosEmb`、`RotaryPositionEncoding` |
| Trajectory normalization | `normalize_xy_rotation` / `denormalize_xy_rotation` |
| Shadow trajectory | `navi_dropout=True` 分支 |
| 多模态候选 | `anchor_size` 维度的噪声采样 |
| 最终选择 | nearest/progress scoring 与 `fusion` 配置 |

## 六、核心创新点与传统方法对比

### 6.1 目标点驱动的生成

目标点把完整轨迹生成拆成：

$$
\text{场景}
\rightarrow
\text{目标点}
\rightarrow
\text{目标约束轨迹}
$$

目标点提供短期终点和方向约束，减少轨迹发散。

### 6.2 Flow Matching 替代多步 diffusion

Rectified Flow 学习从噪声到轨迹分布的速度场，路径更直接，推理步数更少。

### 6.3 Distance + DAC 双评分

只用距离可能选择越界目标；只用 DAC 不能区分场景意图。两者结合提升目标点质量。

### 6.4 Shadow trajectory 可靠性检查

目标点预测错误时，shadow trajectory 提供不依赖目标点的备选路径，降低错误引导的风险。

## 七、理论分析与关键假设

### 7.1 目标点条件分解

GoalFlow 假设目标点是轨迹分布的有效低维中间变量：

$$
 p(\tau\mid scene)
\approx
 p(g\mid scene)p(\tau\mid g,scene)
$$

该分解可以降低生成难度，但目标点并不能完整描述中间路径、速度和交互行为。

### 7.2 Rectified Flow 的效率边界

单步推理效果依赖速度场轨迹是否足够直。单步结果不是理论上精确的目标分布样本；它是数值积分步数减少后的近似。

### 7.3 目标点选择误差会传播

若选中的 $g$ 错误，则生成模型可能被迫生成与场景不匹配的轨迹。shadow trajectory 只能缓解部分错误，不能保证纠正所有导航意图错误。

### 7.4 论文没有证明的内容

- 目标点一定是所有驾驶行为的充分统计量；
- 单步 flow 对任意分布外场景都可靠；
- 目标点词典完全避免了固定词典的离散化问题；
- DAC 目标点检查等价于完整轨迹安全检查；
- 与动态轨迹生成方法在完全相同计算预算下绝对公平。

## 八、实验设计与结果分析

### 8.1 数据集与评测

论文在 NAVSIM/OpenScene 环境上评估：

- 约 120 小时自动驾驶数据；
- navtrain：1192 场景；
- navtest：136 场景；
- 2 Hz 输入/轨迹；
- 4 秒规划时域；
- 生成轨迹通过 LQR 插值到 10 Hz，再进行闭环指标评估。

### 8.2 NAVSIM 主结果

| 方法 | Backbone | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Transfuser | ResNet-34 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DiffusionDrive | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | ResNet-34 | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| GoalFlow | ResNet-34 | 98.4 | **98.3** | **94.6** | 100 | **85.0** | **90.3** |
| GoalFlow + GT endpoint | ResNet-34 | **99.8** | 97.9 | 98.6 | 100 | 85.4 | 92.1 |
| Human | — | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |

GT endpoint 结果不是可部署结果，而是目标点引导能力的上限参考；不能当作普通模型结果比较。

### 8.3 消融实验

| 模型 | 组件 | NC | DAC | TTC | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|
| $\mathcal{M}_0$ | Rectified Flow，无 goal guidance | 97.9 | 94.2 | 94.2 | 79.9 | 85.6 |
| $\mathcal{M}_1$ | + Distance Score Map | 98.5 | 96.4 | 94.9 | 83.0 | 88.5 |
| $\mathcal{M}_2$ | + DAC Score Map | 98.6 | 97.5 | 94.7 | 83.8 | 89.4 |
| $\mathcal{M}_3$ | + Trajectory Scorer | 98.4 | 98.3 | 94.6 | 85.0 | **90.3** |

主要结论：

- distance score map 带来最大单项增益；
- DAC score map 进一步改善可行驶区域合规；
- trajectory scorer 进一步提升总体 PDMS 和 ego progress。

### 8.4 推理步数

| 推理步数 | 时间 | PDMS |
|---:|---:|---:|
| 20 | 177.8 ms | 89.9 |
| 10 | 92.4 ms | 90.1 |
| 5 | 49.0 ms | **90.3** |
| 1 | 10.4 ms | 88.9 |

5 步结果优于 20 步，说明数值误差、随机候选和 scorer 共同影响结果；不能简单认为步数越多必然越好。

### 8.5 初始噪声标准差

| $\sigma$ | PDMS |
|---:|---:|
| 0.05 | 90.1 |
| 0.10 | **90.3** |
| 0.20 | 49.0 |
| 0.30 | 18.8 |

噪声太小会使 flow matching 退化为接近回归，候选多样性不足；噪声太大会造成轨迹不稳定。

### 8.6 证据边界

实验支持：

- goal guidance 提升轨迹质量；
- DAC-aware goal selection 改善道路合规；
- trajectory scorer 有效；
- flow matching 可用少量步数推理。

实验不能完全支持：

- GoalFlow 在所有真实道路上都领先；
- 单步生成在安全性上等同于多步生成；
- 目标点构造完全消除轨迹发散；
- 目标点方法优于所有动态生成方法。

## 九、学术价值、局限性与潜在漏洞

### 9.1 学术价值

1. **合理的任务分解：** 先做短期目标选择，再做轨迹生成。
2. **显式道路约束：** DAC score 从目标点阶段介入，而非只在最终轨迹后处理。
3. **高效生成：** Rectified Flow 支持低步数推理。
4. **多模态与场景相关兼顾：** 目标点词典提供模式，Flow Model 负责连续轨迹细化。
5. **可解释性较好：** 可以观察目标点、目标点得分和最终轨迹的关系。

### 9.2 论文和源码暴露的限制

- 目标点词典仍然来自训练数据聚类，存在离散化和覆盖偏差；
- DAC 检查目标点 footprint，不等价于整条轨迹都安全；
- 目标点错误会影响后续生成；
- 1 步推理性能低于 5 步；
- 目标点、轨迹 scorer 和后处理配置对结果影响较大；
- 代码中存在多个配置分支，论文主流程不能代表所有运行模式；
- 官方仓库的部分变量名仍沿用 diffusion 实现，阅读源码时需结合实际 flow update。

### 9.3 分析者识别出的潜在问题

#### 问题一：目标点可能过度约束轨迹

目标点能减少发散，但也可能把轨迹限制在错误的终点附近，导致模型无法表达中途停车、绕行或临时改变路线。

#### 问题二：点可行不等于轨迹可行

目标点的 shadow vehicle 在 drivable area 内，并不保证：

- 中间 waypoint 不越界；
- 不与动态目标碰撞；
- 轨迹曲率和加速度满足车辆动力学；
- 通过路口时序正确。

#### 问题三：轨迹选择使用简化分数

基于目标距离和 progress 的 scorer 比完整闭环仿真快，但可能遗漏碰撞、舒适性和交通规则等复杂因素。论文中报告的高 PDMS 仍依赖 NAVSIM 评测器，而不是完全由这个简化 scorer 单独保证。

#### 问题四：Flow Matching 的单步效果依赖训练分布

单步积分假设速度场较直。复杂多峰或分布外场景可能需要更多步数，不能把单步结果理解为精确采样。

#### 问题五：GT endpoint 实验存在不可部署信息

GT endpoint 的 PDMS 92.1 说明目标点很重要，但它使用了测试样本真实终点，不能作为实际系统性能，也不能和普通预测模型等价比较。

## 十、通俗讲解

### 10.1 传统扩散轨迹生成

传统方法像是：

```text
从一团噪声开始
    ↓
逐步生成未来 4 秒的完整轨迹
```

因为没有明确终点，可能出现很多无意义轨迹。

### 10.2 GoalFlow 的两步法

GoalFlow 先回答：

```text
车辆几秒后大概要到哪里？
```

再回答：

```text
如何沿道路安全地到达那里？
```

流程是：

```text
目标点词典
    ↓
结合场景选择合理目标点
    ↓
Flow Matching 生成到达目标点的多条轨迹
    ↓
选择最安全、最有进度的一条
```

### 10.3 为什么目标点有用

如果只生成完整轨迹，搜索空间很大；目标点相当于给模型一个方向约束：

```text
没有目标点：从任意噪声探索完整轨迹
有目标点：围绕指定终点生成轨迹
```

这能减少轨迹发散并提升道路合规。

### 10.4 为什么 Flow Matching 快

扩散模型通常需要多次去噪；Rectified Flow 尝试学习一条更直接的路径：

```text
噪声分布 ─────────→ 轨迹分布
             直接流动
```

因此少量积分步也能得到可用结果。

### 10.5 Shadow trajectory

目标点预测有时会错。GoalFlow 同时生成一条不依赖目标点的 shadow trajectory：

```text
主轨迹与 shadow 接近 → 目标点可信，使用主轨迹
主轨迹与 shadow 差异大 → 目标点可疑，使用 shadow 或融合结果
```

### 10.6 一句话理解

> GoalFlow 先选择“车辆应该到哪里”，再用 Flow Matching 生成“如何到那里”的多条轨迹，最后通过评分器选择最优轨迹。

## 十一、综合评价与后续研究方向

### 11.1 综合评价

GoalFlow 的完整因果链为：

$$
(I,L,E)
\rightarrow
F_{bev}
\rightarrow
\text{Goal Point Scoring}
\rightarrow
g
\rightarrow
\text{Goal-conditioned Flow Matching}
\rightarrow
\{\hat\tau_i\}
\rightarrow
\text{Trajectory Scoring}
\rightarrow
\hat\tau
$$

论文的关键贡献不是单独把 diffusion 换成 flow，而是把目标点作为中间结构引入轨迹生成：

- Distance score 让目标点贴近场景/GT 终点分布；
- DAC score 过滤明显越界的目标点；
- Flow Matching 在目标约束下生成连续多模态轨迹；
- Trajectory scorer 负责从候选中选择最终行为；
- Shadow trajectory 缓解目标点误导。

实验结果表明，GoalFlow 在 NAVSIM 上取得 PDMS 90.3，并且 5 步即可达到最佳报告结果，1 步仍有 88.9。其方法在效率、道路合规和候选轨迹质量之间取得了较好折中。

但应注意：

- 目标点 vocabulary 仍存在离散化；
- 目标点安全不能替代整条轨迹安全；
- 目标点错误会传播到 flow generation；
- 单步 flow 是近似积分而非精确生成；
- GT endpoint 结果只表示 oracle 上限。

更准确的评价是：

> GoalFlow 将多模态轨迹规划分解为目标点选择和目标条件轨迹生成，利用显式空间目标降低生成发散，再借助 Rectified Flow 实现低步数推理，是一种结构清晰、效率较高且具有可解释性的端到端自动驾驶规划方案。

### 11.2 后续研究方向

1. **连续目标点生成：** 用连续 density/flow 替代固定目标点词典，减少离散化误差。
2. **目标点与整轨迹联合安全约束：** 在 flow 过程中加入碰撞、道路边界、曲率和加速度约束。
3. **动态目标点预测：** 将交通灯、前车、行人和导航意图纳入目标点 scorer。
4. **不确定性目标分布：** 保留多个目标点分布，而不是只使用单个 argmax/均值目标点。
5. **自适应推理步数：** 根据场景复杂度和 flow 估计误差动态选择 1、3、5 或更多步。
6. **更强 trajectory ranker：** 融合闭环仿真、车辆动力学和风险敏感指标。
7. **长时域滚动规划：** 将短时 goal point 与 receding-horizon 规划结合。
8. **真实车辆验证：** 评估传感器噪声、目标遮挡、道路变化和分布外场景。

## 一句话结论

> GoalFlow 用“目标点选择 + 目标条件 Flow Matching + 轨迹评分”约束多模态轨迹生成，在降低轨迹发散和推理成本的同时，提升了端到端自动驾驶的道路合规与规划性能。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2503.05689](https://arxiv.org/abs/2503.05689)
- 论文 HTML：[https://arxiv.org/html/2503.05689v6](https://arxiv.org/html/2503.05689v6)
- 论文 PDF：[https://arxiv.org/pdf/2503.05689](https://arxiv.org/pdf/2503.05689)
- 官方代码：[https://github.com/YvanYin/GoalFlow](https://github.com/YvanYin/GoalFlow)
