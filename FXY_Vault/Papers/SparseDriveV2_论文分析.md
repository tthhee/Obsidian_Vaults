# SparseDriveV2：论文分析

> 标题：*SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving*  
> 来源：[arXiv:2603.29163](https://arxiv.org/abs/2603.29163)  
> 版本：v1，2026 年 3 月 31 日提交  
> 作者：Wenchao Sun、Xuewu Lin、Keyu Chen、Zixiang Pei、Xiang Li、Yining Shi、Sifa Zheng  
> 代码：[github.com/swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2)  
> 论文原文未说明正式发表的会议或期刊。

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | 端到端自动驾驶、多模态轨迹规划、轨迹评分、候选轨迹词典 |
| 核心问题 | 静态轨迹词典覆盖粗糙，动态轨迹生成复杂且成本高 |
| 核心方法 | Factorized Vocabulary + Scalable Scoring |
| 轨迹分解 | Geometric Path + Velocity Profile |
| 词典规模 | 1024 paths × 256 velocity profiles = 262,144 trajectories |
| 相比旧方法 | 约为 8192 anchors 的 32× 密度 |
| 评分流程 | 路径/速度粗评分 → Top-K 组合 → 轨迹细粒度重评分 |
| NAVSIM v1 | PDMS 92.0，ResNet-34 |
| NAVSIM v2 | EPDMS 90.1，ResNet-34 |
| Bench2Drive | Driving Score 89.15，Success Rate 70.00% |
| 论文类型 | 方法论文、自动驾驶规划研究 |

## 二、极简全文核心总结

SparseDriveV2 重新审视“静态候选轨迹词典是否必须被动态生成替代”这一问题。论文通过 Hydra-MDP 扩展实验发现，轨迹 anchor 越密，规划性能持续提升，瓶颈主要来自覆盖不足而非静态词典本身。为此，SparseDriveV2 将轨迹分解为几何路径和速度曲线，通过组合构造 26 万级超密候选空间，再先分别对路径和速度粗评分、筛选 Top-K，最后对少量组合轨迹进行细粒度评分。在 NAVSIM v1/v2 和 Bench2Drive 上取得领先结果。

## 三、研究背景与研究意义

### 3.1 端到端多模态规划

自动驾驶规划面对同一场景可能存在多种合理行为：

- 直行或变道；
- 跟车或超车；
- 保守刹车或继续通行；
- 不同速度通过路口。

因此，多模态规划通常先生成候选轨迹，再由 scorer 选择最优轨迹：

$$
\tau^*
=
\arg\max_{\tau\in\mathcal{T}}
 s(\tau,o_t)
$$

其中：

- $o_t$：当前传感器观测；
- $\mathcal{T}$：轨迹候选集合；
- $s$：场景条件下的轨迹分数。

### 3.2 静态词典方法

Hydra-MDP、VADv2、DriveSuprim 等方法从大规模驾驶数据聚类得到固定轨迹 vocabulary：

```text
大规模专家轨迹
      ↓ K-means
固定轨迹 anchors
      ↓ scorer
选择当前场景最优轨迹
```

优点：

- 结构简单；
- 推理稳定；
- 容易加入人类和规则教师监督；
- 无需额外动态生成模块。

问题是词典规模受内存和计算限制，通常只有几千条轨迹，导致动作空间离散化过粗。

### 3.3 动态生成方法

DiffusionDrive、GoalFlow、回归式 proposal 方法可以根据场景动态生成轨迹，覆盖更细粒度的动作空间。

但代价是：

- 增加 proposal/generation 模块；
- 扩散方法需要多步去噪；
- 训练和推理复杂度更高；
- 动态生成的候选质量可能不稳定。

### 3.4 论文的核心问题

论文提出：

> 动态轨迹生成是否真的必要？如果静态词典足够密集，是否也可以达到甚至超过动态生成方法？

### 3.5 关键观察：词典规模仍未饱和

论文先对 Hydra-MDP 进行静态词典 scaling study：

| Anchors | EPDMS | 显存 |
|---:|---:|---:|
| 1024 | 85.02 | 9531 MB |
| 2048 | 85.80 | 11451 MB |
| 4096 | 86.33 | 15513 MB |
| 8192 | 86.78 | 23261 MB |
| 16384 | 87.35 | 38877 MB |
| 32768 | OOM | — |

性能在 16,384 anchors 前持续提升，没有出现明显饱和。论文据此认为：

$$
\text{静态词典性能瓶颈}
\approx
\text{动作空间覆盖不足}
$$

## 四、核心方法、模型、公式与流程

### 4.1 SparseDriveV2 整体框架图

![SparseDriveV2 framework](https://arxiv.org/html/2603.29163v1/x1.png)

> **图 1：SparseDriveV2 整体框架。** 模型将时空轨迹分解为几何路径和速度曲线，分别构建路径词典与速度词典，再组合成超密轨迹词典。推理时先独立进行路径/速度粗评分并筛选 Top-K，再对组合轨迹进行细粒度场景条件评分。图片来源：[论文 Figure 1](https://arxiv.org/html/2603.29163v1/x1.png)。

整体流程：

```text
传感器观测 o_t + ego status e_t
              ↓
        场景编码器 Φ
              ↓
        场景特征 F、状态特征 E
              ↓
┌──────────────────────────────┐
│ Factorized Vocabulary         │
│ 1024 geometric paths          │
│  256 velocity profiles        │
│ 组合得到 262,144 trajectories │
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│ Coarse Factorized Scoring     │
│ path score + velocity score   │
└──────────────────────────────┘
              ↓
   Top-K paths × Top-K velocities
              ↓
        composed trajectories
              ↓
┌──────────────────────────────┐
│ Fine-Grained Trajectory Score │
│ trajectory-scene interaction  │
│ metric sub-score prediction   │
└──────────────────────────────┘
              ↓
        aggregate final score
              ↓
        选择最佳轨迹
```

### 4.2 规划问题定义

轨迹由固定时间间隔采样的 waypoint 组成：

$$
\tau
=
\{(x_t,y_t)\}_{t=1}^{T}
$$

其中：

- $T$：规划时域；
- $(x_t,y_t)$：自车坐标系下的第 $t$ 个位置；
- $\Delta t$：时间采样间隔。

静态评分词典为：

$$
\mathcal{T}
=
\{\tau_i\}_{i=1}^{N}
$$

每条候选轨迹得到一个场景条件分数：

$$
 s_i=s(\tau_i,o_t)
$$

最终选择：

$$
\tau^*
=
\arg\max_{\tau\in\mathcal{T}}s(\tau,o_t)
$$

### 4.3 轨迹因子化：Path + Velocity

论文将轨迹分解为：

$$
\tau
\longrightarrow
(p,v)
$$

#### Geometric Path

路径只描述空间几何：

$$
 p
=
\{(x_i,y_i)\}_{i=1}^{S}
$$

相邻路径点按照固定空间间隔 $\Delta s$ 采样。路径表示：

- 转弯方向；
- 道路几何；
- 横向偏移；
- 空间形状。

它不包含时间信息。

#### Velocity Profile

速度曲线描述每个时间步的标量速度：

$$
 v
=
\{v_t\}_{t=1}^{T}
$$

它表示：

- 加速和减速；
- 停车；
- 通过路口的节奏；
- 沿路径前进的速度。

### 4.4 轨迹到 Path/Velocity 的分解

给定原始轨迹 $\tau$，论文先按照累计行驶距离重采样，得到固定空间间隔的 path：

$$
(p,v)=\mathcal{D}(\tau)
$$

速度曲线为：

$$
 v_t
=
\frac{
\left\|
(x_t,y_t)-(x_{t-1},y_{t-1})
\right\|
}{\Delta t}
$$

如果轨迹无法覆盖最大空间范围 $S_{max}$，使用 validity mask 标记缺失路径段。

### 4.5 Path/Velocity 到 Trajectory 的组合

给定路径 $p$ 和速度曲线 $v$，累计行驶距离为：

$$
 s_t
=
\sum_{k=1}^{t}v_k\Delta t
$$

在 path 上以累计距离 $s_t$ 插值得到当前时刻位置：

$$
 \tau
=\mathcal{C}(p,v)
$$

其中 $\mathcal{C}$ 是 path–velocity composition operator。

直观上：

```text
Path：车往哪里走
Velocity：车什么时候走到哪里
Trajectory：两者组合后的完整时空轨迹
```

## 4.6 Factorized Vocabulary Construction

### 4.6.1 Path Vocabulary

从大规模专家轨迹中提取 geometric paths，使用 K-means 聚类：

$$
\mathcal{P}
=\{p_i\}_{i=1}^{N_p}
$$

其中 $N_p$ 是 path anchors 数量。

### 4.6.2 Velocity Vocabulary

对每条专家轨迹计算 velocity profile，再使用 K-means 聚类：

$$
\mathcal{V}
=\{v_j\}_{j=1}^{N_v}
$$

其中 $N_v$ 是 velocity anchors 数量。

### 4.6.3 组合词典

将每条 path 与每条 velocity 组合：

$$
\mathcal{T}
=
\left\{
\tau_{i,j}
\mid
\tau_{i,j}=\mathcal{C}(p_i,v_j),
\ p_i\in\mathcal{P},v_j\in\mathcal{V}
\right\}
$$

词典规模为：

$$
|\mathcal{T}|=N_p\times N_v
$$

论文 NAVSIM v1 设置：

$$
N_p=1024,
\qquad
N_v=256
$$

所以：

$$
|\mathcal{T}|=1024\times256=262,144
$$

相较常见 8192 条轨迹，密度提升：

$$
\frac{262,144}{8,192}=32\times
$$

### 4.6.4 因子化的关键价值

直接存储 262,144 条完整时空轨迹成本高；因子化后只需要存储：

```text
1024 条空间 path
+ 256 条 velocity profile
= 262,144 种组合
```

它利用组合结构扩展动作空间，而不是线性增加完整轨迹模板。

## 4.7 Scalable Scoring

### 4.7.1 Scene Encoding

从传感器观测和 ego status 提取：

$$
F=\Phi(o_t),
\qquad
E=\Psi(e_t)
$$

其中：

- $\Phi$：scene encoder；
- $\Psi$：ego-status encoder；
- $F$：场景特征；
- $E$：自车状态特征。

论文沿用 SparseDrive 的设计，不显式构造 BEV，而是直接使用多视角图像特征进行后续交互。

### 4.7.2 为什么不能直接给所有组合轨迹评分

完整词典有 262,144 条轨迹。如果每条轨迹都做复杂 trajectory-scene interaction：

$$
\mathcal{O}(N_pN_v)
$$

计算和显存都难以接受。

因此论文采用两级评分：

```text
低成本粗评分 → 大规模剪枝
高成本细评分 → 只处理少量候选
```

### 4.7.3 Path Coarse Scoring

每条 path 先通过轻量 MLP 编码：

$$
 e_i^p=E_p(p_i),
\qquad
 e_i^p\in\mathbb{R}^{d}
$$

与场景和 ego status 交互：

$$
 \tilde e_i^p
=
\Psi_p(e_i^p+E,F)
$$

路径分数：

$$
 s_i^p=f_p(\tilde e_i^p)
$$

$\Psi_p$ 可以是：

- multi-head cross-attention；
- 沿 path 几何进行 deformable aggregation。

### 4.7.4 Velocity Coarse Scoring

每条速度曲线编码为：

$$
 e_j^v=E_v(v_j)
$$

与场景交互：

$$
 \tilde e_j^v
=
\Psi_v(e_j^v+E,F)
$$

速度分数：

$$
 s_j^v=f_v(\tilde e_j^v)
$$

速度没有显式空间几何，因此主要使用 cross-attention 与场景特征交互。

### 4.7.5 Top-K Factorized Filtering

分别选出：

$$
\mathcal{P}_{K_p}
=\operatorname{TopK}(s^p,K_p)
$$

$$
\mathcal{V}_{K_v}
=\operatorname{TopK}(s^v,K_v)
$$

组合得到粗筛选候选：

$$
\mathcal{T}_{coarse}
=
\left\{
\mathcal{C}(p_i,v_j)
\mid
p_i\in\mathcal{P}_{K_p},
 v_j\in\mathcal{V}_{K_v}
\right\}
$$

候选数变为：

$$
|\mathcal{T}_{coarse}|=K_pK_v
$$

论文 NAVSIM v1 的 progressive filtering：

```text
初始：1024 paths × 256 velocities
第一层：保留 128 paths × 64 velocities
第二层：保留 20 paths × 20 velocities
最终细评分：400 条组合轨迹
```

### 4.7.6 Fine-Grained Trajectory Scoring

组合轨迹特征首先融合 path 和 velocity embedding：

$$
 e_{i,j}^{\tau}
=\tilde e_i^p+\tilde e_j^v
$$

但简单相加隐含 path 和 velocity 独立，无法处理：

- 急转弯 + 高速度不匹配；
- 拥堵场景 + 高速度不匹配；
- 直行场景 + 大横向偏移不匹配。

因此进行 trajectory re-conditioning：

$$
 \tilde e_{i,j}^{\tau}
=
\Psi_\tau(e_{i,j}^{\tau},F)
$$

最终轨迹分数：

$$
 s_{i,j}^{\tau}
=f_\tau(\tilde e_{i,j}^{\tau})
$$

该阶段可以联合建模：

```text
path geometry + velocity timing + scene context
```

## 4.8 Training Objective

### 4.8.1 Path Scoring Loss

给定 GT path $\hat p$，计算 path anchor 距离：

$$
 d_i^p
=
\frac{1}{|\mathcal{M}|}
\sum_{k\in\mathcal{M}}
\left\|
 p_i(k)-\hat p(k)
\right\|_2^2
$$

构造软标签：

$$
 y^p=\operatorname{Softmax}(-\lambda_p d^p)
$$

path scoring loss：

$$
\mathcal{L}_{path}
=\operatorname{CE}(s^p,y^p)
$$

### 4.8.2 Velocity Scoring Loss

速度 anchor 距离：

$$
 d_j^v
=
\sum_{t=1}^{T}
|v_j(t)-\hat v(t)|
$$

软标签和损失：

$$
 y^v=\operatorname{Softmax}(-\lambda_v d^v)
$$

$$
\mathcal{L}_{vel}
=\operatorname{CE}(s^v,y^v)
$$

### 4.8.3 Fine-Grained Trajectory Loss

组合轨迹与 GT 轨迹的距离：

$$
 d_k^\tau
=
\sum_{t=1}^{T}
\left\|
\tau_k(t)-\hat\tau(t)
\right\|_2^2
$$

轨迹级软分类损失：

$$
\mathcal{L}_{traj}
=
\operatorname{CE}
\left(
 s^\tau,
 \operatorname{Softmax}(-\lambda_\tau d^\tau)
\right)
$$

### 4.8.4 Rule-based Metric Supervision

规则教师对组合轨迹计算：

- 安全；
- 进度；
- 舒适性；
- 交通规则遵守。

学生网络预测子指标 $s^{(m)}$，用 BCE 蒸馏：

$$
\mathcal{L}_{metric}
=
\sum_m
\operatorname{BCE}
\left(
 s^{(m)},\hat s^{(m)}
\right)
$$

### 4.8.5 总目标

$$
\mathcal{L}
=
\mathcal{L}_{path}
+
\mathcal{L}_{vel}
+
\mathcal{L}_{traj}
+
\alpha\mathcal{L}_{metric}
$$

## 4.9 推理流程

```text
1. 编码场景和 ego status
2. 对所有 path anchors 进行粗评分
3. 对所有 velocity anchors 进行粗评分
4. 分别保留 Top-Kp 和 Top-Kv
5. 组合候选 path × velocity
6. 对组合轨迹进行细粒度 scene interaction
7. 预测 trajectory score 和 metric sub-scores
8. 按 benchmark 权重聚合最终分数
9. 选择最高分轨迹
10. 输出轨迹或转换为车辆控制信号
```

## 五、核心代码讲解（对应官方 GitHub 实现）

> 代码仓库：[swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2)。以下内容根据公开 `main` 分支源码整理，重点解释模型入口、因子化词典、两阶段筛选、细粒度评分与训练损失。

### 5.1 代码目录与调用链

| 模块 | 主要源码 | 作用 |
|---|---|---|
| 模型入口 | [`sparsedrive_model.py`](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/sparsedrive_model.py) | `SparseDriveModel`、`TrajectoryHead` |
| 评分 Decoder | [`custom_decoder.py`](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/custom_decoder.py) | 路径、速度、轨迹三级评分与筛选 |
| Backbone | [`sparsedrive_backbone.py`](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/sparsedrive_backbone.py) | ResNet 图像特征提取 |
| Deformable Block | [`blocks.py`](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/blocks.py) | 沿候选轨迹坐标采样图像特征 |
| 配置 | [`sparsedrive_config.py`](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/sparsedrive_config.py) | 词典、Top-K、损失和输入配置 |
| Agent | [`sparsedrive_agent.py`](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/sparsedrive_agent.py) | 传感器输入、模型调用、轨迹输出 |

核心调用链：

```text
SparseDriveAgent
    ↓
SparseDriveModel.forward()
    ├─ SparseBackbone(imgs)
    └─ TrajectoryHead(camera_feature, status_encoding, targets)
          ↓
      CustomTransformerDecoder
          ├─ path scoring
          ├─ velocity scoring
          ├─ Top-K filtering
          ├─ trajectory re-conditioning
          └─ metric scoring + argmax
          ↓
      selected trajectory
```

### 5.2 `SparseDriveModel`：模型入口

源码中的模型主要构造：

```python
self._backbone = SparseBackbone(config)
self._status_encoding = nn.Linear(4 + 2 + 2, config.d_model)
self._trajectory_head = TrajectoryHead(...)
```

前向过程可概括为：

```python
status_encoding = self._status_encoding(status_feature)
feature_maps = self._backbone(camera_feature["imgs"])
camera_feature["feature_maps"] = feature_maps
trajectory, loss_dict = self._trajectory_head(
    camera_feature, status_encoding, targets
)
```

训练时将各项 loss 汇总为：

```python
loss_dict["loss"] = sum(loss_dict.values())
```

因此该实现是**固定候选轨迹的条件评分器**，不是动态轨迹生成器。

### 5.3 `TrajectoryHead`：加载因子化词典

源码从配置指定的 `.npy/.npz` 文件加载固定词典，并设置 `requires_grad=False`：

```python
self.path_vocab = nn.Parameter(
    torch.from_numpy(np.load(config.path_anchor)).float(),
    requires_grad=False,
)
self.vel_vocab = nn.Parameter(
    torch.from_numpy(np.load(config.velocity_anchor)).float(),
    requires_grad=False,
)
trajectory_data = np.load(config.trajectory_anchor)
self.traj_vocab = nn.Parameter(
    torch.from_numpy(trajectory_data["trajectory"]).float(),
    requires_grad=False,
)
self.traj_mask = nn.Parameter(
    torch.from_numpy(trajectory_data["trajectory_mask"]).float(),
    requires_grad=False,
)
```

默认配置：

```python
mode_path = 1024
len_path = 50
path_interval = 1.0
mode_vel = 256
len_vel_seq = 8
vel_time_interval = 0.5
decoder_num_layers = 2
path_filter_num = (128, 20)
velocity_filter_num = (64, 10)
```

词典 embedding：

```python
path_embed = self.path_pos_embed(path_vocab.flatten(-2, -1))
vel_embed = self.vel_pos_embed(vel_vocab)
```

这里的 `path_vocab`、`vel_vocab` 是可组合的静态候选；模型只学习如何在场景条件下评分它们。

### 5.4 Path Scoring：路径级粗评分

`CustomTransformerDecoderLayer` 使用 `DeformableFeatureAggregation` 沿路径坐标读取多视角图像特征：

```python
path_vocab_flat = path_vocab[..., :2].flatten(-2)
path_embed = self.p_deform_model(
    path_embed, path_vocab_flat, None,
    deform_value, camera_feature, None,
)
```

之后执行 path query self-attention、FFN 和 MLP：

```python
path_embed = path_embed + self.p_dropout1(
    self.p_attention(path_embed, path_embed, path_embed)[0]
)
path_embed = self.p_norm1(path_embed)
path_embed = path_embed + self.p_dropout2(self.p_ffn(path_embed))
path_embed = self.p_norm2(path_embed)
path_scores = self.path_mlp(path_embed).squeeze(-1)
```

对应论文公式：

$$
 p_i\rightarrow e_i^p\rightarrow\tilde e_i^p\rightarrow s_i^p
$$

因为 path 含有显式空间几何，代码用 deformable aggregation 沿路径位置采样场景特征。

### 5.5 Velocity Scoring：速度级粗评分

速度曲线没有显式空间坐标，因此源码使用图像 feature 的普通 multi-head attention：

```python
vel_embed = vel_embed + self.v_img_attention(
    vel_embed, img_value, img_value
)[0]
vel_embed = vel_embed + self.v_dropout1(
    self.v_attention(vel_embed, vel_embed, vel_embed)[0]
)
vel_embed = self.v_norm1(vel_embed)
vel_embed = vel_embed + self.v_dropout2(self.v_ffn(vel_embed))
vel_embed = self.v_norm2(vel_embed)
vel_scores = self.vel_mlp(vel_embed).squeeze(-1)
```

对应：

$$
 v_j\rightarrow e_j^v\rightarrow\tilde e_j^v\rightarrow s_j^v
$$

### 5.6 两阶段 Top-K 筛选

源码分别对 path 和 velocity 使用 `torch.topk`：

```python
topk_path_scores, topk_path_indices = torch.topk(
    path_scores,
    config.path_filter_num[decoder_idx],
    dim=1,
)
topk_vel_scores, topk_vel_indices = torch.topk(
    vel_scores,
    config.velocity_filter_num[decoder_idx],
    dim=1,
)
```

随后通过 `torch.gather` 同步筛选：

- path/velocity embedding；
- path/velocity vocabulary；
- 组合 trajectory vocabulary；
- trajectory mask。

默认流程：

```text
第 1 层：1024 paths → 128，256 velocities → 64
第 2 层：128 paths → 20，64 velocities → 20
最终细评分：20 × 20 = 400 条组合轨迹
```

源码中 `filter_traj_vocab` 保持 path × velocity 的二维结构，因此无需先对全部 262,144 条轨迹执行完整重型 attention。

### 5.7 Fine-grained Trajectory Re-conditioning

只有最后一层 decoder 执行完整轨迹评分：

```python
traj_embed = (
    filter_path_embed.unsqueeze(2)
    + filter_vel_embed.unsqueeze(1)
).flatten(1, 2)
```

对应：

$$
 e_{i,j}^{\tau}=\tilde e_i^p+\tilde e_j^v
$$

然后将组合轨迹的坐标送入 deformable aggregation，再执行轨迹 self-attention 和 MLP：

```python
traj_vocab_flat = (
    filter_traj_vocab[..., :2]
    .flatten(1, 2)
    .flatten(-2)
)
traj_embed = self.t_deform_model(
    traj_embed, traj_vocab_flat, None,
    deform_value, camera_feature, None,
)
traj_embed = traj_embed + self.t_dropout1(
    self.t_attention(traj_embed, traj_embed, traj_embed)[0]
)
traj_embed = self.t_norm1(traj_embed)
traj_embed = traj_embed + self.t_dropout2(self.t_ffn(traj_embed))
traj_embed = self.t_norm2(traj_embed)
traj_scores = self.traj_mlp(traj_embed).squeeze(-1)
```

该模块修正 path 和 velocity 独立评分带来的问题，例如急转弯与高速组合不合理。

### 5.8 Metric Heads 与最终选择

最后一层根据配置动态创建 metric heads：

```python
for metric in config.metrics:
    self.metric_heads[metric] = nn.Sequential(
        nn.Linear(d_model, d_ffn),
        nn.ReLU(),
        nn.Linear(d_ffn, 1),
    )
```

NAVSIM v2 默认预测：

```text
NC、DAC、DDC、TL、TTC、EP、LK、HC
```

源码将 logits 经过 sigmoid，并按 benchmark 规则组合。例如 v2：

```python
scores = (
    metric_logit["no_at_fault_collisions"].sigmoid()
    * metric_logit["drivable_area_compliance"].sigmoid()
    * metric_logit["driving_direction_compliance"].sigmoid()
    * metric_logit["traffic_light_compliance"].sigmoid()
) * (
    5 * metric_logit["time_to_collision_within_bound"].sigmoid()
    + 5 * metric_logit["ego_progress"].sigmoid()
    + 2 * metric_logit["lane_keeping"].sigmoid()
    + 2 * metric_logit["history_comfort"].sigmoid()
)
```

最终：

```python
mode_indices = scores.argmax(1)
trajectory = filter_traj_vocab.flatten(1, 2)[
    batch_indices, mode_indices
]
```

也就是说，推理阶段不是生成新轨迹，而是对筛选后的静态候选做 score argmax。

### 5.9 训练损失的源码实现

#### Path loss

```python
diff = (path_vocab - target_path[:, None])[..., :2]
dist = diff.pow(2).sum(-1)
dist = (dist * target_path_mask[:, None]).sum(-1) / valid_cnt
dist = dist * config.path_sigmas * config.len_path
path_loss = F.cross_entropy(
    path_scores, (-dist).softmax(1)
)
```

这是基于 GT path 距离的 soft classification：距离越小，目标概率越高。

#### Velocity loss

```python
dist = (vel_vocab - target_vel[:, None]).abs()
dist = dist.sum(-1) * config.velocity_sigmas
vel_loss = F.cross_entropy(
    vel_scores, (-dist).softmax(1)
)
```

#### Trajectory imitation loss

```python
dist = (
    filter_traj_vocab.flatten(1, 2)
    - target_traj[:, None]
)[..., :2] ** 2
dist = dist.sum((-2, -1)) * config.trajectory_sigmas
traj_loss = F.cross_entropy(
    traj_scores, (-dist).softmax(1)
)
```

#### Rule metric loss

训练时，代码根据 `dataset_version` 调用 v1/v2 的 NAVSIM scorer，生成候选轨迹规则标签，再使用 BCE：

```python
metric_loss = F.binary_cross_entropy_with_logits(
    metric_pred,
    metric_gt,
)
```

最终各项 loss 由 `SparseDriveModel` 汇总。

### 5.10 源码级伪代码

```python
def forward(camera_feature, status_encoding, targets):
    path = load_path_vocab()
    vel = load_velocity_vocab()
    traj = load_composed_trajectory_vocab()

    path_embed = encode_path(path) + status_encoding
    vel_embed = encode_velocity(vel) + status_encoding

    for layer in decoder_layers:
        path_feat, path_score = score_paths(path_embed, path, scene)
        vel_feat, vel_score = score_velocities(vel_embed, vel, scene)

        path_idx = topk(path_score, layer.path_filter_num)
        vel_idx = topk(vel_score, layer.velocity_filter_num)

        path_embed, path = gather(path_embed, path, path_idx)
        vel_embed, vel = gather(vel_embed, vel, vel_idx)
        traj = gather_composed_vocab(traj, path_idx, vel_idx)

    traj_feat = compose(path_embed, vel_embed)
    traj_feat = deformable_scene_interaction(traj_feat, traj)
    traj_score = score_trajectory(traj_feat)
    metric_logits = score_metrics(traj_feat)

    final_score = benchmark_aggregate(metric_logits)
    selected = traj[final_score.argmax(dim=1)]
    return selected
```

### 5.11 源码实现与论文描述的注意事项

1. **词典是固定参数。** `path_vocab`、`vel_vocab` 和 `traj_vocab` 不参与梯度更新。
2. **完整 trajectory vocabulary 已在数据准备阶段构造。** 模型前向主要通过 gather 沿 path/velocity 轴筛选。
3. **完整轨迹评分只在最后 decoder layer 执行。** 前面各层主要负责粗评分和 progressive pruning。
4. **Path 使用 deformable geometry-aware scoring，velocity 使用 image attention。** 这是二者代码实现的主要差异。
5. **Metric supervision 与配置相关。** NAVSIM v1/v2 使用不同 scorer 和 metric 列表；Bench2Drive 附录不使用 metric-based supervision。
6. **最终是 score argmax，不是 diffusion/flow 轨迹生成。**

## 六、核心创新点与传统方法对比

### 6.1 创新一：因子化轨迹词典

将完整时空轨迹拆成：

$$
\text{Trajectory}
=
\text{Path}
\otimes
\text{Velocity}
$$

利用组合覆盖大量动作空间。

### 6.2 创新二：超密静态词典

SparseDriveV2 使用：

$$
1024\times256=262,144
$$

条组合轨迹，密度为传统 8192 anchors 的 32 倍。

### 6.3 创新三：因子化粗评分

不直接为每条完整轨迹做重型评分，而是先分别评估 path 和 velocity，再剪枝。

### 6.4 创新四：轨迹重新条件化

粗评分假设 path/velocity 可部分独立；细评分阶段重新让组合轨迹与场景交互，修正二者之间的时空依赖。

### 6.5 创新五：纯 scoring-based 路线

与 DiffusionDrive、GoalFlow 等动态生成方法不同，SparseDriveV2 不增加动态轨迹生成或扩散去噪模块。

### 6.6 方法对比

| 方法 | 候选来源 | 轨迹覆盖 | 额外生成模块 | 主要瓶颈 |
|---|---|---:|---:|---|
| Hydra-MDP | 静态完整轨迹 anchors | 受 anchor 数量限制 | 否 | 词典稀疏、评分成本 |
| DiffusionDrive | 动态扩散生成 | 灵活 | 是 | 多步采样、训练复杂 |
| GoalFlow | Flow matching 生成 | 灵活 | 是 | 生成模块复杂 |
| GTRS | 静态 + 动态混合 | 较高 | 是 | 多模块复杂度 |
| SparseDriveV2 | Path × Velocity 组合 | 超密 | 否 | 粗筛选和权重设计 |

## 七、理论分析与关键假设

### 7.1 组合覆盖的基本假设

因子化表示假设空间几何和时间速度具有一定可分离性：

$$
 p(\tau)\approx p(p)\,p(v)
$$

这不是严格统计独立，而是用于构造可扩展候选空间的结构假设。

### 7.2 细评分修正独立性假设

由于 path 与 velocity 在实际驾驶中存在耦合，模型使用：

$$
\tilde e_{i,j}^{\tau}
=\Psi_\tau(e_i^p+e_j^v,F)
$$

重新注入 scene context，从而学习：

```text
空间路径是否适合当前速度
速度曲线是否适合当前道路形状
组合是否符合交通场景
```

### 7.3 复杂度分析

完整组合空间：

$$
N_pN_v
$$

粗评分复杂度近似：

$$
\mathcal{O}(N_p+N_v)
$$

细评分复杂度：

$$
\mathcal{O}(K_pK_v)
$$

只要：

$$
K_pK_v\ll N_pN_v
$$

就能以较低成本处理超密词典。

### 7.4 方法没有保证的内容

论文没有严格保证：

- path/velocity 组合能够覆盖真正最优轨迹；
- 粗评分不会误删最终最优 path 或 velocity；
- top-K 筛选后的 400 条轨迹包含所有关键行为模式；
- K-means 得到的 path/velocity anchors 对所有场景均匀有效；
- 静态词典在任意复杂场景都能替代动态轨迹生成。

### 7.5 Scoring 与真实驾驶目标的差异

训练使用 imitation soft labels 和 rule-based metric supervision，最终 benchmark 使用综合指标。因此：

$$
\text{score prediction accuracy}
\not\Rightarrow
\text{真实闭环最优}
$$

尤其是规则教师、候选词典和 benchmark 评分之间可能存在分布偏差。

## 八、实验设计与结果分析

### 8.1 数据集与指标

#### NAVSIM

- navtrain：1192 场景；
- navtest：136 场景；
- NAVSIM v1：PDMS；
- NAVSIM v2：EPDMS。

PDMS：

$$
PDMS
=
NC\times DAC\times
\frac{5TTC+2C+5EP}{12}
$$

EPDMS：

$$
EPDMS
=
NC\times DAC\times DDC\times TL
\times
\frac{5EP+5TTC+2LK+2C+2EC}{16}
$$

#### Bench2Drive

CARLA 闭环 benchmark：

- 220 条测试路线；
- 44 类交互场景；
- Driving Score；
- Success Rate；
- Efficiency；
- Comfort；
- Multi-Ability。

### 8.2 NAVSIM v1 主结果

| 方法 | Backbone | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| DiffusionDriveV2 | ResNet-34 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |
| ipad | ResNet-34 | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 |
| SparseDriveV2 | ResNet-34 | 98.5 | 98.4 | 95.0 | 99.9 | 88.6 | **92.0** |

论文报告 SparseDriveV2：

- 超过 dynamic generation 方法 DiffusionDriveV2；
- 超过使用更大 V2-99 backbone 的 Hydra-MDP 和 GoalFlow；
- 证明 factorized vocabulary + scalable scoring 可在轻量 backbone 上取得较强效果。

### 8.3 NAVSIM v2 主结果

| 方法 | Backbone | EPDMS* | EPDMS |
|---|---|---:|---:|
| DiffusionDriveV2 | ResNet-34 | 85.5 | 87.5 |
| Hydra-MDP++ | V2-99 | 85.1 | — |
| DriveSuprim | V2-99 | 86.0 | — |
| SparseDriveV2 | ResNet-34 | **86.7** | **90.1** |

论文说明：

- EPDMS* 是旧评测实现下的结果；
- EPDMS 是修正人类行为过滤问题后的新实现；
- 论文建议使用更新后的官方实现进行比较。

### 8.4 Bench2Drive 主结果

| 方法 | Driving Score | Success Rate | Efficiency | Comfort |
|---|---:|---:|---:|---:|
| SimLingo | 86.02 | 67.27 | 259.23 | 33.67 |
| HiP-AD | 86.77 | 69.09 | 203.12 | 19.36 |
| SparseDriveV2 | **89.15** | **70.00** | 199.84 | 18.32 |

### 8.5 Bench2Drive 多能力

| 方法 | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|---|---:|---:|---:|---:|---:|---:|
| HiP-AD | 50.00 | 84.44 | 83.33 | 40.00 | 72.10 | 65.98 |
| SparseDriveV2 | **66.25** | **75.55** | 75.00 | 50.00 | 71.57 | **67.67** |

SparseDriveV2 的平均多能力分数最高，在 merging 和 overtaking 上明显提升，但 emergency brake 和 traffic sign 并非所有方法中最高。

### 8.6 Vocabulary Scaling 消融

论文 Table 6：

| Path anchors $N_p$ | Velocity anchors $N_v$ | EPDMS |
|---:|---:|---:|
| 512 | 128 | 88.7 |
| 512 | 256 | 89.2 |
| 1024 | 128 | 89.5 |
| 1024 | 256 | **90.1** |

路径和速度 anchor 同时增加时性能最佳，说明二者都影响动作空间覆盖。

### 8.7 Scalable Scoring 消融

| Path Attention | Re-conditioning | EPDMS |
|---|---:|---:|
| MHA | 否 | 87.7 |
| MHA | 是 | 89.9 |
| DFA | 否 | 89.9 |
| DFA | 是 | **90.1** |

结论：

- deformable aggregation 比普通 MHA 更有效；
- trajectory re-conditioning 能稳定提升两种 attention 配置；
- 组合轨迹的场景条件再建模很重要。

### 8.8 Efficiency 与候选规模

NAVSIM v1：

- 全词典：262,144 条组合轨迹；
- 第一层筛选：128 path × 64 velocity；
- 第二层筛选：20 path × 20 velocity；
- 细评分：400 条轨迹。

因此昂贵的 metric supervision 和 trajectory-level scoring 只作用于 400 条候选，而不是全部 262,144 条。

### 8.9 实验结论与证据边界

实验支持：

- 静态词典性能随密度提升；
- path/velocity 因子化可以扩展候选覆盖；
- coarse-to-fine scoring 能降低超密词典成本；
- SparseDriveV2 在 NAVSIM 和 Bench2Drive 上表现强。

实验不能完全证明：

- 动态生成在原则上不必要；
- 262,144 条组合轨迹覆盖所有真实最优行为；
- 粗评分不会误删关键轨迹；
- NAVSIM/Bench2Drive 结果可直接推广到真实道路。

## 九、学术价值、局限性与潜在漏洞

### 9.1 学术价值

1. **重新审视静态词典的上限。** 论文用 scaling study 说明以前的瓶颈可能是词典太稀疏。
2. **提出轨迹因子化表示。** 将空间几何和时间速度解耦，利用组合扩展覆盖。
3. **提出超密词典的高效评分。** 先因子化粗筛选，再进行小规模细评分。
4. **保持 scoring-based 方法简洁。** 不需要额外扩散生成器或迭代 denoising。
5. **轻量 backbone 仍有较强性能。** ResNet-34 达到 NAVSIM v1 PDMS 92.0。

### 9.2 论文暴露的限制

- 静态 vocabulary 仍受 path/velocity 聚类质量限制；
- top-K 剪枝可能丢弃最优组合；
- 规则 metric supervision 需要候选轨迹离线评估；
- Bench2Drive 使用纯 imitation learning，不使用 metric supervision；
- 真实道路验证仍然不足；
- 论文中缺少对全部粗筛选失败案例的系统分析。

### 9.3 分析者识别出的潜在问题

#### 问题一：因子化组合可能产生大量不合理轨迹

并非所有 path 和 velocity 都物理兼容：

```text
急转弯 + 高速
倒车式速度曲线 + 前进路径
拥堵场景 + 高速 profile
```

虽然 fine-grained scoring 可以过滤一部分，但大量组合仍需要先构造或表示。

#### 问题二：粗评分存在不可逆误删风险

如果正确 path 或 velocity 在 coarse score 中排名较低，它会在进入 fine scoring 前被删除：

$$
\text{coarse score error}
\Rightarrow
\text{optimal trajectory permanently lost}
$$

论文没有完整报告 recall@K 或粗筛选对 oracle best trajectory 的保留率。

#### 问题三：Path/velocity 的独立聚类损失联合分布

分别 K-means 得到 path 和 velocity，再进行笛卡尔组合，可能破坏专家数据中的联合分布。虽然 trajectory re-conditioning 可以修正，但不能恢复被词典表示完全遗漏的耦合模式。

#### 问题四：Scaling study 受显存限制截断

Hydra-MDP 的 32768 anchors 直接 OOM，并没有观察到真正性能饱和。结论“静态词典可以继续扩展”合理，但仍是受限外推，不是对无限 scaling 的证明。

#### 问题五：Benchmark 分数与真实驾驶目标不完全等价

高 PDMS/EPDMS 代表在指定仿真评分下表现好，但不一定覆盖：

- 真实交通参与者反应；
- 传感器异常；
- 长尾道路结构；
- 真实车辆动力学。

#### 问题六：不同 benchmark 的训练协议不一致

NAVSIM 使用 metric supervision；Bench2Drive 附录明确说明不使用 metric-based supervision，采用纯 imitation learning。因此两组结果不能简单归因于同一套训练目标。

#### 问题七：动态生成与静态评分比较仍不完全公平

动态方法和 SparseDriveV2 在模型结构、采样过程、训练目标、候选数量和计算预算上可能不同。论文证明了 SparseDriveV2 的竞争力，但不能单独证明动态生成没有价值。

## 十、通俗讲解

### 10.1 传统静态词典的问题

可以把轨迹词典理解为导航软件里预先准备的路线模板：

```text
模板 1：直行
模板 2：左转
模板 3：右转
模板 4：变道
...
```

模板太少时，车辆只能选择“差不多”的路线，无法精确贴合场景。

### 10.2 动态生成的做法

另一种方法是每次根据场景现场生成路线：

```text
当前场景 → 现场生成很多轨迹
```

灵活，但需要更复杂的生成器，甚至需要多步扩散采样。

### 10.3 SparseDriveV2 的做法

SparseDriveV2 不直接准备几十万条完整轨迹，而是把轨迹拆成两部分：

```text
路径：车往哪里走
速度：车什么时候走到哪里
```

然后组合：

```text
1024 条路径 × 256 条速度曲线
= 262,144 条完整轨迹
```

例如同一条左转路径，可以搭配：

- 慢速左转；
- 正常速度左转；
- 先减速再加速左转；
- 停车后再左转。

### 10.4 为什么不会慢到无法运行

它不对 26 万条轨迹全部做复杂判断，而是分两步：

```text
先判断哪些路径可能合理
先判断哪些速度可能合理
        ↓
只保留少量路径和速度
        ↓
组合后精细判断
```

最终只对 400 条轨迹进行细评分。

### 10.5 为什么路径和速度还要重新组合判断

单独看路径和速度不够：

```text
这条路径看起来合理
这条速度曲线也看起来合理
```

但组合起来可能不合理：

```text
急转弯 + 高速 = 危险
```

所以最后还要把组合后的完整轨迹重新放回场景中评估。

### 10.6 一句话理解

> SparseDriveV2 用“路径词典 × 速度词典”制造超密轨迹空间，再用粗到细评分快速筛选，从而在不使用动态轨迹生成的情况下获得更精细的多模态规划能力。

## 十一、综合评价与后续研究方向

### 11.1 综合评价

SparseDriveV2 的核心贡献可以概括为：

$$
\text{Factorized Vocabulary}
+
\text{Combinatorial Coverage}
+
\text{Coarse-to-Fine Scoring}
$$

完整因果链：

$$
\text{专家轨迹}
\rightarrow
\begin{cases}
\text{Path K-means}\
\text{Velocity K-means}
\end{cases}
\rightarrow
\text{Path × Velocity 组合词典}
\rightarrow
\text{因子化粗评分}
\rightarrow
\text{Top-K 组合}
\rightarrow
\text{细粒度轨迹评分}
\rightarrow
\text{选择最终轨迹}
$$

论文最有价值的观察是：静态词典方法在较大规模下仍未明显饱和，过去性能不足可能主要源于 action space 覆盖不足，而非静态词典范式本身不可行。

方法工程上较简洁：

- 不增加动态扩散生成器；
- 不需要迭代轨迹生成；
- 通过 path/velocity 因子化降低词典存储和评分成本；
- 通过 coarse-to-fine 保持细粒度选择能力。

实验结果较强：

- NAVSIM v1 PDMS 92.0；
- NAVSIM v2 EPDMS 90.1；
- Bench2Drive Driving Score 89.15；
- Success Rate 70.00%。

但更准确的结论是：

> SparseDriveV2 证明了“足够密集的静态候选空间 + 高效评分”可以达到很强的闭环规划性能，但尚未证明静态词典在动作空间覆盖、分布外行为和真实道路泛化方面可以全面替代动态轨迹生成。

### 11.2 后续研究方向

#### 方向一：粗评分 recall 保证

研究粗筛选对 oracle 最优轨迹的保留率，使用候选召回约束避免最优轨迹被过早删除。

#### 方向二：学习 path/velocity 联合分布

在因子化词典基础上加入 compatibility model 或 joint prior，减少不合理的笛卡尔组合。

#### 方向三：自适应 Top-K

根据场景复杂度、交通密度和不确定性动态调整 $K_p,K_v$，简单场景加速，复杂场景扩大搜索。

#### 方向四：连续 refinement

在细评分后对 Top-1/Top-K 轨迹进行小幅连续优化，结合静态词典的稳定性和动态方法的精细度。

#### 方向五：多目标和风险敏感评分

将 NC、TTC、EP、舒适性和交通规则建模为多目标 Pareto 选择，而非固定加权求和。

#### 方向六：与 diffusion/flow hybrid

使用因子化静态词典提供全局候选覆盖，再用轻量 diffusion/flow 对局部候选进行 refinement。

#### 方向七：更强导航和地图条件

论文附录中的失败案例显示，部分错误来自导航信息不足。加入 HD map、route command、导航 goal 和语言条件可能改善高层决策。

#### 方向八：真实闭环和跨域验证

在真实车辆、不同交通规则、不同驾驶风格和分布外道路中验证超密静态词典的可靠性。

## 一句话结论

> SparseDriveV2 通过将轨迹分解为几何路径和速度曲线，利用组合结构构造超密候选空间，再以粗到细的评分策略高效筛选，在不引入动态轨迹生成的情况下显著提升了端到端多模态规划性能。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2603.29163](https://arxiv.org/abs/2603.29163)
- 论文 HTML：[https://arxiv.org/html/2603.29163v1](https://arxiv.org/html/2603.29163v1)
- 论文 PDF：[https://arxiv.org/pdf/2603.29163](https://arxiv.org/pdf/2603.29163)
- 代码：[https://github.com/swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2)
