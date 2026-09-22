# World Simulation with Video Foundation Models for Physical AI：论文分析

> 标题：*World Simulation with Video Foundation Models for Physical AI*  
> 来源：[arXiv:2511.00062](https://arxiv.org/abs/2511.00062)  
> 版本：v2，2026 年 2 月 24 日修订；首次提交于 2025 年 10 月 28 日  
> 作者：NVIDIA 及大规模作者团队。完整贡献者名单见论文 Appendix A；论文原文未给出单一第一作者。  
> 正式发表信息：论文原文未说明正式会议或期刊。  
> 官方代码：[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)、[Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | Physical AI、视频世界模型、机器人与自动驾驶仿真 |
| 核心模型 | Cosmos-Predict2.5、Cosmos-Transfer2.5 |
| 生成形式 | 视频形式的未来世界状态 |
| 主架构 | Latent DiT + Flow Matching |
| 条件类型 | 文本、图像、视频、动作、边缘/模糊/深度/分割等空间控制 |
| 模型规模 | Cosmos-Predict2.5-2B、14B；Transfer2.5-2B |
| 训练数据 | 2 亿条 curated video clips；领域后训练数据 |
| 主要模式 | Text2World、Image2World、Video2World |
| 关键训练 | 渐进式预训练、领域 SFT、model merging、RL、timestep distillation |
| 主要应用 | 合成数据、策略评估、闭环仿真、Sim2Real/Real2Real、VLA 训练 |
| 论文性质 | 系统型技术报告/模型论文，包含数据、训练、架构、应用和开源资源 |

## 二、极简全文核心总结

本文发布 NVIDIA Cosmos-Predict2.5 和 Cosmos-Transfer2.5。前者采用 Flow Matching 和 latent DiT，将 Text2World、Image2World、Video2World 统一到 2B/14B 模型中，并以 Cosmos-Reason1 提供更强文本条件；后者以 ControlNet 风格融合深度、分割、边缘和模糊等空间控制。通过 2 亿视频数据、领域 SFT、模型合并和 RL，模型在视频质量、指令对齐和 Physical AI 场景上提升，并支持机器人、自动驾驶和 VLA 数据生成。

## 三、研究背景与研究意义

### 3.1 Physical AI 为什么需要世界模型

Physical AI agent 通过传感器观察环境，并通过动作改变环境：

```text
传感器观测 → 策略决策 → 物理动作 → 环境变化
```

直接在真实世界训练存在：

- 成本高；
- 速度慢；
- 真实试错可能损坏机器人或车辆；
- 危险动作难以大规模采集；
- 长尾场景覆盖不足。

视频世界模型提供了一个安全的“硅中训练”环境：

$$
(\text{condition},\text{action})
\rightarrow
\text{future video/world state}
$$

### 3.2 Cosmos-Predict1 的改进空间

论文将 Cosmos-Predict2.5 相比 Cosmos-Predict1 的改进概括为：

1. 更强的视频过滤和领域数据整理；
2. 简化架构，并统一 Text2World、Image2World、Video2World；
3. 使用 Flow Matching 替代旧版本的 EDM parameterization；
4. 使用 Cosmos-Reason1 替代 T5 文本编码器；
5. 通过 SFT、model merging 和 RL 改善物理 AI 场景。

### 3.3 研究意义

论文不是只提出一个视频生成网络，而是构建一套 Physical AI 基础设施：

```text
大规模视频数据
    ↓
通用世界生成模型
    ↓
领域后训练与控制条件
    ↓
机器人 / 自动驾驶 / VLA 应用
```

其价值在于把视频生成从“视觉内容生成”推进到“动作、环境和策略评估的模拟工具”。

## 四、核心方法、模型、公式与流程

### 4.1 总体框架图

![Cosmos-Predict2.5 architecture](https://arxiv.org/html/2511.00062v2/cosmos-predict2.5.png)

> **图 2：Cosmos-Predict2.5 总体架构。** 输入条件经 Cosmos-Reason1 编码后，通过 latent DiT 的 self-attention、cross-attention 和 FFN 预测 Flow Matching 速度；网络使用 AdaLN 调制时间步。图片来源：[论文 Figure 2](https://arxiv.org/html/2511.00062v2/cosmos-predict2.5.png)。

```text
Text / Image / Video / Action / Control maps
                  ↓
          tokenizer / condition encoder
                  ↓
       noisy latent x_t + condition c
                  ↓
 latent DiT: self-attn → cross-attn → FFN
                  ↓
          predicted velocity u_theta
                  ↓ ODE solver / distillation
             clean video latent
                  ↓
             WAN2.1 VAE decoder
                  ↓
              generated video
```

### 4.2 Flow Matching

设视频或图像数据 latent 为 $x$，高斯噪声为：

$$
\epsilon\sim\mathcal{N}(0,I)
$$

论文使用从数据到噪声的线性路径：

$$
 x_t=(1-t)x+t\epsilon,
 \qquad t\in[0,1]
$$

对应真实速度为：

$$
 v_t=\epsilon-x
$$

模型 $u(x_t,t,c;\theta)$ 预测速度，使用 MSE 训练：

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{x,\epsilon,c,t}
\left[
\left\|
 u(x_t,t,c;\theta)-v_t
\right\|_2^2
\right]
$$

符号含义：

- $x$：真实图像/视频 latent；
- $\epsilon$：终点噪声；
- $x_t$：时间 $t$ 的插值 latent；
- $c$：文本、参考帧、视频或其他条件；
- $u$：网络预测的速度场。

推理时从噪声出发，沿预测速度场反向积分至数据分布。论文称 FM 与 EDM 在前向/反向过程上具有数学等价性，但网络参数化不同：

- EDM 更强调输入输出标准化和 denoising 参数化；
- FM 直接预测扩散轨迹速度。

### 4.3 Shifted Logit-Normal 时间采样

高分辨率图像具有强局部相关性。若训练过多集中于低噪声，模型可能没有充分学习如何从高度破坏的 latent 恢复结构。

论文先从 logit-normal 分布采样 $t$，再进行 shift：

$$
 t_s=
 \frac{\beta t}{1+(\beta-1)t}
$$

其中 $\beta$ 是 shift 超参数：

- $\beta=1$：不改变时间分布；
- 增大 $\beta$：把采样推向更高噪声区域。

训练中还显式将 5% 样本放到噪声分布最高 2% 的区域，以减少视频帧之间突变和不自然过渡。

### 4.4 Latent 视频表示

Cosmos-Predict2.5 使用 WAN2.1 causal VAE：

- 时间维压缩 4 倍；
- 高度压缩 8 倍；
- 宽度压缩 8 倍。

之后再使用 $1\times2\times2$ patchification。模型一次生成 93 个像素帧，对应 24 个 latent frames。视频帧率为 16 fps，因此约对应：

$$
\frac{93}{16}\approx5.8\text{ 秒}
$$

这里的 93 是像素视频帧数；24 是经过时间压缩后的 latent 帧数。

### 4.5 Latent DiT 网络

Cosmos-Predict2.5 延续 Cosmos-Predict1 的 latent DiT，但做了关键调整：

- 移除 absolute positional embedding；
- 保留 3D RoPE relative positional encoding；
- 使用 repeated self-attention、cross-attention 和 FFN blocks；
- 用 AdaLN-LoRA 根据时间步调制 block。

单个 block 可以抽象为：

```text
latent tokens
    ↓
AdaLN(t) + self-attention
    ↓
AdaLN(t, condition) + cross-attention
    ↓
AdaLN(t) + FFN/MLP
    ↓
next latent tokens
```

去掉 absolute position 的动机是改善对不同分辨率和更长视频序列的泛化。

### 4.6 文本条件：Cosmos-Reason1

Cosmos-Predict2.5 不再使用 Cosmos-Predict1 的 T5 encoder，而使用 Cosmos-Reason1。

其文本表示不是只取单个 Transformer block 的输出，而是：

1. 收集多个 block 的 token activations；
2. 拼接不同层的表示；
3. 投影到 1024 维 embedding；
4. 通过 DiT cross-attention 注入 latent。

因此条件链为：

$$
\text{text prompt}
\rightarrow
\text{Cosmos-Reason1}
\rightarrow
\text{multi-layer token embeddings}
\rightarrow
\text{DiT cross-attention}
$$

### 4.7 三种统一生成模式

#### Text2World

仅输入文本，生成符合文本的世界视频：

$$
\text{text}
\rightarrow
\text{video}
$$

#### Image2World

输入文本和参考图像，生成以参考图像为起点或条件的未来视频：

$$
(\text{text},I_0)
\rightarrow
(I_0,I_1,\ldots,I_T)
$$

#### Video2World

输入文本和视频片段，生成视频延续或变换结果：

$$
(\text{text},V_{1:k})
\rightarrow
V_{1:T}
$$

Image2World 和 Video2World 使用 frame replacement：条件帧在生成序列中持续替换/固定，以增强早期帧一致性并将视觉信息传播到后续帧。

### 4.8 渐进式预训练

论文按分辨率和任务难度逐步训练：

| 阶段 | 任务 | 分辨率 | 帧数 |
|---|---|---|---:|
| 1 | Text2Image | 256p，320×192 | 1 |
| 2 | Text2Image + Video2World | 256p | 1 / 93 |
| 3 | Text2Image + Video2World | 480p，832×480 | 1 / 93 |
| 4 | Text2Image + Video2World + Text2World | 720p，1280×704 | 1 / 93 / 93 |

Video2World 阶段随机使用 1 或 5 条条件帧，生成其余帧。通过 mask token 标识哪些 latent 是条件输入，损失只作用于需要生成的帧。

### 4.9 领域 SFT 与 Model Merging

论文把视频按领域分成：

- object permanence：1040 万；
- high motion：100 万；
- complex scenes：160 万；
- driving：310 万；
- robotic manipulation：73 万；
- 4K：38.8 万。

每个领域单独训练 SFT 模型，而不是直接混合所有数据。每个领域模型训练 30K iterations、batch size 256。

随后与 4K cooldown 模型进行合并，比较：

- model soup；
- TIES；
- DARE-Linear；
- DARE-TIES。

论文最终选择简单的 model soup 方案，因为在其搜索中性能稳定且实现简单。

### 4.10 RL 后训练

论文使用 VideoAlign VLM reward model，奖励包含：

- text alignment；
- motion quality；
- visual quality。

每个条件生成 8 个视频，使用类似 GRPO 的组内 reward normalization 计算 advantage。由于显存限制，整条 denoising trajectory 的概率被拆成多个步骤计算，梯度累积后更新模型。

同时加入原始 diffusion/flow loss 作为正则，减少 reward hacking。

论文报告 2B merged 模型从 RL 前到 RL 后的 Text2World 总奖励：

$$
1.23\rightarrow1.74
$$

Image2World 总奖励：

$$
0.24\rightarrow0.45
$$

### 4.11 Timestep Distillation

论文还使用 rCM 等方法进行时间步蒸馏，目标是将多步视频生成压缩为更少采样步，降低推理成本。此处要区分：

- Flow Matching：训练基础速度场；
- RL：优化人类/模型偏好的生成结果；
- Distillation：近似原模型的生成映射或分布，减少推理步数。

### 4.12 Cosmos-Transfer2.5

Cosmos-Transfer2.5 是基于 Predict2.5 的 ControlNet 风格控制模型，支持：

- edge；
- blur；
- depth；
- segmentation。

与 Transfer1 的主要结构变化：

- Transfer1：4 个 control blocks 集中放在主干开头；
- Transfer2.5：4 个 control blocks 均匀分布，每 7 个主干 block 插入一个。

该设计让空间条件在网络更深层逐步注入。

训练中使用：

- Video Depth Anything 生成约 1000 万视频深度；
- SAMv2 生成约 300 万视频分割；
- 约 1400 万视频生成 edge/blur 条件。

每个控制分支独立训练 100K iterations，有效 batch size 64。

### 4.13 论文官方代码核心结构

官方仓库已经公开源码、checkpoint 和 benchmark：

| 仓库/目录 | 作用 |
|---|---|
| [`cosmos-predict2.5`](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Predict2.5 推理、模型、训练、蒸馏和应用 |
| `cosmos_predict2/` | 主 Python package |
| `examples/` | Text2World、Video2World、机器人和自动驾驶示例 |
| `scripts/` | 下载、推理、训练和数据脚本 |
| `docs/` | 安装、推理、后训练和蒸馏文档 |
| `packages/` | 可安装的项目组件 |
| [`cosmos-transfer2.5`](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | Transfer2.5 控制条件模型 |

代码与论文的映射：

| 论文概念 | 官方代码/配置对应 |
|---|---|
| Text2World/Image2World/Video2World | Predict2.5 inference examples 与 model configs |
| Flow Matching | `cosmos_predict2` 内的 flow/rectified-flow sampler、scheduler 和 denoising path |
| Latent DiT | `cosmos_predict2` 的 transformer/model modules |
| WAN2.1 VAE | tokenizer/decoder pipeline 配置 |
| Cosmos-Reason1 | text conditioning / encoder configuration |
| Timestep distillation | distillation docs、DMD2/rCM 相关 scripts |
| Auto multiview | auto/multiview checkpoint 与 example |
| Action-conditioned | robot/action-cond checkpoint、inference 和 post-training guide |
| Cosmos-Transfer2.5 | 独立仓库中的 control branch 和 spatial condition pipeline |

仓库 README 当前提示 Predict2.5 已进入有限维护状态，后续重点转向 Cosmos 3；这不影响论文对应版本的可复现性，但意味着依赖、checkpoint 和文档可能不再持续更新。

## 五、核心创新点与传统方法对比

### 5.1 相比 Cosmos-Predict1

| 方面 | Predict1 | Predict2.5 |
|---|---|---|
| 训练参数化 | EDM | Flow Matching velocity prediction |
| 文本编码器 | T5 | Cosmos-Reason1 |
| 生成能力 | 多模型/多模式组合 | 单模型统一 Text2World、Image2World、Video2World |
| 位置编码 | 含 absolute embedding | 主要保留 3D RoPE |
| 数据策略 | 较基础 | 更强过滤、领域整理和后训练 |
| 后训练 | 领域优化 | SFT + model merging + RL |
| 开源应用 | 世界生成 | 机器人、自动驾驶、VLA、Transfer2.5 |

### 5.2 相比普通视频扩散模型

Cosmos-Predict2.5 的重点不是仅提升视觉质量，还强调：

- 物体持续存在性；
- 高运动场景；
- 复杂物理场景；
- 动作条件；
- 多视角一致性；
- 长时序误差累积；
- Physical AI 下游用途。

### 5.3 相比单纯 domain fine-tuning

论文不是训练一个统一混合数据模型，而是：

```text
通用预训练模型
    ↓
多个领域 SFT 模型
    ↓
模型合并
    ↓
RL alignment
```

这样可以利用领域特化数据，同时减少单一领域模型对通用能力的破坏。

### 5.4 Cosmos-Transfer2.5 的核心差异

相比普通 image/video-to-video translation，Transfer2.5 能显式接受空间控制输入，并将 simulator 输出转换为更真实的视觉世界：

$$
(\text{control map},\text{text})
\rightarrow
\text{realistic video}
$$

这对 Sim2Real、Real2Real 和自动驾驶场景条件生成有直接意义。

## 六、理论分析与关键假设

### 6.1 Flow Matching 理论角色

训练目标学习条件速度场：

$$
 u_\theta(x_t,t,c)\approx\epsilon-x
$$

理想情况下，沿该速度场积分可以把噪声分布输送到数据分布。论文主要强调工程效果，没有提供新的 FM 收敛定理或误差界。

### 6.2 时间采样的工程假设

论文假设高噪声区域训练不足会导致视频帧过渡不稳定，因此增加高噪声采样可改善 temporal consistency。这是经验性训练策略，不是普适定理。

### 6.3 去除 absolute positional embedding 的假设

论文假设 absolute position 对固定训练分辨率/长度形成限制，使用 relative 3D RoPE 更利于高分辨率、长视频泛化。该结论主要由实验和设计动机支持，论文没有给出形式化泛化证明。

### 6.4 视频世界模型的物理假设

视频真实感不等于真实物理规律。模型从视频数据学习外观、运动和条件关系，但并不自动保证：

- 质量守恒；
- 动力学约束；
- 真实碰撞；
- 控制动作对应的因果结果；
- 长期 rollout 稳定。

### 6.5 RL 的奖励代理问题

VideoAlign 奖励是视觉语言模型的代理目标：

$$
\text{reward}
=
\text{text alignment}
+\text{motion quality}
+\text{visual quality}
$$

它可能提升人类偏好的视频质量，但不等价于真实物理有效性或策略评估可靠性。

### 6.6 论文没有证明的内容

- 生成视频一定可作为真实机器人策略的可靠替代环境；
- PAI-Bench 分数与真实闭环驾驶/机器人成功率严格相关；
- RL 后的视频不存在 reward hacking；
- 更高视觉质量一定带来更好的世界状态预测；
- 长视频中的误差可以被当前方法完全抑制。

## 七、实验设计与结果分析

### 7.1 PAI-Bench Predict

论文在 PAI-Bench 上评估 Text2World 和 Image2World：

- Domain Score：七个 Physical AI domain 的 VQA 评估；
- Quality Score：改造自 VBench 的 8 个视频质量指标；
- Overall Score：Domain Score 与 Quality Score 的平均值。

### 7.2 PAI-Bench Text2World

| 模型 | Domain | Quality | Overall |
|---|---:|---:|---:|
| Cosmos-Predict2.5-2B pre-train | 0.782 | 0.720 | 0.751 |
| Cosmos-Predict2.5-2B post-train | 0.804 | 0.732 | 0.768 |
| Cosmos-Predict2.5-14B pre-train | 0.791 | 0.722 | 0.757 |
| Cosmos-Predict2.5-14B post-train | 0.803 | 0.732 | 0.768 |
| Wan2.2-27B-A14B | 0.810 | 0.728 | 0.769 |

2B 和 14B post-train 的 Overall 均为 0.768，接近 Wan2.2 27B-A14B 的 0.769；但 Domain/Quality 的构成不同，不能只根据 Overall 断言整体能力完全相同。

### 7.3 PAI-Bench Image2World

| 模型 | Domain | Quality | Overall |
|---|---:|---:|---:|
| Cosmos-Predict2.5-2B pre-train | 0.824 | 0.775 | 0.799 |
| Cosmos-Predict2.5-2B post-train | 0.840 | 0.779 | **0.810** |
| Cosmos-Predict2.5-14B pre-train | 0.835 | 0.777 | 0.806 |
| Cosmos-Predict2.5-14B post-train | 0.838 | 0.781 | **0.810** |
| Wan2.2-27B-A14B | 0.841 | 0.772 | 0.806 |

后训练主要改善 domain alignment 和 overall score。

### 7.4 人工评测

论文采用成对视频偏好比较：

- 2B post-trained 相比 Wan2.2 5B：偏好率 30.0% vs 26.2%；
- 2B 相比 Wan2.1 14B：33.0% vs 34.8%，基本接近；
- 14B 相比 Wan2.1 14B：48.6% vs 31.8%；
- 14B 相比 Wan2.2 27B-A14B：38.1% vs 35.9%。

这是人类偏好证据，不是物理正确性的直接证明。

### 7.5 Cosmos-Transfer2.5

PAIBench-Transfer 包含 600 个跨领域视频，评价控制遵循度和整体质量。论文报告 Transfer2.5-2B 相比 Transfer1-7B 在更小模型规模下取得更高结果：

| 模型 | Overall Quality |
|---|---:|
| Transfer1-7B Blur | 6.56 |
| Transfer1-7B Edge | 6.76 |
| Transfer1-7B Depth | 6.89 |
| Transfer1-7B Uniform | 9.24 |
| Transfer2.5-2B Blur | **9.75** |
| Transfer2.5-2B Edge | 8.73 |
| Transfer2.5-2B Depth | 8.85 |
| Transfer2.5-2B Segmentation | 8.81 |
| Transfer2.5-2B Uniform | **9.31** |

论文将收益归因于更强的 Predict2.5 base model 和更聚焦 Physical AI 的控制训练数据。

### 7.6 长视频误差累积

论文提出 RNDS 指标：

$$
RNDS[i]
=
\frac{DOVER[i]/DOVER_{GT}[i]}
{DOVER[1]/DOVER_{GT}[1]}
$$

其中 $i$ 是 autoregressive chunk index，$DOVER[i]$ 是生成 chunk 的质量分数，$DOVER_{GT}[i]$ 是对应 GT chunk 分数。该归一化用于观察视频越生成越长时质量如何下降。

论文在 30–120 秒视频上观察到 Transfer2.5 的 RNDS 随 chunk 下降更慢，说明长视频误差累积较少；但这是有限视频集合上的评估，不能推出任意长时域稳定性。

### 7.7 应用实验

论文还展示：

- 多视角自动驾驶世界生成；
- 相机姿态可控多视角生成；
- 机器人 Real2Real 数据增强；
- VLA 训练合成数据；
- action-conditioned world model；
- 策略评估和闭环仿真。

这些部分主要证明应用可行性和系统覆盖范围，论文并未对每类应用都提供同等规模的真实闭环指标。

### 7.8 实验可比性审计

需要注意：

1. 不同模型的训练数据、分辨率、采样步数和条件方式不完全一致；
2. PAI-Bench Overall 是复合指标，可能掩盖 Domain/Quality 的差异；
3. 人工偏好受提示集合、评审者和视频选择影响；
4. Transfer2.5 与 Transfer1 的收益同时受到 base model、数据和架构影响；
5. 世界模型应用展示不等于控制策略性能已被充分验证。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. 将 Text2World、Image2World 和 Video2World 统一到同一个 flow-based 视频世界模型。
2. 通过 Cosmos-Reason1 加强文本 grounding 和细粒度控制。
3. 通过领域独立 SFT、model merging 和 RL 兼顾通用与 Physical AI 专项能力。
4. 提供 Cosmos-Transfer2.5，将空间控制图转换为真实感视频。
5. 系统覆盖机器人、自动驾驶、VLA、Sim2Real 和 Real2Real 应用。
6. 开源代码、checkpoint 和 benchmark，降低 Physical AI 世界模型使用门槛。

### 8.2 论文/系统局限

- 视频真实感不能替代物理仿真器；
- 主要结果集中于视频质量、指令对齐和代理指标；
- 真实策略闭环收益仍需独立验证；
- 采样过程仍可能较慢，蒸馏会引入质量—速度折中；
- autoregressive 长视频存在误差累积；
- 训练数据规模和硬件成本极高；
- 数据许可、版权、隐私和模型使用限制需要部署方单独审查；
- 官方仓库当前已提示后续重点转向 Cosmos 3，Predict2.5 可能进入有限维护阶段。

### 8.3 分析者识别出的潜在问题

#### 问题一：视觉合理不等于物理合理

视频可能在人类观察下很真实，但违反碰撞、惯性、接触和遮挡约束。VBench/PAI-Bench 质量分数无法完全发现这些问题。

#### 问题二：世界模型存在因果不确定性

同一当前观测可能对应多个合理未来。单一生成或有限采样可能遗漏关键行为，尤其是行人、车辆和机器人接触行为。

#### 问题三：动作条件的真实性需要验证

action-conditioned video 生成如果只学习动作与视觉模式的相关性，可能并不具备“执行该动作会导致该结果”的可靠因果关系。

#### 问题四：RL 可能优化代理指标

VideoAlign 奖励包含语言对齐和视觉质量。模型可能学会生成更讨喜的画面，而不是更真实、更安全、更可用于策略评估的物理结果。

#### 问题五：Model merging 的选择偏差

论文生成 20 多个 merged models，并根据小规模手工挑战样例选择最佳模型，再做较大评测。该过程仍可能产生验证集过拟合或选择偏差。

#### 问题六：长时域指标对真实 rollout 的覆盖有限

RNDS 评价视频 chunk 质量下降，但未充分刻画对象 identity、动作因果、3D 几何和闭环策略影响。

#### 问题七：大规模数据的域覆盖问题

2 亿视频很大，但 Physical AI 关键能力可能集中在少数高质量领域数据。数据量本身不能保证罕见危险场景和真实机器人交互被充分覆盖。

## 九、通俗讲解

### 9.1 它是什么

可以把 Cosmos-Predict2.5 理解为一个“视频版的世界模拟器”：

```text
告诉它场景或给它一张/一段视频
    ↓
它预测接下来世界可能怎样变化
```

### 9.2 Flow Matching 怎么工作

训练时，模型学习从噪声逐渐流向真实视频：

```text
噪声 → 中间状态 → 视频
```

模型不是直接猜每个像素，而是预测当前状态应该往哪个方向移动：

```text
当前 noisy latent + 时间步 + 条件
                ↓
          预测移动方向/速度
```

### 9.3 三种模式

```text
Text2World：文字 → 视频
Image2World：文字 + 图片 → 未来视频
Video2World：文字 + 视频 → 延续/变换视频
```

### 9.4 为什么使用 Cosmos-Reason1

普通文本编码器只负责把文字变成 embedding；Cosmos-Reason1 更强调 Physical AI 场景理解，因此能更细致地理解：

- 物体；
- 动作；
- 空间关系；
- 物理场景描述。

这些表示通过 cross-attention 指导视频生成。

### 9.5 为什么要领域 SFT 和模型合并

通用视频模型可能不擅长机器人抓取或自动驾驶。论文分别训练：

```text
通用模型 → 驾驶模型
          → 机器人模型
          → 高运动模型
          → 复杂场景模型
```

再把这些模型合并，得到一个兼顾多个领域的模型。

### 9.6 Cosmos-Transfer2.5 是什么

如果 Predict2.5 是“根据文字/图片想象世界”，Transfer2.5 就是“给它一张结构控制图，让它生成真实视频”：

```text
深度图 / 分割图 / 边缘图 / 模糊视频
                    ↓
             Transfer2.5
                    ↓
              写实视频
```

这可以把仿真器中的结构化场景转换成更真实的视觉输入。

### 9.7 一句话理解

> Cosmos-Predict2.5 学习“世界接下来会怎样”，Cosmos-Transfer2.5 学习“如何根据结构控制把世界渲染得更真实”，二者共同服务机器人和自动驾驶的 Physical AI 训练与评估。

## 十、综合评价与后续研究方向

### 10.1 综合评价

本文的完整技术链为：

$$
\text{curated videos}
\rightarrow
\text{VAE latent}
\rightarrow
\text{Flow Matching DiT}
\rightarrow
\text{SFT/model merging/RL}
\rightarrow
\text{Physical AI world simulation}
$$

Cosmos-Predict2.5 的主要技术贡献在系统组合和工程闭环：

- Flow Matching 直接学习 latent 速度；
- WAN2.1 VAE 降低视频 token 成本；
- Cosmos-Reason1 提升文本条件表示；
- 移除 absolute position，增强分辨率/长度灵活性；
- 通过高噪声采样减少时序突变；
- 领域 SFT + model merging 兼顾专业能力和通用能力；
- RL 提升对齐与视频质量；
- distillation 降低推理成本。

Cosmos-Transfer2.5 则把同一世界生成能力扩展为控制条件转换：

$$
\text{edge/blur/depth/segmentation}
\rightarrow
\text{realistic world video}
$$

论文结果表明，模型在 PAI-Bench 和人工偏好评测上具备竞争力，Transfer2.5 也在更小模型规模下超过 Transfer1 的报告结果。

但更严谨的结论是：

> Cosmos-Predict2.5 是一个面向 Physical AI 的大规模视频世界生成基础模型，具备较强的视觉生成、条件对齐和领域适配能力；它是机器人/自动驾驶仿真的重要视觉组件，但尚不能仅凭当前指标被视为真实物理世界的完备替代品。

### 10.2 后续研究方向

1. **物理一致性训练：** 加入 3D 几何、深度、光流、碰撞和动力学约束。
2. **动作—结果因果建模：** 用真实 action trajectory 和 intervention 数据验证动作条件预测。
3. **多模态不确定性：** 对同一条件生成多个未来，并建模场景分布而非单一视频。
4. **长时域稳定 rollout：** 结合 memory、latent state correction 和对象级 tracking 降低误差累积。
5. **策略闭环评估：** 直接测试世界模型对 policy ranking、failure prediction 和 sim-to-real 的有效性。
6. **3D/多视角一致性：** 引入显式场景图、occupancy、点云或 Gaussian world representation。
7. **高效推理：** 进一步发展 timestep distillation、缓存、滑窗生成和低比特量化。
8. **领域自适应：** 针对具体车辆、机器人、相机布局和天气条件进行安全后训练。
9. **安全与数据治理：** 加强隐私、版权、训练数据溯源、模型滥用和生成内容检测。
10. **从 Cosmos-Predict2.5 到统一 Physical AI 模型：** 论文后续已在官方仓库提示 Cosmos 3，值得比较其是否进一步统一理解、预测、迁移和策略生成。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2511.00062](https://arxiv.org/abs/2511.00062)
- 论文 HTML：[https://arxiv.org/html/2511.00062v2](https://arxiv.org/html/2511.00062v2)
- 论文 PDF：[https://arxiv.org/pdf/2511.00062](https://arxiv.org/pdf/2511.00062)
- Predict2.5 官方代码：[https://github.com/nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- Transfer2.5 官方代码：[https://github.com/nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
