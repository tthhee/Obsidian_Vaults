# A Survey on Vision-Language-Action Models for Embodied AI：论文分析

> 标题：*A Survey on Vision-Language-Action Models for Embodied AI*  
> 来源：[arXiv:2405.14093](https://arxiv.org/abs/2405.14093)  
> 当前分析版本：v8，2026 年 5 月 1 日修订；首次提交于 2024 年 5 月 23 日  
> 作者：Yueen Ma、Zixing Song、Yuzheng Zhuang、Jianye Hao、Irwin King  
> 正式发表：*IEEE Transactions on Neural Networks and Learning Systems*，Early Access，2026；DOI：[10.1109/TNNLS.2025.3650584](https://doi.org/10.1109/TNNLS.2025.3650584)。  
> 关联资源：[Awesome-VLA](https://github.com/yueen-ma/Awesome-VLA)

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 论文类型 | 综述论文、领域 taxonomy、资源与挑战总结 |
| 研究领域 | Embodied AI、机器人、VLM、VLA |
| 核心对象 | Vision-Language-Action（VLA）模型 |
| 目标 | 总结视觉、语言到机器人动作的建模路线 |
| 核心分类 | VLA 组件、低层控制策略、高层任务规划器 |
| 覆盖资源 | 模型、数据集、仿真器、机器人 benchmark、EQA benchmark |
| 低层输出 | 关节/末端执行器/离散 token/轨迹等动作 |
| 高层输出 | 可执行子任务序列 |
| 重点问题 | 泛化、安全、多模态、长时域、实时性、多智能体 |

## 二、极简全文核心总结

本文系统综述 Embodied AI 中的 Vision-Language-Action 模型，将研究划分为三条主线：支撑 VLA 的视觉、语言、动力学和世界模型组件；直接预测低层动作的控制策略；把长时域指令分解为子任务的高层规划器。论文还整理数据集、仿真器和 benchmark，并从安全、泛化、多模态、实时响应和长时域执行等方面总结当前瓶颈与未来方向。

## 三、研究背景与研究意义

### 3.1 Embodied AI 的基本闭环

机器人需要在物理世界中完成：

$$
\text{观测}
\rightarrow
\text{理解}
\rightarrow
\text{决策}
\rightarrow
\text{动作}
\rightarrow
\text{环境反馈}
$$

传统机器人策略往往针对单一任务、固定环境和固定 embodiment 设计。VLA 的目标是将：

- vision：理解当前环境；
- language：理解用户指令；
- action：输出可执行机器人动作；

统一到一个模型或层级系统中。

### 3.2 为什么需要 VLA

普通 VLM 可以回答“图中有什么”，但不能直接控制机器人；普通控制策略能够执行动作，但通常不理解自然语言和复杂视觉语义。VLA 试图学习：

$$
\pi_\theta(a_t\mid p,s_{\le t},a_{<t})
$$

其中：

- $p$：语言 prompt；
- $s_{\le t}$：当前及历史状态/视觉观测；
- $a_{<t}$：历史动作；
- $a_t$：当前动作。

### 3.3 综述的组织逻辑

论文不是按单一网络结构分类，而是按 VLA 系统中的功能层次组织：

```text
VLA components
    ↓
low-level control policies
    ↓
high-level task planners
```

这一区分很重要：

- 低层策略回答“现在具体怎么动”；
- 高层规划器回答“复杂任务应该拆成哪些步骤”。

## 四、核心方法、模型、公式与流程

> 本文是 survey，不提出一个单独的 VLA 网络。这里的“核心方法”是论文建立的 taxonomy，以及它用来分析不同 VLA 的统一抽象。

### 4.1 VLA 总体架构

![VLA general architecture](https://arxiv.org/html/2405.14093v8/components.svg)

> **图 1：VLA 通用架构。** 展示视觉、语言、状态输入经过多模态表示后，通过动作头输出机器人动作。图片来源：[论文 Figure 1](https://arxiv.org/html/2405.14093v8/components.svg)。

![Embodied AI concepts and timeline](https://arxiv.org/html/2405.14093v8/venn.png)

![Evolution timeline](https://arxiv.org/html/2405.14093v8/timeline.png)

> **图 2：Embodied AI 概念关系与技术演化时间线。** 图片来源：[论文 Figure 2](https://arxiv.org/html/2405.14093v8/venn.png)。

论文中的一般 VLA 体系可抽象为：

```text
视觉观测 / 语言指令 / 历史状态
              ↓
      vision encoder / VLM
              ↓
  cross-modal representation / reasoning
              ↓
        action decoder/head
              ↓
       low-level robot actions
              ↓
      environment feedback
```

对于长时域任务，可进一步加入高层规划器：

```text
用户长指令
    ↓
高层 task planner
    ↓
子任务 p1, p2, ..., pN
    ↓
低层 VLA policy
    ↓
机器人执行与重新规划
```

### 4.2 VLA 的统一数学定义

![Hierarchical robot policy](https://arxiv.org/html/2405.14093v8/illustration.png)

> **图 4：层级机器人策略。** 高层任务规划器先将用户指令分解为子任务，再由低层控制策略逐步执行。图片来源：[论文 Figure 4](https://arxiv.org/html/2405.14093v8/illustration.png)。

低层 VLA 控制策略：

$$
\hat a_t
\sim
\pi_\theta
\left(
 a_t\mid p,s_{\le t},a_{<t}
\right)
$$

高层 task planner：

$$
\hat{\mathbf p}
\sim
\pi_\phi
\left(
\mathbf p\mid \ell,s_t
\right)
$$

其中：

- $\ell$：长时域用户指令；
- $\mathbf p=[p_1,p_2,\ldots,p_N]$：子任务计划；
- $\pi_\phi$：高层规划器；
- $\pi_\theta$：执行单个子任务的低层策略。

层级系统的因果链为：

$$
\ell
\rightarrow
[p_1,\ldots,p_N]
\rightarrow
[a_1,a_2,\ldots]
\rightarrow
\text{environment state}
$$

### 4.3 第一条主线：VLA 的组成组件

#### 4.3.1 强化学习

RL 为 Embodied AI 提供状态—动作—奖励建模基础。论文回顾：

- DQN：从像素直接学习策略；
- Decision Transformer：将 RL trajectory 作为序列建模；
- Trajectory Transformer：建模状态、动作和奖励序列；
- Gato：多模态、多任务、多 embodiment；
- RLHF：用人类偏好改善机器人策略；
- LLM 生成 reward：让语言模型辅助设计奖励。

RL 的优势是可以通过环境反馈优化，但代价是：

- 真实机器人试错昂贵；
- 奖励设计困难；
- 安全约束复杂；
- 长时域 credit assignment 困难。

#### 4.3.2 预训练视觉表示

视觉 encoder 决定模型能否理解：

- 物体类别；
- 位置；
- affordance；
- 视觉变化；
- 操作相关结构。

论文整理的典型预训练路线：

| 表示 | 核心目标 | 重点能力 |
|---|---|---|
| CLIP | image-text contrastive learning | 视觉语义对齐 |
| R3M | temporal contrastive + video-language alignment | 时序和语言 |
| MVP | masked autoencoding | 视觉结构重建 |
| VIP | video temporal representation | 轨迹时序 |
| VC-1 等 | MAE/contrastive 组合 | 机器人视觉迁移 |

CLIP 的典型对比损失为：

$$
\mathcal{L}_{CLIP}
=
\sum_{i=1}^{N}
-
\log
\frac{
\exp(\mathcal{S}(x_i,y_i))
}
{
\sum_{j=1}^{N}\exp(\mathcal{S}(x_i,y_j))
}
$$

其中 $x_i$ 是图像表示，$y_i$ 是匹配文本表示，$\mathcal{S}$ 是相似度。

#### 4.3.3 视频表示

单帧图像不能充分表达机器人动作和环境变化。视频预训练可以学习：

- 时间邻近关系；
- 动作前后状态；
- 物体运动；
- 任务过程；
- 接触和变化。

#### 4.3.4 动力学学习

Dynamics model 学习：

$$
 s_{t+1}=f(s_t,a_t)
$$

或逆动力学：

$$
 a_t=f^{-1}(s_t,s_{t+1})
$$

动力学模型可以用于：

- 预测未来状态；
- 规划动作；
- 从视频反推动作；
- 在 world model 中进行 imagination rollout。

#### 4.3.5 World Model 与 LLM-induced World Model

World model 通过内部状态模拟环境变化。LLM 既可以：

- 作为高层规划器；
- 生成动作/技能序列；
- 通过文本反馈进行反思；
- 作为世界知识先验。

论文同时区分视觉 world model：直接预测未来视觉状态，以及语言/LLM world model：用符号或语言描述未来状态。

#### 4.3.6 Reasoning 与 Policy Steering

Reasoning 用于：

- 分析当前场景；
- 推断对象关系；
- 解释任务目标；
- 决定下一步技能。

Policy steering 则通过语言、视觉 prompt、目标点或中间表示引导低层策略，减少端到端策略的搜索空间。

### 4.4 第二条主线：低层控制策略

低层策略直接输出动作 primitive，常见输出形式包括：

- 连续控制量；
- 离散动作 token；
- 末端执行器位姿；
- 关节位置/速度；
- pick-place pose 对；
- 轨迹或 waypoint；
- 视觉视频后经 inverse dynamics 提取动作。

#### 4.4.1 非 Transformer 策略

![Representative VLA architectures](https://arxiv.org/html/2405.14093v8/architecture.png)

![Cross-attention architecture](https://arxiv.org/html/2405.14093v8/x1.png)

![Concatenation architecture](https://arxiv.org/html/2405.14093v8/x2.png)

![Tool-use architecture](https://arxiv.org/html/2405.14093v8/x3.png)

> **图 5：代表性 VLA 连接结构。** 论文比较 FiLM、cross-attention、concatenation、tool use 等多模态融合方式。图片来源：[论文 Figure 5](https://arxiv.org/html/2405.14093v8/architecture.png)。

论文回顾：

- CLIPort：CLIP 语义流 + Transporter 空间流；
- BC-Z：语言/演示条件 + FiLM；
- MCIL：图像目标/语言目标统一模仿学习；
- HULC：层级行为、离散 latent plan；
- UniPi：先生成条件视频，再通过 inverse dynamics 取动作。

#### 4.4.2 Transformer 控制策略

Transformer 适合把视觉、语言、状态和动作组织成序列：

$$
[\text{vision tokens};\text{text tokens};\text{state tokens};\text{action tokens}]
\rightarrow
\text{Transformer}
\rightarrow
\hat a_t
$$

优点：

- 支持长上下文；
- 易于多模态 token 融合；
- 可以扩展到多任务和多 embodiment。

代价：

- 计算量随序列长度增加；
- 需要大量多模态对齐数据；
- action tokenization 可能损失连续精度。

#### 4.4.3 3D Vision 控制策略

3D 视觉提供：

- 深度；
- 点云；
- 体素；
- 物体空间关系；
- 视角变化下的几何稳定性。

PerAct 等方法将 3D affordance 直接用于动作预测；这对抓取、插入、堆叠和空间导航尤其重要。

#### 4.4.4 Diffusion-based 控制策略

扩散策略将动作序列作为生成对象：

$$
\hat{\mathbf a}_{t:t+H}
\sim
p_\theta
\left(
\mathbf a_{t:t+H}
\mid
\text{vision},\text{language},\text{state}
\right)
$$

它天然可以表达多模态动作分布，例如同一目标有多种合理抓取轨迹。但推理需要多次 denoising，实时性和稳定性是主要问题。

#### 4.4.5 大型 VLA

大型 VLA 通常：

- 使用预训练 VLM/LLM；
- 加入动作 head 或 action token；
- 通过机器人数据 fine-tuning；
- 共享知识到多个任务和 embodiment。

论文将其价值归因于更强的语义泛化，但也指出模型规模增加并不自动解决真实机器人数据不足、动作安全和实时推理问题。

### 4.5 第三条主线：高层任务规划器

高层 planner 接收长指令 $\ell$，输出子任务序列：

$$
\ell
\rightarrow
[p_1,p_2,\dots,p_N]
$$

每个 $p_i$ 再交给低层 VLA 执行。

#### 4.5.1 Monolithic planners

单个 LLM/MLLM 负责视觉理解、推理和规划：

- PaLM-E：图像 + 高层指令 → 文本计划；
- EmbodiedGPT：融合 vision embedding 和 embodied planning；
- LEO、3D-LLM、ShapeLLM：加入点云或 3D 表示。

#### 4.5.3 Grounded planners

![Modular planner architectures](https://arxiv.org/html/2405.14093v8/planner.svg)

![Code-based planner](https://arxiv.org/html/2405.14093v8/x9.svg)

> **图 6：模块化任务规划器。** 左侧展示语言驱动规划，右侧展示代码驱动规划；两者都需要把高层模型输出连接到可执行模块。图片来源：[论文 Figure 6](https://arxiv.org/html/2405.14093v8/planner.svg)。

SayCan 将：

- LLM 的“say”：哪些技能符合语言目标；
- 机器人 affordance 的“can”：当前状态下哪些技能真正可执行；

结合起来选择技能：

$$
\text{skill score}
\approx
\text{language likelihood}
\times
\text{affordance/value}
$$

这比只让 LLM 生成文本计划更接近物理可执行性。

#### 4.5.3 Modular planners

模块化规划器将系统拆分为：

- 语言规划；
- 代码生成；
- 技能库；
- API/动作执行器；
- 状态检查和 replanning。

优点是可解释、易调试；缺点是接口多、组件之间可能不匹配，LLM 生成的子任务不一定能被低层策略执行。

### 4.6 数据集、仿真器与 benchmark 资源

论文整理了：

#### 真实机器人数据

- RT-1/RT-2 系列数据；
- Bridge；
- DROID；
- Open X-Embodiment；
- RoboMIND；
- AgiBot-Beta；
- GR00T 等。

#### 仿真环境

- RLBench；
- CALVIN；
- Meta-World；
- ManiSkill；
- Habitat；
- iGibson；
- VirtualHome；
- Isaac Sim 等。

#### 评测方向

- 低层操控成功率；
- 多任务泛化；
- 语言指令遵循；
- 3D 场景问答；
- 导航；
- 长时域任务规划；
- embodiment 和环境迁移。

## 五、核心创新点与传统方法对比

### 5.1 论文的主要贡献

1. **建立 VLA taxonomy：** 组件、低层控制、高层规划三大路线。
2. **统一低层和高层视角：** 将“动作生成”和“任务分解”放进同一个 Embodied AI 系统框架。
3. **系统整理资源：** 数据集、仿真器、真实机器人 benchmark 和 EQA benchmark。
4. **总结跨模态组件：** 视觉表示、视频、动力学、世界模型、推理和策略 steering。
5. **提出研究议程：** 安全、泛化、多模态、长时域、实时性和多智能体。

### 5.2 与传统机器人学习的区别

| 方向 | 传统方法 | VLA 方法 |
|---|---|---|
| 输入 | 状态/图像/固定任务 ID | 图像 + 自然语言 + 状态 |
| 任务规模 | 单任务或少量任务 | 多任务、开放指令 |
| 表示 | 任务专用 CNN/RL state | VLM/LLM 预训练表示 |
| 输出 | 控制量或轨迹 | action token、轨迹、技能或子任务 |
| 泛化方式 | 任务内插值 | 跨任务、跨环境、跨 embodiment |
| 规划 | 通常显式外部规划 | 端到端、层级或模块化规划 |

### 5.3 端到端与层级系统对比

| 方案 | 优势 | 局限 |
|---|---|---|
| Monolithic VLA | 结构简单、端到端 | 可解释性和可调试性弱 |
| Low-level VLA | 响应快、动作直接 | 难以处理长时域复杂任务 |
| Hierarchical VLA | 可分解复杂任务、可 replanning | 系统复杂、延迟高、接口易失败 |
| Modular planner | 可解释、易插拔 | 子任务与低层技能可能不匹配 |
| Code-based planner | 逻辑清晰、可执行接口明确 | 需要手工封装 API |

## 六、理论分析与关键假设

### 6.1 综述论文没有新的统一定理

本文的主要贡献是分类、整理和研究议程，而不是提出新的收敛定理或统一优化目标。因此下面的“理论分析”主要解释其统一抽象和隐含假设。

### 6.2 VLA 泛化的关键假设

VLA 希望从大规模预训练中获得：

$$
\text{pretrained semantics}
\rightarrow
\text{robot task understanding}
\rightarrow
\text{action generalization}
$$

但语义泛化不自动等于动作泛化。机器人动作还依赖：

- embodiment；
- 相机和传感器；
- 动力学；
- 抓取精度；
- 接触状态；
- 环境布局。

### 6.3 层级规划的误差传播

如果高层 planner 生成错误子任务：

$$
\hat p_i\neq p_i^*
\Rightarrow
\text{低层策略执行错误动作序列}
$$

即使低层策略本身很强，也可能因为目标分解错误而失败。反过来，低层 affordance 估计错误也会使高层 planner 选择不可执行技能。

### 6.4 数据和 embodiment 的分布偏移

真实机器人数据通常具有：

- 任务分布窄；
- embodiment 不统一；
- 语言标注不足；
- 相机位置不一致；
- 高质量失败轨迹稀少。

因此，VLA 的大模型规模和互联网预训练不能直接替代真实机器人交互数据。

### 6.5 安全假设

机器人动作影响真实物理环境，不能仅依赖语言模型概率：

$$
\text{high likelihood}
\not\Rightarrow
\text{safe action}
$$

需要加入：

- 碰撞约束；
- 风险评估；
- human-in-the-loop；
- 可执行性检查；
- evaluation without execution；
- 安全 guardrails。

### 6.6 论文没有证明的内容

- VLA 已经达到类似 LLM 的开放域泛化；
- 统一大模型一定优于专用策略；
- 语言推理一定改善低层动作精度；
- 仿真 benchmark 成功率可直接代表现实部署成功率；
- 层级系统的 replanning 一定能解决长时域任务；
- 单一 embedding space 足以统一视觉、语言、动作、触觉和音频。

## 七、实验设计与结果分析

### 7.1 综述论文的证据类型

本文不是统一 benchmark 实验论文，证据主要来自：

1. 对已有 VLA 方法的结构和训练目标整理；
2. 对不同数据集、仿真器和 benchmark 的归纳；
3. 对已有论文实验结果的横向总结；
4. 对领域挑战和未来方向的分析。

因此，不能把综述中的模型表格理解为在同一数据、同一硬件和同一指标下重新评测。

### 7.2 低层控制评价维度

常见指标包括：

- task success rate；
- language instruction following；
- unseen task generalization；
- unseen object/environment generalization；
- trajectory/action error；
- collision and safety；
- inference latency；
- cross-embodiment transfer。

### 7.3 高层规划评价维度

高层 planner 需要同时评估：

- 子任务分解正确性；
- 子任务可执行性；
- 前置条件满足度；
- 计划长度和效率；
- replanning 能力；
- 长时域最终成功率；
- 语言计划与实际动作的一致性。

只评估文本计划 BLEU/ROUGE 等语言指标是不够的，因为语言流畅不代表机器人能执行。

### 7.4 数据资源审计

综述的价值之一是揭示数据异质性：

| 数据类型 | 优点 | 主要问题 |
|---|---|---|
| 真实机器人演示 | 物理真实性高 | 成本高、规模有限 |
| 仿真轨迹 | 可大规模生成 | sim-to-real gap |
| 互联网视频 | 视觉和语义丰富 | 缺少动作、状态和 embodiment 信息 |
| 自采集数据 | 与目标机器人匹配 | 标注和质量控制成本高 |
| 人类语言数据 | 指令丰富 | 与低层动作对齐困难 |

### 7.5 对论文结论的正确解读

综述支持：

- VLA 已形成组件、策略、规划三条研究路线；
- 预训练视觉/语言表示可显著影响机器人策略；
- 3D、视频、动力学和 world model 是重要增强方向；
- 长时域执行更适合层级或模块化系统；
- 安全、实时和跨 embodiment 泛化仍未解决。

综述不支持：

- 某个单一 VLA 架构在所有任务上最优；
- VLA 已经实现开放域通用机器人控制；
- 统一模型一定优于层级模型；
- 仅靠扩大 LLM/VLM 参数即可解决真实世界问题。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **时间节点重要：** 记录 VLA 从单一策略走向基础模型、层级规划和 world model 的阶段。
2. **分类清晰：** 组件、低层动作和高层任务规划的划分有助于定位研究问题。
3. **跨领域连接：** 将 CV、NLP、RL、机器人控制和 world model 放入同一框架。
4. **资源集中：** 为研究者提供数据集、仿真器、benchmark 和代表性方法入口。
5. **问题意识强：** 不仅总结成功案例，也明确讨论安全、泛化、实时性和多智能体问题。

### 8.2 综述局限

- VLA 发展迅速，版本 v8 仍可能遗漏最新模型或数据；
- 不同论文的命名和“VLA”边界并不完全统一；
- 表格中的方法、数据、指标和 embodiment 往往无法直接横向比较；
- 综述不提供一个统一可复现的实验协议；
- 真实机器人安全和长期部署证据相对有限；
- 主要关注视觉、语言和动作，对触觉、音频、力觉的覆盖较少；
- 资源链接和模型可用性会随时间变化。

### 8.3 分析者识别出的潜在问题

#### 问题一：VLA 定义边界模糊

有些模型显式输出 action，有些模型通过视频生成、逆动力学或外部 planner 间接得到动作。若不区分直接 action policy 与 policy-as-video，比较容易混淆模型能力。

#### 问题二：语言能力可能掩盖动作能力

大型 VLM 能够生成合理的文字计划，但计划可能不满足真实机器人的动力学和技能库约束。语言 benchmark 高分不能代替执行成功率。

#### 问题三：成功率不足以诊断失败

同样的任务失败可能来自：

- 看错物体；
- 语言理解错误；
- 子任务分解错误；
- affordance 错误；
- 控制不精确；
- 碰撞或动力学问题。

需要更细粒度的 benchmark 和 failure taxonomy。

#### 问题四：大规模预训练数据缺少动作因果

互联网图像/视频具有丰富语义，但通常没有 synchronized robot action、proprioception 和接触力信息。因此视觉语义迁移到动作仍是核心瓶颈。

#### 问题五：层级系统的接口脆弱

高层 planner 输出的子任务必须与低层 policy 的 skill vocabulary、前置条件和 action space 对齐。否则模块越多，失败点越多。

#### 问题六：仿真到现实的安全验证不足

仿真成功率高可能来自简化环境、已知对象和低噪声观测。真实部署还要面对遮挡、标定误差、延迟、磨损和非理想接触。

## 九、通俗讲解

### 9.1 VLA 是什么

可以把 VLA 想象成一个能看、能听懂、还能动手的机器人模型：

```text
看见环境 + 听懂指令
          ↓
      决定怎么做
          ↓
      输出机器人动作
```

### 9.2 它和聊天机器人有什么不同

聊天机器人只需要输出文字；机器人必须让输出在物理世界中成立：

```text
聊天机器人：解释如何拿杯子
VLA：真的把手移动到杯子旁边并抓起来
```

### 9.3 低层策略和高层规划器

任务：“把桌上的杯子放进柜子。”

高层规划器可能拆成：

```text
1. 找到杯子
2. 抓住杯子
3. 移动到柜子
4. 打开柜门
5. 放入杯子
```

低层策略负责每一步的具体动作：

```text
手臂向哪里移动？
夹爪什么时候闭合？
速度是多少？
```

### 9.4 为什么需要视觉和语言

语言告诉机器人目标：

```text
“拿起红色杯子”
```

视觉告诉机器人：

```text
红色杯子在哪？附近有没有障碍物？
```

动作模块把二者变成实际控制。

### 9.5 为什么 3D 和视频重要

单张图片只能告诉机器人“现在是什么样”；视频和 3D 可以告诉它：

- 物体如何运动；
- 手应该从哪个方向接近；
- 物体碰到后会怎样；
- 当前动作是否已经成功。

### 9.6 一句话理解

> VLA 是把“看懂世界、听懂指令、执行动作”连接起来的机器人模型；高层规划器负责拆任务，低层策略负责真正控制机械臂或移动机器人。

## 十、综合评价与后续研究方向

![VLA research timeline](https://arxiv.org/html/2405.14093v8/vla_timeline.svg)

![VLA research landscape](https://arxiv.org/html/2405.14093v8/vla_bubble_landscape.svg)

![VLA development trends](https://arxiv.org/html/2405.14093v8/vla_collaboration_heatmap.png)

> **图 9–11：VLA 研究时间线、研究版图和发展趋势。** 这些图展示 VLA 研究数量、机构分布、合作关系及模型演化，可用于理解该领域的发展脉络。图片来源：[论文 Appendix Figures 9–11](https://arxiv.org/html/2405.14093v8/vla_timeline.svg)。

### 10.1 综合评价

本文最有价值的不是给出一个新模型，而是提供一张 VLA 研究地图：

$$
\text{components}
\rightarrow
\text{low-level policies}
\rightarrow
\text{task planners}
\rightarrow
\text{embodied systems}
$$

其中：

- 组件研究提升视觉、语言、视频、动力学和世界建模能力；
- 低层控制策略将多模态输入映射到直接动作；
- 高层规划器将开放语言指令分解为可执行技能序列；
- 数据集、仿真器和 benchmark 决定模型能否被可靠比较。

论文最核心的判断是：VLA 的难点已经不再只是“让模型看懂图像并输出动作”，而是要同时解决：

- 跨任务和跨 embodiment 泛化；
- 真实物理世界安全；
- 长时域规划与执行；
- 实时响应；
- 多模态融合；
- 仿真到现实迁移；
- 可解释性与失败诊断。

更准确的结论是：

> VLA 正从语言条件机器人策略发展为融合视觉基础模型、语言推理、低层控制、层级规划和世界模型的具身基础系统，但其开放域泛化、物理安全和真实闭环可靠性仍未达到成熟通用机器人水平。

### 10.2 后续研究方向

1. **动作—视觉—语言统一表示：** 引入 proprioception、触觉、力觉、音频和 gaze。
2. **对象级和 3D VLA：** 使用点云、occupancy、场景图和 affordance 表示提升空间操作。
3. **视频/世界模型辅助策略：** 在执行前进行 imagined rollout、风险预测和候选动作评估。
4. **安全 VLA：** 加入 collision checking、风险敏感优化、human approval 和 evaluation without execution。
5. **长时域层级控制：** 统一 planner、skill library、低层策略和自动 replanning。
6. **跨 embodiment 学习：** 通过动作抽象、坐标标准化和 embodiment token 支持不同机器人。
7. **更细粒度 benchmark：** 同时评估理解、规划、动作精度、碰撞、延迟、能耗和失败类型。
8. **仿真—现实协同：** 结合真实演示、合成数据、world model 和在线安全数据聚合。
9. **实时推理：** 压缩 VLM、动作 chunking、缓存视觉 token 和高效 diffusion policy。
10. **多智能体具身系统：** 研究通信、协作、资源调度、冲突目标和群体安全。

## 参考链接

- arXiv：[https://arxiv.org/abs/2405.14093](https://arxiv.org/abs/2405.14093)
- 论文 HTML：[https://arxiv.org/html/2405.14093v8](https://arxiv.org/html/2405.14093v8)
- DOI：[https://doi.org/10.1109/TNNLS.2025.3650584](https://doi.org/10.1109/TNNLS.2025.3650584)
- Awesome-VLA：[https://github.com/yueen-ma/Awesome-VLA](https://github.com/yueen-ma/Awesome-VLA)
