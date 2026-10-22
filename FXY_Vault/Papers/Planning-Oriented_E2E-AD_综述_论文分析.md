# Planning-Oriented End-to-End Autonomous Driving：论文分析

> 标题：*Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms*  
> 来源：[arXiv:2608.20111](https://arxiv.org/abs/2608.20111)  
> 版本：v1，2026 年 8 月 20 日提交  
> 作者：Yanchen Guan、Xingcheng Liu、Bin Rao、Chengyue Wang、Guofa Li、Yunjian Li、Lishengsa Yue、Zhiyong Cui、Chengzhong Xu、Zhenning Li。  
> 正式发表信息：论文原文未说明正式会议或期刊。  
> 论文类型：综述论文；不提出一个单独的新驾驶模型。

## 一、论文基础信息速览

| 项目 | 内容 |
|---|---|
| 研究方向 | 端到端自动驾驶、轨迹规划、世界模型、VLA、自动驾驶评测 |
| 核心问题 | 现代 E2E-AD 如何从直接控制走向规划导向，以及如何正确评估规划能力 |
| 综述主线 | 架构、输入/输出、监督信号、评测协议四轴 taxonomy |
| 覆盖架构 | 行为克隆、条件模仿、特权蒸馏、BEV/occupancy/vector planning、统一规划栈、世界模型、VLA |
| 规划输出 | 控制量、waypoints、轨迹、轨迹分布、场景 rollout、语言/动作 token |
| 评测协议 | open-loop、non-reactive log replay、reactive closed-loop、长尾、偏好感知 |
| 关键结论 | displacement-based open-loop 指标不能单独证明安全、可行或人类一致的驾驶 |
| 代码 | 论文为 survey，未提供单一官方实现 |

## 二、极简全文核心总结

本文将端到端自动驾驶重新组织为“规划导向”的研究范式：重点不再是模型是否去除中间结构，而是结构、监督与评测是否共同服务于安全、可行、符合路线的规划。论文沿输入表示、输出空间、监督信号和评测协议四轴梳理行为克隆、BEV/向量规划、世界模型和 VLA，并强调 open-loop 轨迹误差、闭环仿真、长尾测试和人类偏好指标之间不能互相替代。

## 三、研究背景与研究意义

### 3.1 从模块化自动驾驶到 E2E-AD

传统流程是：

```text
感知 → 预测 → 规划 → 控制
```

模块化系统接口清晰，但可能存在：

- 感知指标和规划需求不一致；
- 预测结果无法被规划器充分利用；
- 中间离散表示损失原始传感器信息；
- 上游误差在多个接口中传播。

E2E-AD 试图学习：

$$
\pi_\theta(o_t, e_t, m_t, r_t)
\rightarrow
\{u_t,\tau_t\}
$$

其中 $o_t$ 为传感器观测，$e_t$ 为 ego state，$m_t$ 为地图/场景信息，$r_t$ 为路线或导航意图，输出可以是控制量或规划轨迹。

### 3.2 “端到端”概念已经泛化

早期 E2E 主要指 camera-to-steering；现代系统常包含：

- BEV feature；
- vectorized scene tokens；
- occupancy；
- object queries；
- world-model latent；
- route encoding；
- language tokens；
- safety filter。

论文的立场是：保留中间结构并不一定违背端到端。更合理的判断标准是：这些结构、损失和规划头是否围绕最终驾驶行为联合优化。

### 3.3 为什么评测协议成为核心问题

open-loop 轨迹误差可以大规模计算，但通常无法回答：

- 模型偏离专家后能否恢复？
- 其他交通参与者会如何反应？
- 是否会碰撞、越界或违反交通规则？
- 是否舒适、路线一致并具有足够进度？

因此需要同时研究：

$$
\text{architecture}
+
\text{supervision}
+
\text{evaluation protocol}
$$

## 四、核心方法、模型、公式与流程

> 本文是综述论文，下面的“方法”指其规划导向 taxonomy 和分析框架，而非一个新的端到端驾驶网络。

### 4.1 综述工作流

![Survey search and screen workflow](https://arxiv.org/html/2608.20111v1/E2E2-workflow.png)

> **图 1：综述的检索与筛选流程。** 论文记录数据源、搜索词、引用追踪、筛选标准和最终分类过程。图片来源：[论文 Figure 1](https://arxiv.org/html/2608.20111v1/E2E2-workflow.png)。

![Planning-oriented E2E-AD timeline](https://arxiv.org/html/2608.20111v1/E2E2-timeline.png)

> **图 2：规划导向 E2E-AD 的演进路线。** 研究从直接控制模仿，逐步演化为结构化规划接口、后果感知模型、语言动作系统和更真实的评测协议。图片来源：[论文 Figure 2](https://arxiv.org/html/2608.20111v1/E2E2-timeline.png)。

### 4.2 四轴 taxonomy

![Planning-oriented E2E-AD taxonomy](https://arxiv.org/html/2608.20111v1/E2E2-structure.png)

> **图 3：规划导向 E2E-AD 的四轴分析视角。** 方法应同时从输入表示、规划输出、监督信号和评测协议理解，而不能只按 backbone 分类。图片来源：[论文 Figure 3](https://arxiv.org/html/2608.20111v1/E2E2-structure.png)。

#### 轴一：输入表示

- 前视相机；
- 多视角图像；
- LiDAR/radar fusion；
- BEV/occupancy；
- object/vector tokens；
- latent world state；
- VLM/VLA token space。

#### 轴二：输出空间

- steering/throttle；
- waypoints；
- 单条轨迹；
- 多模态轨迹分布；
- 未来场景 rollout；
- language-conditioned action tokens。

#### 轴三：监督信号

- 人类行为克隆；
- 扰动和数据聚合；
- privileged teacher；
- 规则或 RL expert；
- perception/prediction/occupancy auxiliary losses；
- world-model prediction；
- language QA；
- human/rater preference。

#### 轴四：评测协议

- open-loop log replay；
- non-reactive simulation；
- reactive closed-loop simulation；
- real-world closed-loop；
- long-tail testing；
- preference-aware evaluation。

### 4.3 低层控制、waypoint 与轨迹输出

直接输出控制量的优点是工程链路短、推理快；缺点是：

- 难以表达多模态未来；
- 难以检查碰撞、可行驶区域和舒适性；
- 不易区分策略规划和执行控制；
- open-loop 对控制量的比较受车辆状态影响很大。

轨迹和 waypoint 提供了更清晰的 planning interface：

$$
\hat\tau_{t:t+H}
=
\{(x_i,y_i,\theta_i)\}_{i=1}^{H}
$$

可以由 controller 跟踪，也可以进行：

- 碰撞检查；
- drivable-area 检查；
- route compliance；
- time-to-collision；
- comfort/acceleration 约束。

### 4.4 BEV、occupancy 与 vectorized representation

BEV 将多相机、地图、车道、物体和占用放进统一坐标系：

$$
\{I_t^k\}_{k=1}^{K}
\rightarrow
F_{BEV}
\rightarrow
\text{trajectory planner}
$$

vectorized representation 进一步将道路、车道、agent 和轨迹表示为结构化 token，而不是稠密网格。

其规划优势是：

- 空间结构清晰；
- 易于加入地图和路线；
- 易于做碰撞、越界和车道约束；
- 可以将 agent interaction 显式暴露给规划器。

风险是：

- BEV 计算开销大；
- 依赖检测、地图或 occupancy 标注；
- 中间表示正确不等于最终轨迹安全；
- 结构化监督可能造成标签偏差。

### 4.5 统一 perception-prediction-planning stack

UniAD 等系统将检测、跟踪、地图、预测和规划联合起来：

```text
camera/LiDAR
   ↓
BEV/vectorized representation
   ├─ detection/tracking
   ├─ mapping
   ├─ motion forecasting
   └─ planning
        ↓
     trajectory
```

优点是规划损失可以反向塑造上游表示，减少模块接口失配；风险是：

- 失败归因变难；
- 可能依赖 ego state shortcut；
- 联合优化可能牺牲某个子任务；
- 需要更严格的因果诊断和 ablation。

### 4.6 不确定性与生成式规划

驾驶不是单标签预测。相同场景可能存在多个合理行为：

- 等待；
- 缓慢探出；
- 直接通过；
- 换道或绕行。

确定性回归容易把多模态未来平均掉：

$$
\hat\tau=\mathbb{E}[\tau\mid o_t]
$$

生成式方法尝试预测分布：

$$
\hat\tau^{(1)},\ldots,\hat\tau^{(M)}
\sim
p_\theta(\tau\mid o_t,r_t,e_t)
$$

但“生成多条轨迹”本身不够，还需要对候选轨迹排序：

$$
\operatorname{score}(\tau)
=
\text{safety}
+
\text{progress}
+
\text{comfort}
+
\text{route consistency}
$$

世界模型可以为候选轨迹预测后果，VLA 可以提供语义条件和语言动作 grounding，但两者都必须通过闭环或干预实验验证。

### 4.7 World model 的规划角色

论文区分四类 world model 角色：

| 角色 | 作用 | 规划连接 |
|---|---|---|
| Representation world model | 学习时空特征 | 改善下游预测和规划 |
| Generative scene model | 预测未来图像/BEV | 生成未来候选状态 |
| Trajectory evaluator | 对候选轨迹做 imagined rollout | 按未来后果选轨迹 |
| Closed-loop simulator | 交互式生成环境反馈 | 支持策略评估 |

关键判断：视觉逼真不等于规划安全，未来预测准确也不自动等于动作后果准确。

### 4.8 VLM/VLA driving

VLM/VLA 在驾驶中的语言角色包括：

1. **Annotation：** 用语言描述场景或决策；
2. **Supervision：** QA、解释和 planning labels；
3. **Instruction：** 接受导航、风格或任务指令；
4. **Planner：** 生成高层意图或动作；
5. **Action generator：** 将语言/视觉 token 对齐到轨迹或 action token。

驾驶 VLA 的核心桥梁是：

$$
\text{semantic language space}
\leftrightarrow
\text{metric trajectory space}
$$

语言推理能力不能直接替代：

- 精确几何；
- 低延迟控制；
- 碰撞约束；
- 物理可行性；
- 实时安全监控。

## 五、核心创新点与传统方法对比

### 5.1 论文的主要贡献

1. **规划导向重新定义 E2E-AD：** 不以“是否存在中间结构”为标准，而以是否服务安全规划为标准。
2. **四轴 taxonomy：** 输入表示、输出空间、监督信号、评测协议。
3. **贯通架构与评测：** 强调不能脱离 benchmark protocol 解读架构收益。
4. **连接新范式：** 将 BEV/vector planning、world model、VLM/VLA 放进统一规划框架。
5. **强调评测错配：** open-loop、non-reactive、closed-loop、preference metrics 测量不同属性。
6. **提出复现清单：** 版本、传感器、输出、监督、数据、runtime safety、compute 和统计信息都应报告。

### 5.2 与早期 camera-to-control 的区别

| 方面 | 早期直接控制 | 规划导向 E2E-AD |
|---|---|---|
| 输入 | 前视相机 | 多视角、LiDAR、BEV、地图、语言、ego state |
| 输出 | steering/throttle | waypoint、trajectory、分布或 action token |
| 监督 | 单一专家控制 | BC、扰动、蒸馏、辅助任务、world model、偏好 |
| 可检查性 | 弱 | 可检查碰撞、路线、舒适和进度 |
| 多模态 | 通常缺失 | 概率/生成式规划显式建模 |
| 评测 | 路线完成或控制误差 | open/closed-loop、长尾、偏好和安全 |

### 5.3 论文真正反对的误区

论文并不认为：

```text
端到端 = 完全黑盒
模块化 = 一定更安全
```

更准确的区分是：

- end-to-end training：损失可以联合优化；
- structured representation：中间结构仍可存在；
- planning-oriented：最终目标是安全、可行、路线一致的计划；
- deployable safety：仍需 runtime assurance，不能只依赖神经网络。

## 六、理论分析与关键假设

### 6.1 轨迹误差不是安全性

开环 displacement error 可以写成：

$$
\mathcal{L}_{disp}
=
\frac{1}{H}
\sum_{h=1}^{H}
\|\hat\tau_h-\tau_h^{log}\|_2
$$

它衡量预测轨迹与日志轨迹相似程度，但不直接衡量：

- 轨迹是否可行驶；
- 是否碰撞；
- 是否违反交通规则；
- 是否能从自身错误中恢复；
- 是否具有人类偏好的驾驶风格。

### 6.2 Open-loop 与 closed-loop 的分布差异

Open-loop 评估使用日志状态分布：

$$
s_t\sim p_{log}(s_t),
\qquad
\hat a_t=\pi_\theta(s_t)
$$

闭环中，策略自己的动作决定下一状态：

$$
\hat s_{t+1}\sim p(s_{t+1}\mid \hat s_t,\hat a_t)
$$

当策略产生小偏差后：

$$
\hat s_t\not\sim p_{log}(s_t)
$$

这就是 learner-expert mismatch / compounding error 的来源。

### 6.3 世界模型的验证条件

世界模型若用于候选轨迹排序，需要验证：

$$
\text{predicted consequence quality}
\Rightarrow
\text{trajectory ranking quality}
$$

仅有未来图像质量或 latent prediction loss 不足以证明规划可靠，还需要：

- action-conditioned intervention；
- rare scenario calibration；
- plan ranking accuracy；
- closed-loop safety/progress correlation。

### 6.4 VLA 的 grounding 假设

VLA 需要把语言概念映射到车辆坐标和时间：

$$
\text{“礼让行人”}
\rightarrow
\text{减速、停车、恢复行驶的 metric trajectory}
$$

语言表达正确不代表动作坐标正确。需要 action-aligned language supervision 或 structured spatial/action tokens。

### 6.5 论文没有证明的内容

- 结构化 E2E 一定优于直接控制；
- open-loop 更高分一定代表 closed-loop 更安全；
- world model 生成更逼真一定提升规划；
- VLM/VLA 语言推理一定能改善驾驶；
- runtime safety wrapper 可以提供形式化安全保证；
- 某一个 benchmark 能代表真实道路综合能力。

## 七、实验设计与结果分析

### 7.1 综述论文的证据类型

本文不是统一实验论文，不重新训练或评测所有方法。其证据来自：

- 代表性方法和 benchmark 的文献整理；
- 论文中的架构、监督、输出和评测对照；
- 对 open-loop/closed-loop/long-tail/preference protocol 的方法论分析；
- 对公开资源和复现状态的归纳。

因此表格中的方法不能直接视为在同一数据、硬件、传感器、控制器和指标下公平重跑的排名。

### 7.2 主要 benchmark 类型

| 类型 | 代表资源 | 能测什么 | 不能单独证明什么 |
|---|---|---|---|
| Open-loop log replay | nuScenes 等 | 日志模仿、轨迹误差 | 恢复、交互、长期安全 |
| Non-reactive simulation | NAVSIM | 真实日志背景下的规划 proxy | 背景 agent 反应 |
| Reactive closed-loop | nuPlan、Bench2Drive | 交互、恢复、路线完成 | 真实世界迁移 |
| Long-tail | WOD-E2E、Fail2Drive | 罕见事件和分布偏移 | 全覆盖真实危险事件 |
| Preference-aware | WOD-E2E 等 | 人类偏好、驾驶风格 | 完整物理安全保证 |

### 7.3 应报告的指标族

#### 几何/轨迹指标

- ADE/FDE；
- displacement error；
- minADE/minFDE；
- collision proxy；
- trajectory feasibility。

#### 闭环驾驶指标

- route completion/progress；
- collision rate；
- off-road rate；
- traffic infraction；
- time-to-collision；
- comfort/jerk；
- stuck/deadlock rate。

#### 长尾和泛化指标

- rare-event success；
- degradation curve；
- paired distribution shift；
- failure category；
- uncertainty calibration。

#### 偏好和人类对齐

- rater preference；
- comfort/style score；
- explanation/action faithfulness；
- safety-progress trade-off。

### 7.4 为什么 open-loop 与 closed-loop 会产生排名反转

![Open-loop versus closed-loop](https://arxiv.org/html/2608.20111v1/E2E2-openloop.png)

> **图 4：open-loop 与 closed-loop 评估的差异。** open-loop 在日志状态分布上测相似性；closed-loop 测策略自己诱导出的状态分布和恢复能力。图片来源：[论文 Figure 4](https://arxiv.org/html/2608.20111v1/E2E2-openloop.png)。

一种方法可能在日志轨迹上非常接近，但：

- 遇到自身偏差无法恢复；
- 对其他 agent 的反应错误；
- 过度依赖 ego state；
- 轨迹看似接近但舒适性/规则性差。

另一种方法可能与单条专家轨迹不完全相同，却能安全绕行、保持路线并在闭环中更稳。

### 7.5 复现清单

论文建议至少报告：

1. benchmark/simulator/metric 版本；
2. 传感器数量、分辨率、历史帧和地图；
3. 输出类型和 controller；
4. 训练信号；
5. 数据清洗、扰动和长尾采样；
6. safety filter/fallback/OOD monitor；
7. GPU、参数量、延迟、采样次数；
8. seed、置信区间、路线方差和失败类别。

### 7.6 证据边界

论文支持：

- E2E-AD 已从 direct control 演进到 planning-oriented structured systems；
- 轨迹输出比直接控制更易检查和评测；
- BEV/vector/occupancy 有助于几何约束和规划接口；
- world model/VLA 是重要新方向，但验证标准仍不充分；
- benchmark mismatch 是影响结论解释的核心问题。

论文不支持：

- 对所有方法给出统一绝对排名；
- 证明某种架构在真实道路上一定最安全；
- 证明 open-loop 指标毫无价值；
- 证明 VLA 或 world model 已解决闭环长尾驾驶。

## 八、学术价值、局限性与潜在漏洞

### 8.1 学术价值

1. **把规划作为 E2E-AD 的组织终点：** 避免只按 backbone 或是否有中间模块分类。
2. **把评测协议提升到方法论核心：** 同时分析架构和证据能否支持论文结论。
3. **连接传统与新范式：** 从 BC/CIL 到 BEV、world model、VLA 和 preference learning。
4. **强调结构不等于非端到端：** 结构化中间表示可服务安全和可审计性。
5. **提出可执行的复现规范：** 便于后续 benchmark 对齐和公平比较。

### 8.2 论文自述或明确承认的限制

- 综述偏向公开且信息充分的学术/工业工作；
- 专有 L4 系统的安全验证和运营基础设施通常不可见；
- 2025–2026 年新论文独立复现和长期影响尚未确定；
- 不同 benchmark、传感器和 controller 之间难以建立普适排名；
- 长尾真实事件覆盖仍不完整。

### 8.3 分析者识别出的潜在问题

#### 问题一：taxonomy 很全面，但类别存在重叠

同一系统可能同时属于 BEV planner、world model、VLA、probabilistic planner 和 unified stack。四轴分类解决了单一标签不足，但不会完全消除边界模糊。

#### 问题二：综述中的横向比较受文献选择偏差影响

公开论文更容易被纳入，工业系统和负结果不易公开，导致研究图景可能高估公开方法的成熟度。

#### 问题三：benchmark mismatch 仍缺少统一统计模型

论文指出不同 benchmark 测量不同属性，但尚未给出跨 benchmark 的统一相关性模型或因果分析框架。

#### 问题四：偏好指标与安全指标可能冲突

人类可能偏好平顺、自然、积极进取的行为，但安全系统可能需要保守等待。单一 preference score 不能替代安全约束。

#### 问题五：runtime shield 可能掩盖 planner 弱点

如果安全过滤器频繁接管，最终闭环成绩可能不错，但 learned planner 本身并未真正解决场景理解或规划问题。因此必须报告干预频率、触发条件和接管后的进度/舒适性。

#### 问题六：世界模型的“逼真”缺少干预验证

视频质量或未来状态误差不一定与候选轨迹排序相关。需要 action-conditioned counterfactual 和闭环 consequence evaluation。

## 九、通俗讲解

### 9.1 早期端到端驾驶

早期模型像这样：

```text
摄像头图片 → 神经网络 → 方向盘/油门
```

优点是简单，缺点是很难知道：

- 为什么这样打方向；
- 是否会撞车；
- 能不能恢复；
- 未来几秒会怎样。

### 9.2 现代 planning-oriented E2E

现代模型通常变成：

```text
相机/LiDAR/地图/路线
          ↓
BEV、occupancy、vector、latent state
          ↓
未来轨迹或多条候选轨迹
          ↓
碰撞/道路/舒适/路线检查
          ↓
控制器执行
```

它仍然可以端到端训练，但保留了对规划有用的结构。

### 9.3 为什么轨迹比直接方向盘更适合规划

方向盘值只告诉你“现在怎么转”；轨迹可以告诉你：

```text
未来 1 秒到哪里
未来 2 秒到哪里
是否越过道路边界
是否会接近其他车辆
```

因此轨迹更容易检查、解释和优化。

### 9.4 四种评测的区别

```text
Open-loop：看得像不像专家日志
Non-reactive：背景场景不反应，测规划 proxy
Closed-loop：车辆自己开，环境会反馈
Long-tail/preference：测罕见场景和人类驾驶偏好
```

一个模型在第一项好，不代表后面三项也好。

### 9.5 World model 是什么

World model 像一个“驾驶后果模拟器”：

```text
当前场景 + 候选轨迹
          ↓
预测未来会发生什么
          ↓
选择更安全、更有进度的轨迹
```

但如果模拟器没学会行人突然横穿，或者不能正确模拟车辆对自车的反应，就可能给规划器错误建议。

### 9.6 VLA 在驾驶中的作用

VLA 可以理解类似：

```text
“前方行人较多，谨慎减速并保持车道”
```

但真正落地还要把这句话转换成精确的车辆轨迹、时间和控制量。语言理解正确，不等于动作坐标正确。

### 9.7 一句话理解

> 现代端到端自动驾驶的重点，已经从“图像直接回归方向盘”转向“用学习到的结构、轨迹、世界模型和语言信息生成可检查、可评测、可执行的安全驾驶计划”。

## 十、综合评价与后续研究方向

### 10.1 综合评价

本文的核心逻辑为：

$$
\text{input representation}
+
\text{planning output}
+
\text{supervision}
+
\text{evaluation protocol}
\rightarrow
\text{claim validity}
$$

论文最重要的判断是：

> 评估 E2E-AD 时，不应问模型看起来有多“端到端”，而应问其表示、监督和评测是否共同支持安全、可行、符合路线且可复现的规划。

该综述的价值主要在方法论层面：

- 统一梳理 direct control、waypoint、trajectory、probabilistic planning；
- 将 BEV/vector/occupancy 与规划可检查性联系起来；
- 将 world model 从“生成逼真”转向“能否正确评价候选轨迹后果”；
- 将 VLM/VLA 从“能否回答问题”转向“能否改善 metric action”；
- 将 open-loop 和 closed-loop 的排名差异作为核心科学问题；
- 提出 benchmark 版本、控制器、runtime safety 和计算预算的复现规范。

更谨慎的结论是：

> 规划导向 E2E-AD 已从单纯行为克隆发展为融合结构化几何、预测、轨迹生成、世界模型和语言动作对齐的系统范式，但安全性、长尾鲁棒性、世界模型可靠性、语言—动作 grounding 和跨 benchmark 可比性仍未解决。

### 10.2 后续研究方向

1. **统一 benchmark：** 建立同时覆盖 open-loop、non-reactive、reactive closed-loop、长尾和偏好的成套协议。
2. **跨 benchmark 相关性：** 研究哪些指标真正能预测真实闭环安全和路线完成。
3. **不确定性规划：** 输出多模态轨迹、风险置信度和可校准的拒绝/交接机制。
4. **因果和反事实评测：** 对 ego action、交通参与者和路线意图进行干预，验证策略后果。
5. **世界模型校准：** 测试 action-conditioned future、rare-case prediction 和 trajectory ranking，而不是只测视频质量。
6. **语言—动作 grounding：** 将高层语义转为可验证的空间、时间和轨迹约束。
7. **runtime assurance：** 研究 safety envelope、fallback、control barrier function、OOD monitor 与 learned planner 的边界。
8. **learner-aware expert：** 设计学生可观测、可模仿且包含恢复行为的专家数据。
9. **部署约束：** 把 latency、功耗、显存、模型规模和在线采样次数纳入主结果。
10. **公开基线和可复现 stack：** 固定 simulator、route、metric、seed、controller 和评测容器，减少协议漂移。

## 参考链接

- 论文摘要：[https://arxiv.org/abs/2608.20111](https://arxiv.org/abs/2608.20111)
- 论文 HTML：[https://arxiv.org/html/2608.20111v1](https://arxiv.org/html/2608.20111v1)
- 论文 PDF：[https://arxiv.org/pdf/2608.20111](https://arxiv.org/pdf/2608.20111)
