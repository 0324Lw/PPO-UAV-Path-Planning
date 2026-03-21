# 🚀 PPO-UAV-Path-Planning

> 基于 Proximal Policy Optimization (PPO) 深度强化学习的无人机复杂环境路径规划与动态避障算法。
> 本项目构建了一个高度自定义的 `gymnasium` 强化学习环境，支持超大规模地图、静态/动态混合障碍物、流体动力学引导以及全自动的训练数据可视化。

## ✨ 核心特性 (Features)

* 🌍 **超大规模复杂环境**：支持 20km x 20km 的大比例尺地图，并内置了带有随机相位的高复杂度平滑地形生成算法。
* 🛸 **混合动态/静态避障**：环境内部集成了静态禁飞区 (NFZ)、静态威胁源、动态禁飞区以及动态穿插威胁源的复杂场景，全方位考验智能体的避障能力。
* 🧠 **PPO 强化学习底座**：采用 MLP 结构的 Actor-Critic 网络，集成了广义优势估计 (GAE)、梯度截断 (Clip)、熵正则化以及动态学习率衰减等 PPO 核心机制。
* 📈 **全自动数据分析与可视化**：提供完备的 `plot_results` 接口。训练结束后自动保存 `training_data_log.csv`，并生成包含成功率、奖励值、损失函数、平均障碍物距离等所有指标的高清滑动平均双曲线图表。

## 🧠 强化学习环境设计 (Environment Design)

本项目针对无人机大尺度（宏观）环境下的连续控制任务进行建模。为了有效应对地图范围广、障碍物类型多（静态禁飞区、动态巡逻区、软性威胁源）的挑战，核心 MDP 设计如下：

### 1. 状态空间 (Observation Space)

为了避免维度爆炸并提升感知效率，智能体并未输入全局地图，而是提取了 **12 维连续感知特征向量 (Continuous 12D)**。系统在每步会动态扫描并筛选出距离无人机**最近的 5 个障碍物**（包含混合的动/静态禁飞区与威胁源）进行状态编码：

| 维度索引 | 物理含义 (Description) | 数据范围 (Range) |
| :---: | :--- | :---: |
| `0` | 智能体到目标点的归一化欧氏距离 | $[0, 1.0]$ |
| `1` | 智能体朝向目标点的相对航向角正弦值 $\sin(\theta_{goal})$ | $[-1.0, 1.0]$ |
| `2:12` | **最近 5 个障碍物的信息**：交替排列每个障碍物的归一化距离与相对航向角正弦值 | $[0, 1.0], [-1.0, 1.0]$ |

### 2. 动作空间 (Action Space)

考虑到大尺度巡航任务的物理特性，本项目对动作空间进行了降维处理，采用 **1 维连续动作空间 (Continuous 1D)**：

* **线速度**：在巡航任务中设为恒定值 (`self.cfg.UAV_SPEED`)，不由神经网络控制。
* **$a_0$ (角速度控制)**：网络仅输出偏航角速度（Yaw Rate），范围映射在 $[-1.0, 1.0]$ 之间，用于控制无人机的平滑转向。

### 3. 奖励函数设计 (Reward Function)

大尺度环境面临极其严重的“稀疏奖励”问题。本项目通过引入 PBRS (势能奖励塑形) 与多维度密集奖励组件，极大地加速了 PPO 算法的收敛。单步原始奖励由以下部分构成：

$$R_{step} = r_{fwd} + r_{dir} + r_{curve} + r_{threat} + r_{height} + r_{pbrs}$$

* 🟢 **趋近奖励 ($r_{fwd}$)**：基于当前步缩短的距离比例。向目标靠近给予正向线性奖励，若偏离目标则给予二次方惩罚（放大后退代价）。
* 🟢 **航向对齐 ($r_{dir}$)**：鼓励无人机机头对准目标点，偏差越小奖励越高。
* 🔴 **转向惩罚 ($r_{curve}$)**：对角速度的平方 ($\omega^2$) 施加惩罚，约束无人机产生过于剧烈的抖动，保证飞行轨迹的动力学平滑度。
* 🔴 **软威胁惩罚 ($r_{threat}$)**：当进入静/动态威胁源的辐射半径时，基于侵入深度的三次方施加非线性惩罚（越靠近中心惩罚极速放大）。
* 🟢 **高度增益 ($r_{height}$)**：结合 3D 地形，对合理的飞行高度 ($Z$ 轴) 给予引导奖励。

**🏆 终局事件与机制保障：**
* **硬性碰撞**：撞击任何静/动态禁飞区 (NFZ)，触发极刑惩罚并终止回合 (`Terminated=True`)。
* **成功到达**：距离目标点 $< 500\text{m

## 🛠️ 环境依赖 (Requirements)

本项目基于 Python 开发，核心依赖库如下：

* `torch` (PyTorch)
* `gymnasium` 
* `numpy`
* `pandas`
* `matplotlib`

**快速安装指南 (Conda)：**
```bash
conda create -n ppo_nav python=3.10 -y
conda activate ppo_nav
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install numpy pandas matplotlib gymnasium -c conda-forge -y
```

<img width="400" height="400" alt="env_1_3d_terrain" src="https://github.com/user-attachments/assets/c3f85b1a-571f-49d0-893e-54f1ef22cf33" />
<img width="400" height="400" alt="env_5_3d_terrain" src="https://github.com/user-attachments/assets/6ca0bce6-6d9e-4c86-8901-759b9a8c0488" />  

![env_1_dynamic_obstacles](https://github.com/user-attachments/assets/5cd2d4be-82a2-4f57-95f0-2163e83b21c0)
