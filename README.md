# 🚀 PPO-UAV-Path-Planning

> 基于 Proximal Policy Optimization (PPO) 深度强化学习的无人机复杂环境路径规划与动态避障算法。
> 本项目构建了一个高度自定义的 `gymnasium` 强化学习环境，支持超大规模地图、静态/动态混合障碍物、流体动力学引导以及全自动的训练数据可视化。

## ✨ 核心特性 (Features)

* 🌍 **超大规模复杂环境**：支持 20km x 20km 的大比例尺地图，并内置了带有随机相位的高复杂度平滑地形生成算法。
* 🛸 **混合动态/静态避障**：环境内部集成了静态禁飞区 (NFZ)、静态威胁源、动态禁飞区以及动态穿插威胁源的复杂场景，全方位考验智能体的避障能力。
* 🧠 **PPO 强化学习底座**：采用 MLP 结构的 Actor-Critic 网络，集成了广义优势估计 (GAE)、梯度截断 (Clip)、熵正则化以及动态学习率衰减等 PPO 核心机制。
* 📈 **全自动数据分析与可视化**：提供完备的 `plot_results` 接口。训练结束后自动保存 `training_data_log.csv`，并生成包含成功率、奖励值、损失函数、平均障碍物距离等所有指标的高清滑动平均双曲线图表。

## 🛠️ 环境依赖 (Requirements)

本项目基于 Python 开发，核心依赖库如下：

* `torch` (PyTorch)
* `gymnasium` (替代原有 Gym 的新一代强化学习 API)
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
