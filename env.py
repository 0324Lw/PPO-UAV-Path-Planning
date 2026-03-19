import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces


class Config:
    """
    配置类：全局通用超参数。
    包含环境参数与强化学习训练参数的统一配置。
    """
    # ==========================================
    # 1. 环境与物理参数
    # ==========================================
    MAP_SIZE = 20000.0  # 地图扩大至 20km (根据起终点间距调整)
    DT = 1.0  # 仿真步长 (s)
    UAV_SPEED = 100.0  # 线速度 (m/s)
    MAX_YAW_RATE = np.pi / 4  # 最大偏航角速度 (rad/s)
    SAFE_RADIUS = 3000.0  # 起终点安全区半径 (m)

    # 障碍物数量配置
    NUM_STATIC_NFZ = 4  # 静态禁飞区数量
    NUM_STATIC_THREATS = 2  # 静态威胁源数量
    NUM_DYN_NFZ = 1  # 动态禁飞区数量
    NUM_DYN_THREATS = 1  # 动态威胁源数量

    # ==========================================
    # 2. PPO 强化学习训练超参数
    # ==========================================
    TOTAL_TIMESTEPS = 1000000  # 总训练步数
    NUM_STEPS = 2048  # 每个 PPO 更新周期的交互步数
    NUM_EPOCHS = 10  # PPO 优化迭代次数
    MINIBATCH_SIZE = 64  # 批次大小
    GAMMA = 0.99  # 折扣因子
    GAE_LAMBDA = 0.95  # GAE 优势估计平滑参数
    CLIP_COEF = 0.2  # PPO 截断系数
    ENT_COEF = 0.01  # 熵正则化系数
    VF_COEF = 0.5  # 价值损失系数
    MAX_GRAD_NORM = 0.5  # 梯度裁剪阈值
    LR = 3e-4  # 初始学习率

    # ==========================================
    # 3. 奖励函数系数
    # ==========================================
    COEFF_STEP = -0.05  # 步数惩罚 (鼓励尽快到达)
    COEFF_FORWARD = 0.5  # 前进奖励
    COEFF_BACKWARD = -2.0  # 后退平方惩罚
    COEFF_DIR = 0.05  # 终点方向对齐奖励
    COEFF_CURVE = -0.08  # 弯曲度惩罚 (惩罚剧烈转向)
    COEFF_HEIGHT = -0.05  # 高度惩罚
    COEFF_THREAT = -10.0  # 威胁源惩罚 (按三次衰减)
    COEFF_COLLISION = -150.0  # 碰撞硬惩罚
    COEFF_GOAL = 150.0  # 到达终点大额奖励

    PBRS_SCALE = 10.0  # PBRS 势能乘数


class UAVEnv(gym.Env):
    """
    无人机路径规划环境
    支持纯雷达输出与流体特征引导，内置了全套评估指标的回传接口。
    """

    def __init__(self, cfg=Config(), use_pbrs=False, use_fluid=False):
        super(UAVEnv, self).__init__()
        self.use_pbrs = use_pbrs
        self.use_fluid = use_fluid
        self.cfg = cfg

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 状态空间维度：纯雷达 12 维，开启流体后为 14 维
        obs_dim = 14 if self.use_fluid else 12
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 预定义动态变量
        self.start_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.step_count = 0
        self.path_history = []

        # 障碍物列表
        self.static_nfz = []
        self.static_threats = []
        self.dyn_obs = {}
        self.dyn_threat = {}

        # 地形相位
        self.terrain_phase_x = 0.0
        self.terrain_phase_y = 0.0

    def _generate_terrain_height(self, x, y):
        """带有随机相位的高复杂度平滑地形生成"""
        base_freq = 4.0
        norm_x = x / self.cfg.MAP_SIZE * base_freq * np.pi + self.terrain_phase_x
        norm_y = y / self.cfg.MAP_SIZE * base_freq * np.pi + self.terrain_phase_y

        wave1 = np.sin(norm_x) * np.cos(norm_y)
        wave2 = np.sin(2.0 * norm_x + norm_y) * 0.5
        wave3 = np.cos(3.0 * norm_x - 2.0 * norm_y) * 0.25
        wave4 = np.sin(5.0 * norm_x) * np.cos(5.0 * norm_y) * 0.125

        height = wave1 + wave2 + wave3 + wave4
        max_amplitude = 1.875
        height = (height + max_amplitude) / (2 * max_amplitude)
        return np.clip(height, 0.0, 1.0)

    def _generate_random_scenario(self):
        """核心：起终点 -> 静态障碍物(同类防重叠，异类可重叠) -> 多个动态障碍物"""
        # ==========================================
        # 1. 优先生成起终点 (间距 >= 20000m)
        # ==========================================
        margin = 2000.0
        while True:
            s_pos = np.random.uniform(margin, self.cfg.MAP_SIZE - margin, 2)
            g_pos = np.random.uniform(margin, self.cfg.MAP_SIZE - margin, 2)
            if np.linalg.norm(s_pos - g_pos) >= 20000.0:
                self.start_pos = s_pos
                self.goal_pos = g_pos
                break

        # ==========================================
        # 2. 九宫格分区法生成静态障碍物 (按同类互斥规则)
        # ==========================================
        grid_dim = 3
        cell_size = self.cfg.MAP_SIZE / grid_dim
        cells = [(i, j) for i in range(grid_dim) for j in range(grid_dim)]
        np.random.shuffle(cells)

        def get_valid_static_center(cell_idx, radius, same_type_obstacles):
            """在指定格子内寻找安全点，仅与 same_type_obstacles 进行重叠校验"""
            i, j = cell_idx
            # 尝试 100 次在分配的格子内寻找
            for _ in range(100):
                cx = np.random.uniform(i * cell_size + radius, (i + 1) * cell_size - radius)
                cy = np.random.uniform(j * cell_size + radius, (j + 1) * cell_size - radius)
                center = np.array([cx, cy])

                # 【严格校验 1】起终点绝对安全区：圆心距离 必须大于 规定安全半径 + 障碍物自身半径
                if np.linalg.norm(center - self.start_pos) <= self.cfg.SAFE_RADIUS + radius: continue
                if np.linalg.norm(center - self.goal_pos) <= self.cfg.SAFE_RADIUS + radius: continue

                # 【严格校验 2】仅校验同类型障碍物的重叠 (保留1500m最小通道)
                is_overlap = False
                for ex_center, ex_radius in same_type_obstacles:
                    if np.linalg.norm(center - ex_center) < (radius + ex_radius + 1500.0):
                        is_overlap = True
                        break
                if not is_overlap:
                    return center

            # 兜底机制：全局寻找
            for _ in range(100):
                cx = np.random.uniform(radius, self.cfg.MAP_SIZE - radius)
                cy = np.random.uniform(radius, self.cfg.MAP_SIZE - radius)
                center = np.array([cx, cy])

                if np.linalg.norm(center - self.start_pos) <= self.cfg.SAFE_RADIUS + radius: continue
                if np.linalg.norm(center - self.goal_pos) <= self.cfg.SAFE_RADIUS + radius: continue

                is_overlap = False
                for ex_center, ex_radius in same_type_obstacles:
                    if np.linalg.norm(center - ex_center) < (radius + ex_radius + 500.0):  # 兜底时放宽通道要求
                        is_overlap = True
                        break
                if not is_overlap: return center

            # 极限兜底：放在地图边缘
            return np.array([radius, radius])

        cell_ptr = 0

        # 生成静态禁飞区 (NFZ)
        self.static_nfz = []
        existing_nfz = []  # 专属 NFZ 检测列表
        for _ in range(self.cfg.NUM_STATIC_NFZ):
            r = np.random.uniform(2000.0, 2000.0)
            c = get_valid_static_center(cells[cell_ptr % len(cells)], r, existing_nfz)
            self.static_nfz.append([c[0], c[1], r])
            existing_nfz.append((c, r))
            cell_ptr += 1

        # 生成静态威胁源 (Threats) - 互不影响 NFZ
        self.static_threats = []
        existing_threats = []  # 专属 Threats 检测列表
        for _ in range(self.cfg.NUM_STATIC_THREATS):
            r = np.random.uniform(3000.0, 3000.0)
            c = get_valid_static_center(cells[cell_ptr % len(cells)], r, existing_threats)
            self.static_threats.append([c[0], c[1], r])
            existing_threats.append((c, r))
            cell_ptr += 1

        # ==========================================
        # 3. 按照规定数量生成动态障碍物 (任意重叠)
        # ==========================================
        def create_intersecting_dyn_obj(u_range, r, speed):
            path_vec = self.goal_pos - self.start_pos
            path_len = np.linalg.norm(path_vec)
            path_dir = path_vec / path_len
            perp_dir = np.array([-path_dir[1], path_dir[0]])

            u = np.random.uniform(u_range[0], u_range[1])
            intersect_pt = self.start_pos + u * path_vec
            sweep_len = np.random.uniform(5000.0, 8000.0)

            start_p = intersect_pt + perp_dir * sweep_len
            end_p = intersect_pt - perp_dir * sweep_len
            start_p = np.clip(start_p, r, self.cfg.MAP_SIZE - r)
            end_p = np.clip(end_p, r, self.cfg.MAP_SIZE - r)
            # 在字典中直接维护当前坐标 'pos'
            return {'start': start_p, 'end': end_p, 'radius': r, 'speed': speed, 'pos': np.copy(start_p)}

        # 动态禁飞区列表
        self.dyn_obs_list = []
        for _ in range(self.cfg.NUM_DYN_NFZ):
            u_min, u_max = np.random.uniform(0.1, 0.4), np.random.uniform(0.5, 0.9)
            obj = create_intersecting_dyn_obj([u_min, u_max], np.random.uniform(1500.0, 2000.0),
                                              np.random.uniform(50.0, 80.0))
            self.dyn_obs_list.append(obj)

        # 动态威胁源列表
        self.dyn_threat_list = []
        for _ in range(self.cfg.NUM_DYN_THREATS):
            u_min, u_max = np.random.uniform(0.2, 0.5), np.random.uniform(0.6, 0.9)
            obj = create_intersecting_dyn_obj([u_min, u_max], np.random.uniform(3000.0, 3000.0),
                                              np.random.uniform(60.0, 90.0))
            self.dyn_threat_list.append(obj)

        # ==========================================
        # 4. 随机化地形相位
        # ==========================================
        self.terrain_phase_x = np.random.uniform(0, 2 * np.pi)
        self.terrain_phase_y = np.random.uniform(0, 2 * np.pi)

    def _calculate_fluid_dynamics(self, pos):
        """计算流体势场"""
        diff_goal = self.goal_pos - pos
        dist_goal = np.linalg.norm(diff_goal)
        v_goal = diff_goal / (dist_goal ** 2 + 1e-5)

        v_obs_total = np.zeros(2)
        for obs in self.static_nfz:
            diff_obs = pos - np.array(obs[:2])
            dist_obs = np.linalg.norm(diff_obs)
            if dist_obs < obs[2] * 2:
                v_obs_total += diff_obs / (dist_obs ** 3 + 1e-5) * (obs[2] ** 2)

        v_fluid = v_goal + v_obs_total
        u_fluid = v_fluid / (np.linalg.norm(v_fluid) + 1e-5)
        return u_fluid

    def _update_dynamic_obstacles(self):
        """更新所有动态障碍物的位置 (改为遍历列表)"""
        # 合并遍历所有的动态禁飞区和动态威胁源
        for obj in self.dyn_obs_list + self.dyn_threat_list:
            total_dist = np.linalg.norm(obj['end'] - obj['start'])
            if total_dist < 1e-5: continue
            cycle_time = total_dist / obj['speed'] / self.cfg.DT
            phase = (self.step_count % (2 * cycle_time)) / cycle_time
            if phase <= 1.0:
                obj['pos'] = obj['start'] + phase * (obj['end'] - obj['start'])
            else:
                obj['pos'] = obj['end'] + (phase - 1.0) * (obj['start'] - obj['end'])

    def _get_obs(self):
        """获取当前状态：兼容多个动态障碍物"""
        dist_goal = np.linalg.norm(self.goal_pos - self.pos) / self.cfg.MAP_SIZE
        dir_goal = np.arctan2(self.goal_pos[1] - self.pos[1], self.goal_pos[0] - self.pos[0]) - self.yaw

        all_obs = []
        for obs in self.static_nfz: all_obs.append(obs)
        for thr in self.static_threats: all_obs.append(thr)

        # 遍历动态障碍物列表，提取它们的实时坐标和半径
        for dyn_obs in self.dyn_obs_list:
            all_obs.append([dyn_obs['pos'][0], dyn_obs['pos'][1], dyn_obs['radius']])
        for dyn_thr in self.dyn_threat_list:
            all_obs.append([dyn_thr['pos'][0], dyn_thr['pos'][1], dyn_thr['radius']])

        obs_info = []
        for obs in all_obs:
            dist = np.linalg.norm(np.array(obs[:2]) - self.pos) / self.cfg.MAP_SIZE
            direction = np.arctan2(obs[1] - self.pos[1], obs[0] - self.pos[0]) - self.yaw
            obs_info.append((dist, direction))

        obs_info.sort(key=lambda x: x[0])
        closest_5_obs = obs_info[:5]

        obs_array = [dist_goal, np.sin(dir_goal)]
        for ob in closest_5_obs:
            obs_array.extend([ob[0], np.sin(ob[1])])

        if getattr(self, 'use_fluid', False):
            fluid_vec = self._calculate_fluid_dynamics(self.pos)
            obs_array.extend(fluid_vec)

        return np.array(obs_array, dtype=np.float32)

    def step(self, action):
        """环境步进：更新所有碰撞检测与惩罚逻辑以支持列表遍历"""
        prev_pos = np.copy(self.pos)
        prev_dist_goal = np.linalg.norm(self.goal_pos - self.pos)

        self.yaw_rate = action[0] * self.cfg.MAX_YAW_RATE
        self.yaw += self.yaw_rate * self.cfg.DT
        vx = self.cfg.UAV_SPEED * np.cos(self.yaw)
        vy = self.cfg.UAV_SPEED * np.sin(self.yaw)
        self.pos[0] += vx * self.cfg.DT
        self.pos[1] += vy * self.cfg.DT

        # 收集步级统计指标
        static_dists = [np.linalg.norm(np.array(obs[:2]) - self.pos) for obs in self.static_nfz + self.static_threats]
        step_avg_static_dist = np.mean(static_dists) if static_dists else 0.0

        # 【修改点】计算多个动态障碍物的平均距离
        dyn_dists = [np.linalg.norm(obj['pos'] - self.pos) for obj in self.dyn_obs_list + self.dyn_threat_list]
        step_avg_dyn_dist = np.mean(dyn_dists) if dyn_dists else 0.0

        step_curvature = np.abs(self.yaw_rate)

        # 边界检测
        if self.pos[0] < 0 or self.pos[0] > self.cfg.MAP_SIZE or self.pos[1] < 0 or self.pos[1] > self.cfg.MAP_SIZE:
            info = {
                'is_success': 0, 'is_collision': 1, 'reason': 'out_of_bounds',
                'metrics': {'static_dist': step_avg_static_dist, 'dyn_dist': step_avg_dyn_dist,
                            'curvature': step_curvature}
            }
            return self._get_obs(), self.cfg.COEFF_COLLISION, True, False, info

        current_z = self._generate_terrain_height(self.pos[0], self.pos[1])
        self.path_history.append((self.pos[0], self.pos[1], current_z))

        self._update_dynamic_obstacles()
        self.step_count += 1

        reward = self.cfg.COEFF_STEP
        terminated = False
        truncated = self.step_count >= 400
        info = {'is_success': 0, 'is_collision': 0}

        current_dist_goal = np.linalg.norm(self.goal_pos - self.pos)
        dist_diff = prev_dist_goal - current_dist_goal
        max_possible_dist = self.cfg.UAV_SPEED * self.cfg.DT
        norm_diff = dist_diff / max_possible_dist

        if norm_diff > 0:
            reward += self.cfg.COEFF_FORWARD * norm_diff
        else:
            reward += self.cfg.COEFF_BACKWARD * (norm_diff ** 2)

        target_yaw = np.arctan2(self.goal_pos[1] - self.pos[1], self.goal_pos[0] - self.pos[0])
        yaw_diff = (target_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi
        dir_reward = self.cfg.COEFF_DIR * (1.0 - np.abs(yaw_diff) / np.pi)

        reward += dir_reward
        reward += self.cfg.COEFF_CURVE * (self.yaw_rate ** 2)
        reward += self.cfg.COEFF_HEIGHT * current_z

        # 【修改点】威胁区惩罚计算 (整合动态威胁源列表)
        dyn_threats_format = [[obj['pos'][0], obj['pos'][1], obj['radius']] for obj in self.dyn_threat_list]
        for thr in self.static_threats + dyn_threats_format:
            dist = np.linalg.norm(np.array(thr[:2]) - self.pos)
            if dist < thr[2]:
                intensity = 1.0 - (dist / thr[2])
                reward += self.cfg.COEFF_THREAT * (intensity ** 3)

        if self.use_pbrs:
            phi_prev = self.cfg.PBRS_SCALE * (1.0 - prev_dist_goal / self.cfg.MAP_SIZE)
            phi_curr = self.cfg.PBRS_SCALE * (1.0 - current_dist_goal / self.cfg.MAP_SIZE)
            reward += self.cfg.GAMMA * phi_curr - phi_prev

        # 【修改点】碰撞检测 (整合动态禁飞区列表)
        dyn_obs_format = [[obj['pos'][0], obj['pos'][1], obj['radius']] for obj in self.dyn_obs_list]
        for obs in self.static_nfz + dyn_obs_format:
            if np.linalg.norm(np.array(obs[:2]) - self.pos) < obs[2]:
                reward += self.cfg.COEFF_COLLISION
                terminated = True
                info['is_success'] = 0
                info['is_collision'] = 1
                info['reason'] = 'collision'
                break

        if current_dist_goal < 500.0:
            reward += self.cfg.COEFF_GOAL
            terminated = True
            info['is_success'] = 1
            info['is_collision'] = 0
            info['reason'] = 'goal_reached'

        if not terminated:
            reward = np.clip(reward, -2.0, 2.0)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_random_scenario()
        self.pos = np.copy(self.start_pos)
        self.yaw = np.arctan2(self.goal_pos[1] - self.pos[1], self.goal_pos[0] - self.pos[0])
        self.yaw_rate = 0.0
        self.step_count = 0
        self.path_history = [(self.pos[0], self.pos[1], self._generate_terrain_height(self.pos[0], self.pos[1]))]
        return self._get_obs(), {}


def plot_results(training_data_list, save_dir="Training_Results"):
    """
    通用绘图与数据保存函数。(支持独立子图与平滑双曲线)
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    if not training_data_list:
        print("暂无训练数据可供保存与绘图。")
        return

    # 1. 转换为 DataFrame 并保存为 CSV (由于传入的是 dict 列表，绝对不会有空白/错位)
    df = pd.DataFrame(training_data_list)
    csv_path = os.path.join(save_dir, "training_data_log.csv")
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"训练数据已成功保存至 CSV: {csv_path}")
    except Exception as e:
        print(f"保存 CSV 文件时发生异常: {e}")

    # 2. 动态创建绘图网格
    # 排除 'episode' 列，剩下的全部独立绘图
    cols_to_plot = [col for col in df.columns if col != 'episode']
    n_cols = len(cols_to_plot)

    n_grid_cols = 2  # 排成 2 列
    n_grid_rows = (n_cols + 1) // 2

    fig, axs = plt.subplots(n_grid_rows, n_grid_cols, figsize=(15, 4 * n_grid_rows))
    axs = axs.flatten()

    window_size = 50  # 平滑窗口大小

    # 3. 循环遍历每个数据列进行独立绘图
    for i, col in enumerate(cols_to_plot):
        ax = axs[i]
        raw_data = df[col]

        # 针对 0 和 1 的离散数据，计算滚动平均后乘以 100 转换为百分比率
        if col in ['is_success', 'is_collision']:
            smoothed_data = raw_data.rolling(window=window_size, min_periods=1).mean() * 100
            raw_data = raw_data * 100
            ylabel = f"{col} Rate (%)"
        else:
            smoothed_data = raw_data.rolling(window=window_size, min_periods=1).mean()
            ylabel = col

        # 绘制原始数据（透明度 0.2，浅色）
        ax.plot(df['episode'], raw_data, alpha=0.2, color='blue', label='Raw (Per Episode)')
        # 绘制平滑数据（透明度 1.0，深色加粗）
        ax.plot(df['episode'], smoothed_data, alpha=1.0, color='red', linewidth=2, label=f'Smoothed (MA={window_size})')

        ax.set_title(col.replace('_', ' ').title())
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    # 隐藏多余的空白子图
    for j in range(len(cols_to_plot), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    img_path = os.path.join(save_dir, "Comprehensive_Metrics.png")
    plt.savefig(img_path)
    plt.close()
    print(f"训练图像已独立分图并保存至: {img_path}")