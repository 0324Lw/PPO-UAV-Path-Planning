import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 從環境文件中導入配置、環境類和統一繪圖接口
from env import UAVEnv, Config, plot_results


# ==========================================
# 訓練技巧: 運行狀態歸一化模組 (RunningMeanStd)
# ==========================================
class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.n + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.n
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.n * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.n = tot_count


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ==========================================
# 經驗回放池 (Replay Buffer)
# ==========================================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dead = np.zeros((max_size, 1), dtype=np.float32)  # Terminated 標誌

    def add(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.dead[ind])
        )


# ==========================================
# TD3 網路結構: Actor 與 Twin-Critic
# ==========================================
class MLP_Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(MLP_Actor, self).__init__()
        # Actor 直接輸出確定性動作
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
            nn.Tanh()  # 確保輸出範圍在 [-1, 1] 之間
        )

    def forward(self, state):
        return self.actor(state)


class MLP_Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(MLP_Critic, self).__init__()

        # Q1 網路
        self.q1 = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Q2 網路
        self.q2 = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)


# ==========================================
# 主訓練循環
# ==========================================
def train():
    cfg = Config()
    # 【基線設置】：關閉 PBRS 勢能獎勵和全局流體特徵，模擬傳統純雷達避障
    env = UAVEnv(cfg=cfg, use_pbrs=False, use_fluid=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 純基線觀察維度與動作維度
    obs_dim = 12
    act_dim = 1
    max_action = 1.0

    # TD3 專屬超參數
    BATCH_SIZE = 256
    GAMMA = cfg.GAMMA
    TAU = 0.005  # 軟更新係數
    EXPL_NOISE = 0.1  # 探索噪聲
    POLICY_NOISE = 0.2  # 策略平滑噪聲
    NOISE_CLIP = 0.5  # 平滑噪聲裁剪範圍
    POLICY_FREQ = 2  # 延遲更新頻率
    LEARNING_STARTS = 10000  # 預先收集經驗步數

    # 初始化網路與目標網路
    actor = MLP_Actor(obs_dim, act_dim).to(device)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.LR)

    critic = MLP_Critic(obs_dim, act_dim).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.LR)

    # 經驗回放池與歸一化器
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    training_data_list = []
    recent_rewards = deque(maxlen=50)
    recent_success = deque(maxlen=50)
    recent_collisions = deque(maxlen=50)

    episode_count = 0
    ep_reward = 0
    ep_steps = 0
    ep_static_dists, ep_dyn_dists, ep_curvatures = [], [], []
    current_pg_loss, current_v_loss = 0.0, 0.0

    print("開始 TD3 基線訓練 (關閉先驗引導與歷史差分)...")

    obs, _ = env.reset()
    obs_rms.update(np.array([obs]))
    obs_norm = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    # 主循環採用 Timesteps 取代原本的 Epoch 邏輯
    for t in range(int(cfg.TOTAL_TIMESTEPS)):

        # 1. 動作選擇 (預熱期採用隨機動作)
        if t < LEARNING_STARTS:
            action = np.random.uniform(-max_action, max_action, size=act_dim)
        else:
            obs_tensor = torch.FloatTensor(obs_norm).to(device).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            # 添加探索高斯噪聲
            noise = np.random.normal(0, max_action * EXPL_NOISE, size=act_dim)
            action = np.clip(action + noise, -max_action, max_action)

        # 2. 環境交互
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 終止狀態 (用於計算 TD Error，超時截斷不被視為死亡)
        dead = 1.0 if terminated else 0.0

        # 3. 狀態記錄與指標攔截
        ep_reward += reward
        ep_steps += 1

        try:
            stat_d = [np.linalg.norm(np.array(o[:2]) - env.pos) for o in env.static_nfz + env.static_threats]
            ep_static_dists.append(np.mean(stat_d) if stat_d else 0.0)

            if hasattr(env, 'dyn_obs_list'):
                dyn_d = [np.linalg.norm(obj['pos'] - env.pos) for obj in env.dyn_obs_list + env.dyn_threat_list]
            elif hasattr(env, 'dyn_obs_pos'):
                dyn_d = [np.linalg.norm(env.dyn_obs_pos - env.pos), np.linalg.norm(env.dyn_threat_pos - env.pos)]
            else:
                dyn_d = [0.0]
            ep_dyn_dists.append(np.mean(dyn_d) if dyn_d else 0.0)

            ep_curvatures.append(np.abs(env.yaw_rate))
        except Exception:
            pass

        # 4. 更新歸一化模塊
        obs_rms.update(np.array([next_obs]))
        next_obs_norm = np.clip((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

        # 5. 存儲經驗至回放池
        replay_buffer.add(obs_norm, action, reward, next_obs_norm, dead)
        obs_norm = next_obs_norm

        # 6. 回合結束處理
        if done:
            episode_count += 1
            is_success = 1 if info.get('is_success') == True else 0
            is_collision = 1 if info.get('reason') == 'collision' else 0

            if is_success == 1:
                avg_stat = np.mean(ep_static_dists) if len(ep_static_dists) > 0 else np.nan
                avg_dyn = np.mean(ep_dyn_dists) if len(ep_dyn_dists) > 0 else np.nan
                avg_curv = np.mean(ep_curvatures) if len(ep_curvatures) > 0 else np.nan
            else:
                avg_stat = np.nan
                avg_dyn = np.nan
                avg_curv = np.nan

            ep_data = {
                'episode': episode_count,
                'reward': ep_reward,
                'steps': ep_steps,
                'is_success': is_success,
                'is_collision': is_collision,
                'avg_static_dist': avg_stat,
                'avg_dyn_dist': avg_dyn,
                'path_curvature': avg_curv,
                'v_loss': current_v_loss,
                'pg_loss': current_pg_loss
            }
            training_data_list.append(ep_data)

            recent_rewards.append(ep_reward)
            recent_success.append(is_success)
            recent_collisions.append(is_collision)

            # 重置回合參數
            ep_reward, ep_steps = 0, 0
            ep_static_dists, ep_dyn_dists, ep_curvatures = [], [], []

            obs, _ = env.reset()
            obs_rms.update(np.array([obs]))
            obs_norm = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

        # 7. TD3 訓練過程 (滿足經驗長度後開始)
        if t >= LEARNING_STARTS:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dead = replay_buffer.sample(BATCH_SIZE)
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_rewards = batch_rewards.to(device)
            batch_next_states = batch_next_states.to(device)
            batch_dead = batch_dead.to(device)

            with torch.no_grad():
                # 目標策略平滑化 (Target Policy Smoothing)
                noise = (torch.randn_like(batch_actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                next_action = (actor_target(batch_next_states) + noise).clamp(-max_action, max_action)

                # 雙 Q 網路計算目標 Q 值
                target_q1, target_q2 = critic_target(batch_next_states, next_action)
                target_q = batch_rewards + GAMMA * (1 - batch_dead) * torch.min(target_q1, target_q2)

            # 更新 Critic 網路
            current_q1, current_q2 = critic(batch_states, batch_actions)
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            current_v_loss = critic_loss.item()

            # 延遲更新 Actor 網路與目標網路
            if t % POLICY_FREQ == 0:
                # 採用 Q1 的期望來更新策略
                actor_loss = -critic.Q1(batch_states, actor(batch_states)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                current_pg_loss = actor_loss.item()

                # 軟更新 (Soft Update) 目標網路
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        # 8. 終端訓練日誌輸出
        if t > 0 and t % (cfg.NUM_STEPS * 5) == 0 and len(recent_rewards) > 0:
            print(f"Timestep: {t:07d} | Ep: {episode_count} | Rew: {np.mean(recent_rewards):6.2f} | "
                  f"Succ: {np.mean(recent_success) * 100:5.1f}% | "
                  f"Q_loss: {current_v_loss:5.2f} | Pi_loss: {current_pg_loss:5.2f}")

    print("\n訓練完成，正在調用環境接口繪圖...")
    plot_results(training_data_list, save_dir="TD3_Results")


if __name__ == "__main__":
    train()