import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

# 從環境文件中導入配置、環境類和統一繪圖接口
from env import UAVEnv, Config, plot_results


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


class MLP_ActorCritic(nn.Module):
    def __init__(self, obs_dim=12, act_dim=1):
        super(MLP_ActorCritic, self).__init__()
        # Actor 网络
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 26)),
            nn.Tanh(),
            layer_init(nn.Linear(26, act_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Critic 网络
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 26)),
            nn.Tanh(),
            layer_init(nn.Linear(26, 1), std=1.0)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train():
    cfg = Config()
    env = UAVEnv(cfg=cfg, use_pbrs=False, use_fluid=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    obs_dim = 12
    act_dim = 1

    agent = MLP_ActorCritic(obs_dim, act_dim).to(device)
    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"模型总参数量 (Trainable Parameters): {total_params}")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.LR, eps=1e-5)
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    training_data_list = []

    recent_rewards = deque(maxlen=50)
    recent_success = deque(maxlen=50)
    recent_collisions = deque(maxlen=50)

    obs_buf = torch.zeros((cfg.NUM_STEPS, obs_dim)).to(device)
    act_buf = torch.zeros((cfg.NUM_STEPS, act_dim)).to(device)
    logprob_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)
    rew_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)
    val_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)
    done_buf = torch.zeros((cfg.NUM_STEPS,)).to(device)

    global_step = 0
    episode_count = 0
    num_updates = cfg.TOTAL_TIMESTEPS // cfg.NUM_STEPS

    obs, _ = env.reset()
    obs_rms.update(np.array([obs]))
    obs_norm = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    # 回合級別指標容器
    ep_reward = 0
    ep_steps = 0
    ep_static_dists, ep_dyn_dists, ep_curvatures = [], [], []
    current_pg_loss, current_v_loss = 0.0, 0.0

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * cfg.LR

        for step in range(cfg.NUM_STEPS):
            global_step += 1
            obs_tensor = torch.FloatTensor(obs_norm).to(device)
            obs_buf[step] = obs_tensor
            done_buf[step] = 0

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor.unsqueeze(0))

            val_buf[step] = value.flatten()
            act_buf[step] = action.flatten()
            logprob_buf[step] = logprob.flatten()

            cpu_action = action.cpu().numpy().flatten()
            a_rl = np.tanh(cpu_action[0])

            next_obs, reward, terminated, truncated, info = env.step(np.array([a_rl]))

            # 記錄本步指標
            ep_reward += reward
            ep_steps += 1
            rew_buf[step] = reward

            # ==========================================
            # 【完美數據修復】：直接讀取物理坐標
            # ==========================================
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
            # ==========================================

            obs_rms.update(np.array([next_obs]))
            next_obs_norm = np.clip((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
            obs_norm = next_obs_norm

            if terminated or truncated:
                episode_count += 1

                is_success = 1 if info.get('is_success') == True else 0
                is_collision = 1 if info.get('reason') == 'collision' else 0

                # ==========================================
                # 【僅在成功時保存】：失敗則填補 NaN
                # ==========================================
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

                # 重置回合數據
                ep_reward, ep_steps = 0, 0
                ep_static_dists, ep_dyn_dists, ep_curvatures = [], [], []

                next_obs, _ = env.reset()
                obs_norm = np.clip((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
                done_buf[step] = 1

        with torch.no_grad():
            next_value = agent.get_value(torch.FloatTensor(obs_norm).to(device).unsqueeze(0)).flatten()
            advantages = torch.zeros_like(rew_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.NUM_STEPS)):
                nextnonterminal = 1.0 - (0 if t == cfg.NUM_STEPS - 1 else done_buf[t])
                nextvalues = next_value if t == cfg.NUM_STEPS - 1 else val_buf[t + 1]
                delta = rew_buf[t] + cfg.GAMMA * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + cfg.GAMMA * cfg.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_logprobs = logprob_buf.reshape(-1)
        b_actions = act_buf.reshape((-1, act_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_inds = np.arange(cfg.NUM_STEPS)

        for epoch in range(cfg.NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.NUM_STEPS, cfg.MINIBATCH_SIZE):
                end = start + cfg.MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.CLIP_COEF, 1 + cfg.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ENT_COEF * entropy_loss + v_loss * cfg.VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()

                current_pg_loss = pg_loss.item()
                current_v_loss = v_loss.item()

        if update % 5 == 0 and len(recent_rewards) > 0:
            print(f"Update: {update:04d} | Rew: {np.mean(recent_rewards):6.2f} | "
                  f"Succ: {np.mean(recent_success) * 100:5.1f}% | "
                  f"v_loss: {current_v_loss:5.2f} | p_loss: {current_pg_loss:5.2f}")

    plot_results(training_data_list, save_dir="MLPmini_Results")


if __name__ == "__main__":
    train()