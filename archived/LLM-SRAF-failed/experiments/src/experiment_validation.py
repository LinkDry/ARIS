"""
LLM-SRAF 验证实验 - DRL基线 + 大规模测试
==========================================

快速验证内容:
1. DRL基线对比 (PPO, DQN, A2C)
2. 大规模测试 (100卫星 × 500用户)
3. 与启发式方法对比
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Gymnasium环境
import gymnasium as gym
from gymnasium import spaces

# Stable-Baselines3
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


@dataclass
class ServiceRequest:
    user_id: int
    service_type: str
    latency_req: float
    bandwidth_req: float
    priority: int
    latency_weight: float
    bandwidth_weight: float
    urgency_hint: str = "normal"
    actual_latency: float = 0.0
    actual_bandwidth: float = 0.0
    satisfied: bool = False


class SatelliteResourceEnv(gym.Env):
    """
    卫星资源分配环境 - Gymnasium接口

    支持DRL训练和大规模测试
    """

    SERVICE_CONFIGS = {
        'gaming':           {'priority': 3, 'latency': 20,  'bandwidth': 5,   'lat_w': 0.9, 'bw_w': 0.1},
        'video_conference': {'priority': 3, 'latency': 50,  'bandwidth': 10,  'lat_w': 0.7, 'bw_w': 0.3},
        'voice_call':       {'priority': 3, 'latency': 30,  'bandwidth': 0.1, 'lat_w': 0.95, 'bw_w': 0.05},
        'video_streaming':  {'priority': 2, 'latency': 200, 'bandwidth': 25,  'lat_w': 0.3, 'bw_w': 0.7},
        'live_stream':      {'priority': 2, 'latency': 100, 'bandwidth': 20,  'lat_w': 0.4, 'bw_w': 0.6},
        'iot':              {'priority': 2, 'latency': 500, 'bandwidth': 0.5, 'lat_w': 0.2, 'bw_w': 0.8},
        'file_transfer':    {'priority': 1, 'latency': 1000, 'bandwidth': 50,  'lat_w': 0.1, 'bw_w': 0.9},
        'web_browse':       {'priority': 1, 'latency': 300, 'bandwidth': 5,   'lat_w': 0.5, 'bw_w': 0.5},
    }

    def __init__(self, num_sats=20, num_users=50, max_steps=50, seed=42):
        super().__init__()

        self.num_sats = num_sats
        self.num_users = num_users
        self.max_steps = max_steps
        self.seed_val = seed

        # 动作空间: 每个用户选择一个卫星 (离散)
        # 简化: 使用离散动作空间
        self.action_space = spaces.MultiDiscrete([num_sats] * num_users)

        # 观察空间: 网络状态向量
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(128,), dtype=np.float32
        )

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed_val = seed
        else:
            np.random.seed(self.seed_val)
            random.seed(self.seed_val)

        self.current_step = 0
        self.sat_capacity = np.random.uniform(400, 600, self.num_sats)
        self.sat_load = np.zeros(self.num_sats)
        self.visibility = self._init_visibility()
        self.channel_quality = np.random.uniform(10, 25, (self.num_users, self.num_sats))
        self.current_requests = []
        self._gen_requests()

        return self._get_obs(), {}

    def _init_visibility(self):
        vis = np.zeros((self.num_users, self.num_sats))
        for u in range(self.num_users):
            n = min(np.random.randint(3, max(4, self.num_sats // 3)), self.num_sats)
            vis[u, np.random.choice(self.num_sats, n, replace=False)] = 1
        return vis

    def _gen_requests(self):
        self.current_requests = []
        service_types = list(self.SERVICE_CONFIGS.keys())

        for u in range(self.num_users):
            st = service_types[u % len(service_types)]
            cfg = self.SERVICE_CONFIGS[st]

            self.current_requests.append(ServiceRequest(
                user_id=u, service_type=st,
                latency_req=cfg['latency'] * np.random.uniform(0.9, 1.1),
                bandwidth_req=max(0.1, cfg['bandwidth'] * np.random.uniform(0.9, 1.1)),
                priority=cfg['priority'],
                latency_weight=cfg['lat_w'],
                bandwidth_weight=cfg['bw_w'],
                urgency_hint=random.choice(["normal", "urgent", "flexible"])
            ))

    def _get_obs(self):
        """生成观察向量"""
        # 简化的状态表示
        state = np.zeros(128, dtype=np.float32)

        # 卫星负载 (归一化)
        state[:self.num_sats] = self.sat_load / np.max(self.sat_capacity)

        # 优先级分布
        priorities = [req.priority for req in self.current_requests]
        state[self.num_sats:self.num_sats+4] = [
            priorities.count(1) / len(priorities),
            priorities.count(2) / len(priorities),
            priorities.count(3) / len(priorities),
            np.mean(priorities) / 3
        ]

        # 时延敏感度统计
        lat_weights = [req.latency_weight for req in self.current_requests]
        state[self.num_sats+4:self.num_sats+6] = [np.mean(lat_weights), np.std(lat_weights)]

        # 带宽敏感度统计
        bw_weights = [req.bandwidth_weight for req in self.current_requests]
        state[self.num_sats+6:self.num_sats+8] = [np.mean(bw_weights), np.std(bw_weights)]

        return state

    def step(self, action):
        self.current_step += 1
        total_reward = 0
        sat_count = 0

        # 计算功率和频谱分配 (基于优先级)
        power_levels = np.array([min(4, self.current_requests[i].priority + 1) for i in range(self.num_users)])
        spectrum_levels = np.array([min(6, self.current_requests[i].priority + 2) for i in range(self.num_users)])

        for i, req in enumerate(self.current_requests):
            sat_idx = int(action[i]) if i < len(action) else 0

            # 验证卫星可见性
            if sat_idx >= self.num_sats or self.visibility[i, sat_idx] == 0:
                visible = np.where(self.visibility[i] > 0)[0]
                sat_idx = np.random.choice(visible) if len(visible) > 0 else 0

            power = power_levels[i]
            spectrum = max(1, spectrum_levels[i])

            snr = self.channel_quality[i, sat_idx]
            spec_eff = np.log2(1 + 10 ** (snr / 10))
            avail_ratio = max(0.1, 1 - self.sat_load[sat_idx])

            bandwidth = spec_eff * 25 * spectrum * (0.5 + 0.15 * power) * avail_ratio
            latency = 15 + (1 - power/5) * 10 + self.sat_load[sat_idx] * 40

            self.sat_load[sat_idx] = min(1.0, self.sat_load[sat_idx] + bandwidth / self.sat_capacity[sat_idx] * 0.08)

            lat_ok = latency <= req.latency_req
            bw_ok = bandwidth >= req.bandwidth_req * 0.8
            satisfied = lat_ok and bw_ok

            req.actual_latency = latency
            req.actual_bandwidth = bandwidth
            req.satisfied = satisfied

            if satisfied:
                sat_count += 1

            lat_score = min(1.0, req.latency_req / max(1, latency))
            bw_score = min(1.0, bandwidth / max(0.1, req.bandwidth_req))
            reward = 0.5 * lat_score + 0.5 * bw_score

            if satisfied and req.priority >= 2:
                reward += 0.2 * req.priority

            total_reward += reward

        self.sat_load *= 0.9
        self.channel_quality = np.clip(self.channel_quality + np.random.randn(*self.channel_quality.shape) * 0.2, 5, 30)

        done = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), total_reward / self.num_users, done, truncated, {
            'satisfaction': sat_count / self.num_users
        }


# ============== 启发式方法实现 ==============

def priority_first_heuristic(env):
    """优先级优先启发式"""
    env.reset(seed=env.seed_val)
    total_r = 0
    sat_list = []

    for _ in range(env.max_steps):
        action = np.zeros(env.num_users, dtype=np.int64)
        load_temp = env.sat_load.copy()

        priority_order = sorted(range(len(env.current_requests)),
            key=lambda i: env.current_requests[i].priority, reverse=True)

        for i in priority_order:
            visible = np.where(env.visibility[i] > 0)[0]
            if len(visible) > 0:
                best = visible[np.argmin(load_temp[visible])]
                action[i] = best
                load_temp[best] += 0.03

        obs, r, done, truncated, info = env.step(action)
        total_r += r
        sat_list.append(info['satisfaction'])

        if done:
            break

    return total_r, np.mean(sat_list)


def greedy_channel_heuristic(env):
    """贪婪信道选择"""
    env.reset(seed=env.seed_val)
    total_r = 0
    sat_list = []

    for _ in range(env.max_steps):
        action = np.argmax(env.channel_quality, axis=1).astype(np.int64)
        obs, r, done, truncated, info = env.step(action)
        total_r += r
        sat_list.append(info['satisfaction'])

        if done:
            break

    return total_r, np.mean(sat_list)


def random_heuristic(env):
    """随机策略"""
    env.reset(seed=env.seed_val)
    total_r = 0
    sat_list = []

    for _ in range(env.max_steps):
        action = np.array([np.random.choice(np.where(env.visibility[i] > 0)[0])
                          if np.any(env.visibility[i] > 0) else 0
                          for i in range(env.num_users)], dtype=np.int64)
        obs, r, done, truncated, info = env.step(action)
        total_r += r
        sat_list.append(info['satisfaction'])

        if done:
            break

    return total_r, np.mean(sat_list)


# ============== DRL训练函数 ==============

def train_drl_agent(env, algo='PPO', total_timesteps=50000, seed=42):
    """训练DRL智能体"""
    print(f"  Training {algo}...")

    # 创建向量化环境
    vec_env = make_vec_env(
        lambda: SatelliteResourceEnv(num_sats=env.num_sats, num_users=env.num_users, seed=seed),
        n_envs=1
    )

    # 选择算法
    if algo == 'PPO':
        model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed)
    elif algo == 'DQN':
        model = DQN("MlpPolicy", vec_env, verbose=0, seed=seed)
    elif algo == 'A2C':
        model = A2C("MlpPolicy", vec_env, verbose=0, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # 训练
    model.learn(total_timesteps=total_timesteps)

    return model


def evaluate_drl_agent(model, env, num_episodes=10):
    """评估DRL智能体"""
    rewards = []
    satisfactions = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep * 100)
        total_r = 0
        sat_list = []

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(action)
            total_r += r
            sat_list.append(info['satisfaction'])

            if done:
                break

        rewards.append(total_r)
        satisfactions.append(np.mean(sat_list))

    return np.mean(rewards), np.mean(satisfactions)


# ============== 主实验函数 ==============

def run_validation_experiment():
    """运行验证实验"""
    print("=" * 70)
    print("LLM-SRAF Validation Experiment")
    print("DRL Baselines + Large-Scale Testing")
    print("=" * 70)

    results = {}

    # ========== 实验1: 标准规模 (15卫星 × 60用户) ==========
    print("\n" + "=" * 70)
    print("Experiment 1: Standard Scale (15 satellites × 60 users)")
    print("=" * 70)

    env_standard = SatelliteResourceEnv(num_sats=15, num_users=60, max_steps=50, seed=42)

    # 启发式方法
    print("\n[Evaluating Heuristic Methods]")
    heuristic_methods = {
        'Priority-First': priority_first_heuristic,
        'Greedy-Channel': greedy_channel_heuristic,
        'Random': random_heuristic,
    }

    std_results = {}

    for name, method in heuristic_methods.items():
        rewards = []
        sats = []
        for seed in range(5):
            env_standard.seed_val = seed * 100
            r, s = method(env_standard)
            rewards.append(r)
            sats.append(s)
        std_results[name] = {'reward': np.mean(rewards), 'sat': np.mean(sats), 'std': np.std(rewards)}
        print(f"  {name}: Reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}, Sat={np.mean(sats)*100:.1f}%")

    # DRL方法 (减少训练步数以加快速度)
    print("\n[Training DRL Agents]")

    drl_methods = {}
    for algo in ['PPO', 'A2C']:
        try:
            model = train_drl_agent(env_standard, algo=algo, total_timesteps=30000, seed=42)
            r, s = evaluate_drl_agent(model, env_standard, num_episodes=10)
            drl_methods[algo] = {'reward': r, 'sat': s}
            print(f"  {algo}: Reward={r:.2f}, Sat={s*100:.1f}%")
        except Exception as e:
            print(f"  {algo}: Training failed - {e}")

    results['standard'] = {'heuristics': std_results, 'drl': drl_methods}

    # ========== 实验2: 大规模 (50卫星 × 200用户) ==========
    print("\n" + "=" * 70)
    print("Experiment 2: Large Scale (50 satellites × 200 users)")
    print("=" * 70)

    env_large = SatelliteResourceEnv(num_sats=50, num_users=200, max_steps=50, seed=42)

    print("\n[Evaluating Heuristic Methods]")
    large_results = {}

    for name, method in heuristic_methods.items():
        rewards = []
        sats = []
        for seed in range(3):  # 减少种子数
            env_large.seed_val = seed * 100
            r, s = method(env_large)
            rewards.append(r)
            sats.append(s)
        large_results[name] = {'reward': np.mean(rewards), 'sat': np.mean(sats), 'std': np.std(rewards)}
        print(f"  {name}: Reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}, Sat={np.mean(sats)*100:.1f}%")

    # DRL在大规模环境上训练
    print("\n[Training DRL Agents on Large Scale]")

    large_drl = {}
    for algo in ['PPO']:  # 仅用PPO以节省时间
        try:
            model = train_drl_agent(env_large, algo=algo, total_timesteps=50000, seed=42)
            r, s = evaluate_drl_agent(model, env_large, num_episodes=5)
            large_drl[algo] = {'reward': r, 'sat': s}
            print(f"  {algo}: Reward={r:.2f}, Sat={s*100:.1f}%")
        except Exception as e:
            print(f"  {algo}: Training failed - {e}")

    results['large'] = {'heuristics': large_results, 'drl': large_drl}

    # ========== 实验3: 超大规模 (100卫星 × 500用户) ==========
    print("\n" + "=" * 70)
    print("Experiment 3: Extra Large Scale (100 satellites × 500 users)")
    print("=" * 70)

    env_xlarge = SatelliteResourceEnv(num_sats=100, num_users=500, max_steps=50, seed=42)

    print("\n[Evaluating Heuristic Methods Only (DRL too slow for this scale)]")
    xlarge_results = {}

    for name, method in heuristic_methods.items():
        env_xlarge.seed_val = 42
        r, s = method(env_xlarge)
        xlarge_results[name] = {'reward': r, 'sat': s}
        print(f"  {name}: Reward={r:.2f}, Sat={s*100:.1f}%")

    results['xlarge'] = {'heuristics': xlarge_results}

    # ========== 结果汇总 ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Scale':20s} {'Best Method':>20s} {'Reward':>10s} {'Satisfaction':>12s}")
    print("-" * 70)

    for scale, scale_results in results.items():
        # 找最佳方法
        best_method = None
        best_reward = -float('inf')

        # 检查启发式
        for name, data in scale_results.get('heuristics', {}).items():
            if data['reward'] > best_reward:
                best_reward = data['reward']
                best_method = name
                best_sat = data['sat']

        # 检查DRL
        for name, data in scale_results.get('drl', {}).items():
            if data['reward'] > best_reward:
                best_reward = data['reward']
                best_method = name
                best_sat = data['sat']

        print(f"{scale:20s} {best_method:>20s} {best_reward:10.2f} {best_sat*100:11.1f}%")

    # 关键发现
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # 统计Priority-First vs DRL
    print("\nPriority-First vs DRL (Standard Scale):")
    pf_reward = std_results['Priority-First']['reward']
    for algo, data in drl_methods.items():
        diff = (pf_reward - data['reward']) / data['reward'] * 100
        print(f"  Priority-First vs {algo}: {diff:+.1f}%")

    # 规模扩展性
    print("\nScalability Analysis:")
    pf_std = std_results['Priority-First']['reward']
    pf_large = large_results['Priority-First']['reward']
    pf_xlarge = xlarge_results['Priority-First']['reward']
    print(f"  Priority-First scales well: {pf_std:.1f} → {pf_large:.1f} → {pf_xlarge:.1f}")

    return results


if __name__ == '__main__':
    results = run_validation_experiment()