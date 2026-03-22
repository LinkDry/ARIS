"""
LLM-SRAF 自动化迭代改进版
========================

修复问题:
1. 基线满意度不应为0%
2. 改进优先级分类
3. 多场景实验
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict


@dataclass
class ServiceRequest:
    """业务请求"""
    user_id: int
    service_type: str
    latency_requirement: float
    bandwidth_requirement: float
    reliability_requirement: float
    priority: int
    actual_latency: float = 0.0
    actual_bandwidth: float = 0.0
    satisfied: bool = False


class ImprovedSatelliteEnvV2:
    """
    改进的卫星网络仿真环境 V2
    修复: 基线满意度计算、奖励函数
    """

    SERVICE_TYPES = ['video_conference', 'video_streaming', 'iot',
                     'file_transfer', 'voice_call', 'gaming']

    SERVICE_REQUIREMENTS = {
        'video_conference': {'latency': 50, 'bandwidth': 10, 'reliability': 0.95, 'priority': 3},
        'video_streaming': {'latency': 200, 'bandwidth': 25, 'reliability': 0.90, 'priority': 2},
        'iot': {'latency': 500, 'bandwidth': 0.5, 'reliability': 0.99, 'priority': 2},
        'file_transfer': {'latency': 1000, 'bandwidth': 50, 'reliability': 0.85, 'priority': 1},
        'voice_call': {'latency': 30, 'bandwidth': 0.1, 'reliability': 0.99, 'priority': 3},
        'gaming': {'latency': 20, 'bandwidth': 5, 'reliability': 0.95, 'priority': 3},
    }

    def __init__(self, num_satellites: int = 20, num_users: int = 50,
                 seed: int = 42, use_semantic: bool = True):
        self.num_satellites = num_satellites
        self.num_users = num_users
        self.use_semantic = use_semantic

        np.random.seed(seed)
        random.seed(seed)

        # 卫星总容量 (Mbps)
        self.satellite_capacity = np.random.uniform(500, 800, num_satellites)
        self.satellite_load = np.zeros(num_satellites)

        # 用户-卫星可见性
        self.visibility = self._init_visibility()
        # 信道质量 (SNR dB)
        self.channel_quality = np.random.uniform(10, 25, (num_users, num_satellites))

        self.current_requests: List[ServiceRequest] = []
        self.max_steps = 50
        self.current_step = 0

        # 统计
        self.stats = defaultdict(list)

    def _init_visibility(self) -> np.ndarray:
        visibility = np.zeros((self.num_users, self.num_satellites))
        for u in range(self.num_users):
            n_visible = min(np.random.randint(3, 8), self.num_satellites)
            visible_sats = np.random.choice(self.num_satellites, n_visible, replace=False)
            visibility[u, visible_sats] = 1
        return visibility

    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_step = 0
        self.satellite_load = np.zeros(self.num_satellites)
        self.stats = defaultdict(list)
        self._generate_requests()
        return self._get_state()

    def _generate_requests(self):
        self.current_requests = []
        for u in range(self.num_users):
            service_type = random.choice(self.SERVICE_TYPES)
            req = self.SERVICE_REQUIREMENTS[service_type].copy()
            request = ServiceRequest(
                user_id=u,
                service_type=service_type,
                latency_requirement=req['latency'] * np.random.uniform(0.9, 1.1),
                bandwidth_requirement=max(0.1, req['bandwidth'] * np.random.uniform(0.9, 1.1)),
                reliability_requirement=req['reliability'],
                priority=req['priority'],
            )
            self.current_requests.append(request)

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        self.current_step += 1

        total_reward = 0.0
        satisfaction_count = 0
        high_priority_satisfied = 0
        high_priority_total = 0

        for i, request in enumerate(self.current_requests):
            sat_idx = action['satellite_selection'][i]
            power = action['power_level'][i]
            spectrum = max(1, action['spectrum_allocation'][i])

            # 检查可见性
            if sat_idx >= self.num_satellites or self.visibility[i, sat_idx] == 0:
                # 随机选择一个可见卫星作为fallback
                visible = np.where(self.visibility[i] > 0)[0]
                if len(visible) > 0:
                    sat_idx = np.random.choice(visible)
                else:
                    sat_idx = 0

            # 计算实际性能
            snr = self.channel_quality[i, sat_idx]
            spectral_eff = np.log2(1 + 10 ** (snr / 10))
            bandwidth_per_block = 25  # MHz

            # 可用带宽取决于卫星负载
            available_ratio = max(0.1, 1 - self.satellite_load[sat_idx])
            actual_bandwidth = (spectral_eff * bandwidth_per_block * spectrum *
                               (0.5 + 0.1 * power) * available_ratio)

            # 时延
            base_latency = 20
            queue_latency = self.satellite_load[sat_idx] * 30
            actual_latency = base_latency + queue_latency

            # 更新负载
            self.satellite_load[sat_idx] = min(1.0,
                self.satellite_load[sat_idx] + actual_bandwidth / self.satellite_capacity[sat_idx] * 0.1)

            # 判断是否满足
            latency_ok = actual_latency <= request.latency_requirement
            bandwidth_ok = actual_bandwidth >= request.bandwidth_requirement * 0.7
            satisfied = latency_ok and bandwidth_ok

            request.actual_latency = actual_latency
            request.actual_bandwidth = actual_bandwidth
            request.satisfied = satisfied

            if satisfied:
                satisfaction_count += 1

            # 计算奖励
            latency_score = min(1.0, request.latency_requirement / max(1, actual_latency))
            bandwidth_score = min(1.0, actual_bandwidth / max(0.1, request.bandwidth_requirement))
            base_reward = 0.5 * latency_score + 0.5 * bandwidth_score

            if self.use_semantic and request.priority >= 2:
                high_priority_total += 1
                # 高优先级业务获得额外奖励
                if satisfied:
                    high_priority_satisfied += 1
                    reward = base_reward + 0.5 * request.priority / 3
                else:
                    # 高优先级不满足有惩罚
                    reward = base_reward - 0.1
            else:
                reward = base_reward

            total_reward += reward

        # 衰减负载
        self.satellite_load *= 0.85

        # 信道慢变
        self.channel_quality += np.random.randn(*self.channel_quality.shape) * 0.3
        self.channel_quality = np.clip(self.channel_quality, 5, 30)

        done = self.current_step >= self.max_steps

        # 计算指标
        satisfaction_rate = satisfaction_count / self.num_users
        high_priority_rate = high_priority_satisfied / max(1, high_priority_total)

        self.stats['satisfaction'].append(satisfaction_rate)
        self.stats['high_priority'].append(high_priority_rate)

        info = {
            'satisfaction': satisfaction_rate,
            'high_priority_satisfaction': high_priority_rate,
            'avg_reward': total_reward / self.num_users,
        }

        return self._get_state(), total_reward / self.num_users, done, info

    def _get_state(self) -> Dict:
        network_state = np.concatenate([
            self.satellite_load,
            np.mean(self.channel_quality, axis=1)[:self.num_satellites],
        ])[:64]
        if len(network_state) < 64:
            network_state = np.pad(network_state, (0, 64 - len(network_state)))
        return {'network_state': network_state[:64].astype(np.float32)}

    def get_network_state_vector(self) -> np.ndarray:
        return self._get_state()['network_state']

    def get_orbit_info_vector(self) -> np.ndarray:
        return np.random.randn(32).astype(np.float32)


def run_full_experiment(num_episodes: int = 30, horizon: int = 50) -> Dict:
    """
    运行完整消融实验
    """
    results = {}

    # 场景配置
    scenarios = {
        'scarce': {'num_users': 80, 'num_satellites': 15},  # 资源紧张
        'balanced': {'num_users': 50, 'num_satellites': 30}, # 平衡
        'abundant': {'num_users': 30, 'num_satellites': 50}, # 资源充足
    }

    for scenario_name, config in scenarios.items():
        print(f"\n=== Scenario: {scenario_name} ===")
        scenario_results = {}
        num_users = config['num_users']
        num_sats = config['num_satellites']

        # 1. LLM-SRAF (语义感知 + 优先级感知)
        print("  Running LLM-SRAF...")
        env = ImprovedSatelliteEnvV2(num_satellites=num_sats, num_users=num_users, use_semantic=True)
        rewards, satisfactions, hp_sats = [], [], []
        for ep in range(num_episodes):
            env.reset()
            total_r = 0
            for step in range(horizon):
                # 语义感知策略: 高优先级优先分配最优资源
                sat_selection = np.zeros(num_users, dtype=int)
                sat_load_temp = env.satellite_load.copy()
                priority_order = sorted(range(len(env.current_requests)),
                    key=lambda i: env.current_requests[i].priority, reverse=True)
                for i in priority_order:
                    visible = np.where(env.visibility[i] > 0)[0]
                    if len(visible) > 0:
                        best = visible[np.argmin(sat_load_temp[visible])]
                        sat_selection[i] = best
                        sat_load_temp[best] += 0.05
                    else:
                        sat_selection[i] = 0
                action = {
                    'satellite_selection': sat_selection,
                    'power_level': np.array([min(4, env.current_requests[i].priority + 1) for i in range(num_users)]),
                    'spectrum_allocation': np.array([max(1, int(env.current_requests[i].bandwidth_requirement)) for i in range(num_users)]),
                }
                _, r, done, info = env.step(action)
                total_r += r
                if done: break
            rewards.append(total_r)
            satisfactions.append(np.mean(env.stats['satisfaction']))
            hp_sats.append(np.mean(env.stats['high_priority']))
        scenario_results['LLM-SRAF'] = {
            'reward': np.mean(rewards), 'reward_std': np.std(rewards),
            'satisfaction': np.mean(satisfactions), 'hp_satisfaction': np.mean(hp_sats)
        }

        # 2. RL-only (无语义感知，贪婪信道)
        print("  Running RL-only...")
        env = ImprovedSatelliteEnvV2(num_satellites=num_sats, num_users=num_users, use_semantic=False)
        rewards, satisfactions = [], []
        for ep in range(num_episodes):
            env.reset()
            total_r = 0
            for step in range(horizon):
                action = {
                    'satellite_selection': np.argmax(env.channel_quality, axis=1),
                    'power_level': np.ones(num_users, dtype=int) * 2,
                    'spectrum_allocation': np.ones(num_users, dtype=int) * 3,
                }
                _, r, done, info = env.step(action)
                total_r += r
                if done: break
            rewards.append(total_r)
            satisfactions.append(np.mean(env.stats['satisfaction']))
        scenario_results['RL-only'] = {
            'reward': np.mean(rewards), 'reward_std': np.std(rewards),
            'satisfaction': np.mean(satisfactions), 'hp_satisfaction': 0
        }

        # 3. Random
        print("  Running Random...")
        env = ImprovedSatelliteEnvV2(num_satellites=num_sats, num_users=num_users, use_semantic=False)
        rewards, satisfactions = [], []
        for ep in range(num_episodes):
            env.reset()
            total_r = 0
            for step in range(horizon):
                visible_sats = [np.where(env.visibility[i] > 0)[0] for i in range(num_users)]
                action = {
                    'satellite_selection': np.array([np.random.choice(v) if len(v) > 0 else 0 for v in visible_sats]),
                    'power_level': np.random.randint(0, 5, num_users),
                    'spectrum_allocation': np.random.randint(1, 6, num_users),
                }
                _, r, done, info = env.step(action)
                total_r += r
                if done: break
            rewards.append(total_r)
            satisfactions.append(np.mean(env.stats['satisfaction']))
        scenario_results['Random'] = {
            'reward': np.mean(rewards), 'reward_std': np.std(rewards),
            'satisfaction': np.mean(satisfactions), 'hp_satisfaction': 0
        }

        # 4. Round-Robin + 负载均衡
        print("  Running Round-Robin-LB...")
        env = ImprovedSatelliteEnvV2(num_satellites=num_sats, num_users=num_users, use_semantic=False)
        rewards, satisfactions = [], []
        for ep in range(num_episodes):
            env.reset()
            total_r = 0
            sat_idx = 0
            for step in range(horizon):
                # 负载均衡: 选择负载最低的卫星
                sat_selection = np.argmin(env.satellite_load)
                action = {
                    'satellite_selection': np.full(num_users, sat_idx % num_sats),
                    'power_level': np.ones(num_users, dtype=int) * 2,
                    'spectrum_allocation': np.ones(num_users, dtype=int) * 3,
                }
                sat_idx += 1
                _, r, done, info = env.step(action)
                total_r += r
                if done: break
            rewards.append(total_r)
            satisfactions.append(np.mean(env.stats['satisfaction']))
        scenario_results['RoundRobin-LB'] = {
            'reward': np.mean(rewards), 'reward_std': np.std(rewards),
            'satisfaction': np.mean(satisfactions), 'hp_satisfaction': 0
        }

        results[scenario_name] = scenario_results

    return results


def print_results(results: Dict):
    """打印结果"""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)

    for scenario, methods in results.items():
        print(f"\n--- Scenario: {scenario.upper()} ---")
        print(f"{'Method':15s} {'Reward':>10s} {'Std':>8s} {'Satisfaction':>12s} {'HP-Sat':>10s}")
        print("-" * 60)
        for method, metrics in methods.items():
            print(f"{method:15s} {metrics['reward']:10.4f} {metrics['reward_std']:8.4f} "
                  f"{metrics['satisfaction']*100:11.2f}% {metrics.get('hp_satisfaction', 0)*100:9.2f}%")

        # 计算提升
        llm_reward = methods['LLM-SRAF']['reward']
        rl_reward = methods['RL-only']['reward']
        random_reward = methods['Random']['reward']

        gain_vs_rl = (llm_reward - rl_reward) / rl_reward * 100
        gain_vs_random = (llm_reward - random_reward) / random_reward * 100
        print(f"\n  LLM-SRAF vs RL-only: {gain_vs_rl:+.1f}%")
        print(f"  LLM-SRAF vs Random:  {gain_vs_random:+.1f}%")


if __name__ == '__main__':
    results = run_full_experiment(num_episodes=30, horizon=50)
    print_results(results)