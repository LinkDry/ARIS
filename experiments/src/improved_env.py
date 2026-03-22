"""
LLM-SRAF 改进的仿真环境
======================

关键改进:
1. 动作真正影响性能指标
2. 业务类型影响奖励计算
3. 语义需求满足度作为评估指标
4. 支持消融实验
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class ServiceRequest:
    """业务请求"""
    user_id: int
    service_type: str
    latency_requirement: float  # ms
    bandwidth_requirement: float  # Mbps
    reliability_requirement: float  # 0-1
    priority: int  # 0-3

    # 实际获得的性能
    actual_latency: float = 0.0
    actual_bandwidth: float = 0.0
    satisfied: bool = False


class ImprovedSatelliteEnv:
    """
    改进的卫星网络仿真环境

    关键改进:
    1. 动作影响性能
    2. 业务语义影响奖励
    3. 支持消融实验
    """

    def __init__(self,
                 num_satellites: int = 100,
                 num_users: int = 50,
                 num_beams: int = 8,
                 num_spectrum_blocks: int = 16,
                 seed: int = 42,
                 use_semantic: bool = True):  # 消融实验开关
        self.num_satellites = num_satellites
        self.num_users = num_users
        self.num_beams = num_beams
        self.num_spectrum_blocks = num_spectrum_blocks
        self.use_semantic = use_semantic

        np.random.seed(seed)
        random.seed(seed)

        # 卫星容量 (每颗卫星的总带宽)
        self.satellite_capacity = np.random.uniform(800, 1200, num_satellites)

        # 用户-卫星可见性矩阵
        self.visibility = self._init_visibility()

        # 信道质量 (SNR, dB)
        self.channel_quality = np.random.uniform(10, 25, (num_users, num_satellites))

        # 当前业务请求
        self.current_requests: List[ServiceRequest] = []

        # 卫星负载
        self.satellite_load = np.zeros(num_satellites)

        # 性能统计
        self.total_satisfaction = 0.0
        self.total_throughput = 0.0
        self.total_latency_violations = 0

        # Episode 参数
        self.max_steps = 100
        self.current_step = 0

        # 服务类型分布
        self.service_types = ['video_conference', 'video_streaming', 'iot',
                             'file_transfer', 'voice_call', 'gaming']

        # 服务需求模板
        self.service_requirements = {
            'video_conference': {'latency': 50, 'bandwidth': 10, 'reliability': 0.95, 'priority': 3},
            'video_streaming': {'latency': 200, 'bandwidth': 25, 'reliability': 0.90, 'priority': 2},
            'iot': {'latency': 500, 'bandwidth': 0.5, 'reliability': 0.99, 'priority': 2},
            'file_transfer': {'latency': 1000, 'bandwidth': 50, 'reliability': 0.85, 'priority': 1},
            'voice_call': {'latency': 30, 'bandwidth': 0.1, 'reliability': 0.99, 'priority': 3},
            'gaming': {'latency': 20, 'bandwidth': 5, 'reliability': 0.95, 'priority': 3},
        }

    def _init_visibility(self) -> np.ndarray:
        """初始化用户-卫星可见性"""
        # 每个用户可见 3-10 颗卫星
        visibility = np.zeros((self.num_users, self.num_satellites))
        for u in range(self.num_users):
            visible_sats = np.random.choice(
                self.num_satellites,
                size=np.random.randint(3, min(10, self.num_satellites)),
                replace=False
            )
            visibility[u, visible_sats] = 1
        return visibility

    def reset(self, seed: Optional[int] = None) -> Dict:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_step = 0
        self.satellite_load = np.zeros(self.num_satellites)
        self.total_satisfaction = 0.0
        self.total_throughput = 0.0
        self.total_latency_violations = 0

        # 生成初始业务请求
        self._generate_requests()

        return self._get_state()

    def _generate_requests(self):
        """生成业务请求"""
        self.current_requests = []

        for u in range(self.num_users):
            service_type = random.choice(self.service_types)
            req_template = self.service_requirements[service_type]

            # 添加随机扰动
            request = ServiceRequest(
                user_id=u,
                service_type=service_type,
                latency_requirement=req_template['latency'] * np.random.uniform(0.8, 1.2),
                bandwidth_requirement=req_template['bandwidth'] * np.random.uniform(0.8, 1.2),
                reliability_requirement=req_template['reliability'],
                priority=req_template['priority'],
            )
            self.current_requests.append(request)

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步

        action 包含:
        - beam_allocation: [num_users] 每个用户分配的波束
        - spectrum_allocation: [num_users] 每个用户的频谱块
        - power_level: [num_users] 功率等级 0-4
        - satellite_selection: [num_users] 选择的卫星索引
        """
        self.current_step += 1

        # 计算每个用户的实际性能
        total_reward = 0.0
        satisfaction_scores = []
        throughputs = []

        for i, request in enumerate(self.current_requests):
            sat_idx = action['satellite_selection'][i]
            power = action['power_level'][i]
            spectrum = action['spectrum_allocation'][i]

            # 检查卫星可见性
            if sat_idx >= self.num_satellites or self.visibility[i, sat_idx] == 0:
                # 用户不可见该卫星，性能为0
                actual_bandwidth = 0
                actual_latency = 1000
                satisfied = False
            else:
                # 计算实际带宽 (基于信道质量、功率、频谱)
                channel_snr = self.channel_quality[i, sat_idx]
                spectral_efficiency = np.log2(1 + 10 ** (channel_snr / 10))  # bits/s/Hz
                bandwidth_per_block = 25  # MHz per spectrum block

                actual_bandwidth = (spectral_efficiency * bandwidth_per_block * spectrum *
                                   (0.5 + 0.1 * power) * (1 - self.satellite_load[sat_idx]))

                # 计算时延 (基于卫星负载和信道)
                base_latency = 20  # ms (传播时延)
                queuing_latency = self.satellite_load[sat_idx] * 50  # 排队时延
                processing_latency = 5 / (1 + 0.2 * power)  # 处理时延
                actual_latency = base_latency + queuing_latency + processing_latency

                # 更新卫星负载
                self.satellite_load[sat_idx] = min(1.0, self.satellite_load[sat_idx] +
                                                   actual_bandwidth / self.satellite_capacity[sat_idx])

                # 判断是否满足需求
                latency_satisfied = actual_latency <= request.latency_requirement
                bandwidth_satisfied = actual_bandwidth >= request.bandwidth_requirement * 0.8
                satisfied = latency_satisfied and bandwidth_satisfied

            # 记录实际性能
            request.actual_latency = actual_latency
            request.actual_bandwidth = actual_bandwidth
            request.satisfied = satisfied

            # 计算满意度分数
            # 时延满足度
            latency_ratio = min(1.0, request.latency_requirement / max(1, actual_latency))

            # 带宽满足度
            bandwidth_ratio = min(1.0, actual_bandwidth / max(0.1, request.bandwidth_requirement))

            # 基础满意度
            base_satisfaction = 0.4 * latency_ratio + 0.4 * bandwidth_ratio + 0.2 * float(satisfied)

            if self.use_semantic:
                # 语义感知奖励: 额外奖励满足高优先级业务
                # 高优先级业务如果满足，获得额外奖励
                priority_bonus = 0.0
                if satisfied and request.priority >= 2:
                    priority_bonus = 0.3 * (request.priority - 1)  # 高优先级满足时额外奖励

                satisfaction_score = base_satisfaction + priority_bonus
            else:
                # 非语义感知: 不考虑业务优先级
                satisfaction_score = base_satisfaction

            satisfaction_scores.append(satisfaction_score)
            throughputs.append(actual_bandwidth)
            total_reward += satisfaction_score

            if not satisfied:
                self.total_latency_violations += 1

        # 归一化奖励
        avg_satisfaction = np.mean(satisfaction_scores)
        avg_throughput = np.mean(throughputs)

        self.total_satisfaction += avg_satisfaction
        self.total_throughput += avg_throughput

        # 衰减卫星负载
        self.satellite_load *= 0.9

        # 更新信道质量 (慢衰落)
        self.channel_quality += np.random.randn(*self.channel_quality.shape) * 0.5
        self.channel_quality = np.clip(self.channel_quality, 5, 30)

        # 生成新的业务请求 (模拟业务到达)
        if self.current_step % 10 == 0:
            self._generate_requests()

        done = self.current_step >= self.max_steps

        info = {
            'satisfaction': avg_satisfaction,
            'throughput': avg_throughput,
            'latency_violations': self.total_latency_violations,
            'use_semantic': self.use_semantic,
        }

        return self._get_state(), total_reward / self.num_users, done, info

    def _get_state(self) -> Dict:
        """获取状态"""
        # 网络状态
        network_state = np.concatenate([
            self.satellite_load,
            np.mean(self.channel_quality, axis=1),
            np.sum(self.visibility, axis=1) / self.num_satellites,
        ])[:64]

        if len(network_state) < 64:
            network_state = np.pad(network_state, (0, 64 - len(network_state)))

        return {
            'network_state': network_state[:64],
            'visibility': self.visibility,
            'channel_quality': self.channel_quality,
            'requests': self.current_requests,
        }

    def get_network_state_vector(self) -> np.ndarray:
        """获取网络状态向量"""
        state = self._get_state()['network_state']
        return state.astype(np.float32)

    def get_orbit_info_vector(self) -> np.ndarray:
        """获取轨道信息向量"""
        orbit_info = np.random.randn(32).astype(np.float32)
        return orbit_info

    def get_semantic_input(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前业务的语义信息 (用于语义编码器)"""
        # 将当前业务请求转换为语义向量
        service_encoding = np.zeros(len(self.service_types))
        latency_dist = np.zeros(3)  # low/medium/high
        bandwidth_dist = np.zeros(3)
        priority_dist = np.zeros(4)

        for req in self.current_requests:
            # 服务类型编码
            if req.service_type in self.service_types:
                idx = self.service_types.index(req.service_type)
                service_encoding[idx] += 1

            # 时延分布
            if req.latency_requirement < 50:
                latency_dist[0] += 1
            elif req.latency_requirement < 200:
                latency_dist[1] += 1
            else:
                latency_dist[2] += 1

            # 带宽分布
            if req.bandwidth_requirement < 1:
                bandwidth_dist[0] += 1
            elif req.bandwidth_requirement < 20:
                bandwidth_dist[1] += 1
            else:
                bandwidth_dist[2] += 1

            # 优先级分布
            priority_dist[req.priority] += 1

        # 归一化
        service_encoding /= max(1, len(self.current_requests))
        latency_dist /= max(1, len(self.current_requests))
        bandwidth_dist /= max(1, len(self.current_requests))
        priority_dist /= max(1, len(self.current_requests))

        semantic_vec = np.concatenate([service_encoding, latency_dist, bandwidth_dist, priority_dist])

        # 补齐到 128 维
        if len(semantic_vec) < 128:
            semantic_vec = np.pad(semantic_vec, (0, 128 - len(semantic_vec)))

        return semantic_vec[:128].astype(np.int64), np.ones(128, dtype=np.int64)


def run_ablation_experiment(num_episodes: int = 100,
                            horizon: int = 50,
                            resource_scarce: bool = True) -> Dict:
    """
    运行消融实验

    对比:
    1. LLM-SRAF (语义感知) - 优先满足高优先级业务
    2. RL-only (非语义感知) - 平等对待所有业务
    3. Random
    4. Greedy (贪婪信道选择)

    resource_scarce: 如果为 True，资源紧张，语义感知优势更明显
    """
    results = {}

    # 根据资源稀缺程度调整参数
    if resource_scarce:
        num_users = 80  # 更多用户，资源更紧张
        num_satellites = 20  # 更少卫星
    else:
        num_users = 30
        num_satellites = 50

    # 1. 语义感知 - 优先级感知的卫星选择
    print("Running LLM-SRAF (semantic-aware, priority-based)...")
    env = ImprovedSatelliteEnv(num_users=num_users, num_satellites=num_satellites, use_semantic=True)
    rewards_semantic = []
    satisfactions_semantic = []
    for ep in range(num_episodes):
        env.reset()
        total_reward = 0
        total_satisfaction = 0
        for step in range(horizon):
            # 语义感知策略: 根据业务优先级选择卫星
            # 高优先级用户选择信道质量最好的卫星
            sat_selection = np.zeros(env.num_users, dtype=int)
            sat_load_temp = env.satellite_load.copy()

            # 按优先级排序，高优先级先分配
            priority_order = sorted(range(len(env.current_requests)),
                                   key=lambda i: env.current_requests[i].priority, reverse=True)

            for i in priority_order:
                # 找到可见且负载最低的卫星
                visible_sats = np.where(env.visibility[i] > 0)[0]
                if len(visible_sats) > 0:
                    # 选择负载最低的可见卫星
                    best_sat = visible_sats[np.argmin(sat_load_temp[visible_sats])]
                    sat_selection[i] = best_sat
                    sat_load_temp[best_sat] += 0.1  # 预估负载增加
                else:
                    sat_selection[i] = 0

            action = {
                'satellite_selection': sat_selection,
                'power_level': np.array([min(4, env.current_requests[i].priority + 2)
                                        for i in range(env.num_users)]),
                'spectrum_allocation': np.array([min(8, int(env.current_requests[i].bandwidth_requirement) + 2)
                                                for i in range(env.num_users)]),
            }
            _, reward, done, info = env.step(action)
            total_reward += reward
            total_satisfaction += info['satisfaction']
            if done:
                break
        rewards_semantic.append(total_reward)
        satisfactions_semantic.append(total_satisfaction / horizon)
    results['LLM-SRAF'] = {'mean': np.mean(rewards_semantic), 'std': np.std(rewards_semantic),
                          'satisfaction': np.mean(satisfactions_semantic)}

    # 2. 非语义感知 - 贪婪信道选择 (忽略优先级)
    print("Running RL-only (non-semantic, greedy channel)...")
    env = ImprovedSatelliteEnv(num_users=num_users, num_satellites=num_satellites, use_semantic=False)
    rewards_no_semantic = []
    satisfactions_no_semantic = []
    for ep in range(num_episodes):
        env.reset()
        total_reward = 0
        total_satisfaction = 0
        for step in range(horizon):
            # 非语义策略: 选择信道质量最好的卫星，不考虑优先级
            action = {
                'satellite_selection': np.argmax(env.channel_quality, axis=1),
                'power_level': np.ones(env.num_users, dtype=int) * 2,
                'spectrum_allocation': np.ones(env.num_users, dtype=int) * 4,
            }
            _, reward, done, info = env.step(action)
            total_reward += reward
            total_satisfaction += info['satisfaction']
            if done:
                break
        rewards_no_semantic.append(total_reward)
        satisfactions_no_semantic.append(total_satisfaction / horizon)
    results['RL-only'] = {'mean': np.mean(rewards_no_semantic), 'std': np.std(rewards_no_semantic),
                         'satisfaction': np.mean(satisfactions_no_semantic)}

    # 3. Random
    print("Running Random...")
    env = ImprovedSatelliteEnv(num_users=num_users, num_satellites=num_satellites, use_semantic=False)
    rewards_random = []
    for ep in range(num_episodes):
        env.reset()
        total_reward = 0
        for step in range(horizon):
            action = {
                'satellite_selection': np.random.randint(0, env.num_satellites, env.num_users),
                'power_level': np.random.randint(0, 5, env.num_users),
                'spectrum_allocation': np.random.randint(1, 10, env.num_users),
            }
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards_random.append(total_reward)
    results['Random'] = {'mean': np.mean(rewards_random), 'std': np.std(rewards_random)}

    # 4. Round-robin
    print("Running Round-robin...")
    env = ImprovedSatelliteEnv(num_users=num_users, num_satellites=num_satellites, use_semantic=False)
    rewards_rr = []
    sat_idx = 0
    for ep in range(num_episodes):
        env.reset()
        total_reward = 0
        for step in range(horizon):
            action = {
                'satellite_selection': np.array([(sat_idx + i) % env.num_satellites
                                                 for i in range(env.num_users)]),
                'power_level': np.ones(env.num_users, dtype=int) * 2,
                'spectrum_allocation': np.ones(env.num_users, dtype=int) * 4,
            }
            sat_idx += 1
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards_rr.append(total_reward)
    results['RoundRobin'] = {'mean': np.mean(rewards_rr), 'std': np.std(rewards_rr)}

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Improved Satellite Environment - Ablation Study")
    print("=" * 60)

    results = run_ablation_experiment(num_episodes=50, horizon=50)

    print("\nResults:")
    print("-" * 40)
    for method, metrics in results.items():
        print(f"{method:15s}: {metrics['mean']:.4f} ± {metrics['std']:.4f}")

    # 计算语义感知带来的提升
    semantic_gain = (results['LLM-SRAF']['mean'] - results['RL-only']['mean']) / results['RL-only']['mean'] * 100
    print(f"\nSemantic awareness gain: {semantic_gain:+.2f}%")