"""
LLM-SRAF V5: 语义-结构解耦架构
==================================

核心洞察: 语义信息不应直接参与数值决策，而应作为元信息层

架构变化:
- V1-V4: 语义嵌入 → 直接参与资源分配 → 失败
- V5: 语义嵌入 → 策略选择/参数调整 → 结构化分配

三种解耦方式:
1. 语义引导的策略选择 (LLM Strategy Selector)
2. 语义引导的参数调整 (LLM Parameter Tuner)
3. 语义作为约束生成器 (LLM Constraint Generator)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict
from scipy import stats as scipy_stats
from enum import Enum


class AllocationStrategy(Enum):
    PRIORITY_FIRST = "priority_first"
    FAIRNESS_AWARE = "fairness_aware"
    LATENCY_OPTIMIZED = "latency_optimized"
    BANDWIDTH_OPTIMIZED = "bandwidth_optimized"
    HYBRID = "hybrid"


@dataclass
class ServiceRequest:
    user_id: int
    service_type: str
    latency_req: float
    bandwidth_req: float
    priority: int
    latency_weight: float
    bandwidth_weight: float
    urgency_hint: str = "normal"  # normal, urgent, flexible
    actual_latency: float = 0.0
    actual_bandwidth: float = 0.0
    satisfied: bool = False


class DecoupledEnv:
    """
    解耦架构环境

    支持多种分配策略，语义层负责选择策略而非直接参与分配
    """

    SERVICE_CONFIGS = {
        'gaming':           {'priority': 3, 'latency': 20,  'bandwidth': 5,   'lat_w': 0.9, 'bw_w': 0.1},
        'video_conference': {'priority': 3, 'latency': 50,  'bandwidth': 10,  'lat_w': 0.7, 'bw_w': 0.3},
        'voice_call':       {'priority': 3, 'latency': 30,  'bandwidth': 0.1, 'lat_w': 0.95, 'bw_w': 0.05},
        'video_streaming':  {'priority': 2, 'latency': 200, 'bandwidth': 25,  'lat_w': 0.3, 'bw_w': 0.7},
        'live_stream':      {'priority': 2, 'latency': 100, 'bandwidth': 20,  'lat_w': 0.4, 'bw_w': 0.6},
        'iot':              {'priority': 2, 'latency': 500, 'bandwidth': 0.5, 'lat_w': 0.2, 'bw_w': 0.8},
        'file_transfer':    {'priority': 1, 'latency': 1000, 'bandwidth': 50, 'lat_w': 0.1, 'bw_w': 0.9},
        'web_browse':       {'priority': 1, 'latency': 300, 'bandwidth': 5,   'lat_w': 0.5, 'bw_w': 0.5},
    }

    def __init__(self, num_sats=20, num_users=50, seed=42, service_weights=None):
        self.num_sats = num_sats
        self.num_users = num_users
        self.service_weights = service_weights

        np.random.seed(seed)
        random.seed(seed)

        self.sat_capacity = np.random.uniform(400, 600, num_sats)
        self.sat_load = np.zeros(num_sats)
        self.visibility = self._init_visibility()
        self.channel_quality = np.random.uniform(10, 25, (num_users, num_sats))
        self.current_requests = []
        self.max_steps = 50
        self.current_step = 0

    def _init_visibility(self):
        vis = np.zeros((self.num_users, self.num_sats))
        for u in range(self.num_users):
            n = min(np.random.randint(3, 6), self.num_sats)
            vis[u, np.random.choice(self.num_sats, n, replace=False)] = 1
        return vis

    def reset(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        self.current_step = 0
        self.sat_load = np.zeros(self.num_sats)
        self._gen_requests()
        return self._get_state()

    def _gen_requests(self):
        self.current_requests = []
        service_types = list(self.SERVICE_CONFIGS.keys())

        if self.service_weights:
            # 使用权重采样
            weighted_types = []
            for st, weight in self.service_weights.items():
                weighted_types.extend([st] * weight)
            # 如果权重不足，补充均匀分布
            remaining = self.num_users - len(weighted_types)
            if remaining > 0:
                weighted_types.extend(random.choices(service_types, k=remaining))
            service_assignment = weighted_types[:self.num_users]
        else:
            # 均匀分布
            service_assignment = [service_types[i % len(service_types)] for i in range(self.num_users)]

        for u in range(self.num_users):
            st = service_assignment[u]
            cfg = self.SERVICE_CONFIGS[st]

            # 添加紧急度提示
            urgency = random.choice(["normal", "urgent", "flexible"])

            self.current_requests.append(ServiceRequest(
                user_id=u, service_type=st,
                latency_req=cfg['latency'] * np.random.uniform(0.9, 1.1),
                bandwidth_req=max(0.1, cfg['bandwidth'] * np.random.uniform(0.9, 1.1)),
                priority=cfg['priority'],
                latency_weight=cfg['lat_w'],
                bandwidth_weight=cfg['bw_w'],
                urgency_hint=urgency
            ))

    def step(self, action):
        """执行一步"""
        self.current_step += 1
        total_reward = 0
        sat_count = 0

        for i, req in enumerate(self.current_requests):
            sat_idx = action['satellite_selection'][i]
            power = action['power_level'][i]
            spectrum = max(1, action['spectrum_allocation'][i])

            if sat_idx >= self.num_sats or self.visibility[i, sat_idx] == 0:
                visible = np.where(self.visibility[i] > 0)[0]
                sat_idx = np.random.choice(visible) if len(visible) > 0 else 0

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

            # 基础奖励
            lat_score = min(1.0, req.latency_req / max(1, latency))
            bw_score = min(1.0, bandwidth / max(0.1, req.bandwidth_req))
            reward = 0.5 * lat_score + 0.5 * bw_score

            # 优先级加成
            if satisfied and req.priority >= 2:
                reward += 0.2 * req.priority

            total_reward += reward

        self.sat_load *= 0.9
        self.channel_quality = np.clip(self.channel_quality + np.random.randn(*self.channel_quality.shape) * 0.2, 5, 30)

        done = self.current_step >= self.max_steps
        return self._get_state(), total_reward / self.num_users, done, {
            'satisfaction': sat_count / self.num_users
        }

    def _get_state(self):
        return {'network_state': np.zeros(64, dtype=np.float32)}


# ============== 分配策略实现 ==============

def priority_first_strategy(env):
    """策略1: 优先级优先"""
    sat_sel = np.zeros(env.num_users, dtype=int)
    load_temp = env.sat_load.copy()

    priority_order = sorted(range(len(env.current_requests)),
        key=lambda i: env.current_requests[i].priority, reverse=True)

    for i in priority_order:
        visible = np.where(env.visibility[i] > 0)[0]
        if len(visible) > 0:
            best = visible[np.argmin(load_temp[visible])]
            sat_sel[i] = best
            load_temp[best] += 0.03

    power_sel = np.array([min(4, env.current_requests[i].priority + 1) for i in range(env.num_users)])
    spectrum_sel = np.array([min(6, env.current_requests[i].priority + 2) for i in range(env.num_users)])

    return {
        'satellite_selection': sat_sel,
        'power_level': power_sel,
        'spectrum_allocation': spectrum_sel
    }


def fairness_aware_strategy(env):
    """策略2: 公平性感知"""
    sat_sel = np.zeros(env.num_users, dtype=int)
    load_temp = env.sat_load.copy()

    # 按资源需求排序，小需求优先
    demand_order = sorted(range(len(env.current_requests)),
        key=lambda i: env.current_requests[i].bandwidth_req)

    for i in demand_order:
        visible = np.where(env.visibility[i] > 0)[0]
        if len(visible) > 0:
            best = visible[np.argmin(load_temp[visible])]
            sat_sel[i] = best
            load_temp[best] += 0.02

    power_sel = np.ones(env.num_users, dtype=int) * 2
    spectrum_sel = np.array([min(6, max(1, int(env.current_requests[i].bandwidth_req))) for i in range(env.num_users)])

    return {
        'satellite_selection': sat_sel,
        'power_level': power_sel,
        'spectrum_allocation': spectrum_sel
    }


def latency_optimized_strategy(env):
    """策略3: 时延优化"""
    sat_sel = np.zeros(env.num_users, dtype=int)

    # 时延敏感用户选择信道质量最好的卫星
    for i, req in enumerate(env.current_requests):
        visible = np.where(env.visibility[i] > 0)[0]
        if len(visible) > 0:
            # 综合信道质量和负载
            scores = env.channel_quality[i, visible] - env.sat_load[visible] * 10
            best = visible[np.argmax(scores)]
            sat_sel[i] = best

    power_sel = np.ones(env.num_users, dtype=int) * 4  # 高功率
    spectrum_sel = np.ones(env.num_users, dtype=int) * 3

    return {
        'satellite_selection': sat_sel,
        'power_level': power_sel,
        'spectrum_allocation': spectrum_sel
    }


def bandwidth_optimized_strategy(env):
    """策略4: 带宽优化"""
    sat_sel = np.zeros(env.num_users, dtype=int)
    load_temp = env.sat_load.copy()

    # 带宽需求大的用户优先
    bw_order = sorted(range(len(env.current_requests)),
        key=lambda i: env.current_requests[i].bandwidth_req, reverse=True)

    for i in bw_order:
        visible = np.where(env.visibility[i] > 0)[0]
        if len(visible) > 0:
            best = visible[np.argmin(load_temp[visible])]
            sat_sel[i] = best
            load_temp[best] += 0.05

    power_sel = np.ones(env.num_users, dtype=int) * 2
    spectrum_sel = np.ones(env.num_users, dtype=int) * 6  # 高频谱

    return {
        'satellite_selection': sat_sel,
        'power_level': power_sel,
        'spectrum_allocation': spectrum_sel
    }


def hybrid_strategy(env):
    """策略5: 混合策略 - 根据业务特征动态调整"""
    sat_sel = np.zeros(env.num_users, dtype=int)
    power_sel = np.zeros(env.num_users, dtype=int)
    spectrum_sel = np.zeros(env.num_users, dtype=int)
    load_temp = env.sat_load.copy()

    # 综合得分: 优先级 + 业务特征
    scored_users = []
    for i, req in enumerate(env.current_requests):
        score = req.priority * 10
        if req.latency_weight > 0.7:
            score += 5  # 时延敏感加成
        if req.urgency_hint == "urgent":
            score += 8
        scored_users.append((i, score))
    scored_users.sort(key=lambda x: x[1], reverse=True)

    for i, _ in scored_users:
        visible = np.where(env.visibility[i] > 0)[0]
        if len(visible) > 0:
            best = visible[np.argmin(load_temp[visible])]
            sat_sel[i] = best
            load_temp[best] += 0.03

        req = env.current_requests[i]
        if req.latency_weight > 0.7:
            power_sel[i], spectrum_sel[i] = 4, 3
        elif req.bandwidth_weight > 0.7:
            power_sel[i], spectrum_sel[i] = 2, 6
        else:
            power_sel[i], spectrum_sel[i] = 3, 4

    return {
        'satellite_selection': sat_sel,
        'power_level': power_sel,
        'spectrum_allocation': spectrum_sel
    }


# ============== 语义策略选择器 ==============

class SemanticStrategySelector:
    """
    语义引导的策略选择器

    根据当前请求集合的语义特征，选择最合适的分配策略
    """

    def __init__(self):
        self.strategies = {
            AllocationStrategy.PRIORITY_FIRST: priority_first_strategy,
            AllocationStrategy.FAIRNESS_AWARE: fairness_aware_strategy,
            AllocationStrategy.LATENCY_OPTIMIZED: latency_optimized_strategy,
            AllocationStrategy.BANDWIDTH_OPTIMIZED: bandwidth_optimized_strategy,
            AllocationStrategy.HYBRID: hybrid_strategy,
        }

    def analyze_scenario(self, requests: List[ServiceRequest]) -> Dict:
        """分析当前场景的语义特征"""
        # 统计业务类型分布
        type_counts = defaultdict(int)
        priority_dist = defaultdict(int)
        urgency_counts = defaultdict(int)

        total_lat_weight = 0
        total_bw_weight = 0

        for req in requests:
            type_counts[req.service_type] += 1
            priority_dist[req.priority] += 1
            urgency_counts[req.urgency_hint] += 1
            total_lat_weight += req.latency_weight
            total_bw_weight += req.bandwidth_weight

        n = len(requests)

        return {
            'diversity': len(type_counts) / len(DecoupledEnv.SERVICE_CONFIGS),  # 业务多样性
            'avg_latency_sensitivity': total_lat_weight / n,
            'avg_bandwidth_sensitivity': total_bw_weight / n,
            'high_priority_ratio': (priority_dist[3] + priority_dist[2]) / n,
            'urgency_ratio': urgency_counts.get('urgent', 0) / n,
        }

    def select_strategy(self, requests: List[ServiceRequest]) -> AllocationStrategy:
        """根据语义特征选择策略"""
        features = self.analyze_scenario(requests)

        # 决策规则（基于语义特征）
        # 规则1: 高时延敏感场景 → 时延优化策略
        if features['avg_latency_sensitivity'] > 0.6:
            return AllocationStrategy.LATENCY_OPTIMIZED

        # 规则2: 高带宽敏感场景 → 带宽优化策略
        if features['avg_bandwidth_sensitivity'] > 0.6:
            return AllocationStrategy.BANDWIDTH_OPTIMIZED

        # 规则3: 高优先级集中场景 → 优先级优先策略
        if features['high_priority_ratio'] > 0.5:
            return AllocationStrategy.PRIORITY_FIRST

        # 规则4: 高多样性 + 高紧急度 → 混合策略
        if features['diversity'] > 0.6 and features['urgency_ratio'] > 0.2:
            return AllocationStrategy.HYBRID

        # 规则5: 默认公平策略
        return AllocationStrategy.FAIRNESS_AWARE

    def get_action(self, env) -> Dict:
        """获取动作"""
        strategy = self.select_strategy(env.current_requests)
        return self.strategies[strategy](env)


# ============== 实验函数 ==============

def run_decoupled_method(env, selector, horizon=50):
    """运行解耦方法"""
    env.reset()
    total_r = 0
    sat_list = []

    for step in range(horizon):
        action = selector.get_action(env)
        _, r, done, info = env.step(action)
        total_r += r
        sat_list.append(info['satisfaction'])
        if done:
            break

    return total_r, np.mean(sat_list)


def run_fixed_strategy(env, strategy_func, horizon=50):
    """运行固定策略"""
    env.reset()
    total_r = 0
    sat_list = []

    for step in range(horizon):
        action = strategy_func(env)
        _, r, done, info = env.step(action)
        total_r += r
        sat_list.append(info['satisfaction'])
        if done:
            break

    return total_r, np.mean(sat_list)


def run_experiment_v5():
    """运行V5实验: 语义-结构解耦架构"""
    print("=" * 70)
    print("LLM-SRAF V5 - Decoupled Semantic-Structure Architecture")
    print("=" * 70)

    methods = {
        'Semantic-Selector': lambda e: run_decoupled_method(e, SemanticStrategySelector()),
        'Priority-First': lambda e: run_fixed_strategy(e, priority_first_strategy),
        'Fairness-Aware': lambda e: run_fixed_strategy(e, fairness_aware_strategy),
        'Latency-Opt': lambda e: run_fixed_strategy(e, latency_optimized_strategy),
        'Bandwidth-Opt': lambda e: run_fixed_strategy(e, bandwidth_optimized_strategy),
        'Hybrid': lambda e: run_fixed_strategy(e, hybrid_strategy),
    }

    # 场景配置
    scenarios = {
        'latency_critical': {
            'desc': '时延敏感场景 (gaming/voice为主)',
            'config': lambda: DecoupledEnv(num_sats=15, num_users=60, seed=42,
                service_weights={'gaming': 3, 'voice_call': 3, 'video_conference': 2})
        },
        'bandwidth_heavy': {
            'desc': '带宽密集场景 (视频流/文件传输为主)',
            'config': lambda: DecoupledEnv(num_sats=15, num_users=60, seed=42,
                service_weights={'video_streaming': 3, 'file_transfer': 3, 'live_stream': 2})
        },
        'mixed': {
            'desc': '混合场景 (均匀分布)',
            'config': lambda: DecoupledEnv(num_sats=15, num_users=60, seed=42)
        },
        'high_priority': {
            'desc': '高优先级集中场景',
            'config': lambda: DecoupledEnv(num_sats=15, num_users=60, seed=42,
                service_weights={'gaming': 5, 'video_conference': 5, 'voice_call': 5})
        }
    }

    all_results = {}

    for scenario_name, scenario_info in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_info['desc']}")
        print("=" * 70)

        results = {m: {'rewards': [], 'sats': []} for m in methods}

        for seed in range(5):
            # 使用场景配置创建环境
            env = scenario_info['config']()
            # 重新设置seed
            np.random.seed(seed*100)
            random.seed(seed*100)

            for name, run_func in methods.items():
                for _ in range(10):
                    r, s = run_func(env)
                    results[name]['rewards'].append(r)
                    results[name]['sats'].append(s)

        all_results[scenario_name] = results

        # 打印结果
        print(f"\n{'Method':20s} {'Reward':>10s} {'Std':>8s} {'Satisfaction':>12s}")
        print("-" * 55)

        for name in methods:
            r = results[name]['rewards']
            s = results[name]['sats']
            print(f"{name:20s} {np.mean(r):10.4f} {np.std(r):8.4f} {np.mean(s)*100:11.2f}%")

        # 统计检验
        selector_r = results['Semantic-Selector']['rewards']
        priority_r = results['Priority-First']['rewards']

        t_stat, p_val = scipy_stats.ttest_ind(selector_r, priority_r)
        gain = (np.mean(selector_r) - np.mean(priority_r)) / np.mean(priority_r) * 100

        print(f"\n  Semantic-Selector vs Priority-First: {gain:+.2f}% (p={p_val:.4f})")

    # 跨场景分析
    print(f"\n{'='*70}")
    print("Cross-Scenario Analysis")
    print("=" * 70)

    # 计算每个场景的最优策略
    print(f"\n{'Scenario':20s} {'Best Fixed':>15s} {'Selector Gain':>15s}")
    print("-" * 55)

    for scenario_name, results in all_results.items():
        selector_r = np.mean(results['Semantic-Selector']['rewards'])

        # 找最佳固定策略
        best_fixed = max(methods.keys(), key=lambda m: np.mean(results[m]['rewards']) if m != 'Semantic-Selector' else -1)
        best_fixed_r = np.mean(results[best_fixed]['rewards'])

        gain = (selector_r - best_fixed_r) / best_fixed_r * 100
        print(f"{scenario_name:20s} {best_fixed:15s} {gain:+14.2f}%")

    return all_results


if __name__ == '__main__':
    results = run_experiment_v5()