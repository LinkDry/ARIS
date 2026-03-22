"""
LLM-SRAF V6: 智能策略选择器
============================

Level 1 参数微调迭代:
- 优化策略选择器的判断规则
- 调整阈值参数
- 添加场景自适应机制
- 实现加权评分决策

改进点:
1. 更细粒度的场景特征分析
2. 加权评分选择策略而非简单阈值
3. 动态阈值自适应
4. 策略组合支持
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict
from scipy import stats as scipy_stats
from enum import Enum
import json


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
    urgency_hint: str = "normal"
    actual_latency: float = 0.0
    actual_bandwidth: float = 0.0
    satisfied: bool = False


class DecoupledEnv:
    """解耦架构环境"""

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

    # 服务类型分类
    LATENCY_CRITICAL = {'gaming', 'voice_call', 'video_conference'}
    BANDWIDTH_HEAVY = {'video_streaming', 'file_transfer', 'live_stream'}
    BALANCED = {'iot', 'web_browse'}

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
            weighted_types = []
            for st, weight in self.service_weights.items():
                weighted_types.extend([st] * weight)
            remaining = self.num_users - len(weighted_types)
            if remaining > 0:
                weighted_types.extend(random.choices(service_types, k=remaining))
            service_assignment = weighted_types[:self.num_users]
        else:
            service_assignment = [service_types[i % len(service_types)] for i in range(self.num_users)]

        for u in range(self.num_users):
            st = service_assignment[u]
            cfg = self.SERVICE_CONFIGS[st]
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

            lat_score = min(1.0, req.latency_req / max(1, latency))
            bw_score = min(1.0, bandwidth / max(0.1, req.bandwidth_req))
            reward = 0.5 * lat_score + 0.5 * bw_score

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

    for i, req in enumerate(env.current_requests):
        visible = np.where(env.visibility[i] > 0)[0]
        if len(visible) > 0:
            scores = env.channel_quality[i, visible] - env.sat_load[visible] * 10
            best = visible[np.argmax(scores)]
            sat_sel[i] = best

    power_sel = np.ones(env.num_users, dtype=int) * 4
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

    bw_order = sorted(range(len(env.current_requests)),
        key=lambda i: env.current_requests[i].bandwidth_req, reverse=True)

    for i in bw_order:
        visible = np.where(env.visibility[i] > 0)[0]
        if len(visible) > 0:
            best = visible[np.argmin(load_temp[visible])]
            sat_sel[i] = best
            load_temp[best] += 0.05

    power_sel = np.ones(env.num_users, dtype=int) * 2
    spectrum_sel = np.ones(env.num_users, dtype=int) * 6

    return {
        'satellite_selection': sat_sel,
        'power_level': power_sel,
        'spectrum_allocation': spectrum_sel
    }


def hybrid_strategy(env):
    """策略5: 混合策略"""
    sat_sel = np.zeros(env.num_users, dtype=int)
    power_sel = np.zeros(env.num_users, dtype=int)
    spectrum_sel = np.zeros(env.num_users, dtype=int)
    load_temp = env.sat_load.copy()

    scored_users = []
    for i, req in enumerate(env.current_requests):
        score = req.priority * 10
        if req.latency_weight > 0.7:
            score += 5
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


# ============== 智能策略选择器 V2 ==============

class SmartStrategySelector:
    """
    智能策略选择器 V2

    改进点:
    1. 更细粒度的场景特征分析
    2. 加权评分选择策略
    3. 动态阈值自适应
    4. 策略历史表现追踪
    """

    def __init__(self):
        self.strategies = {
            AllocationStrategy.PRIORITY_FIRST: priority_first_strategy,
            AllocationStrategy.FAIRNESS_AWARE: fairness_aware_strategy,
            AllocationStrategy.LATENCY_OPTIMIZED: latency_optimized_strategy,
            AllocationStrategy.BANDWIDTH_OPTIMIZED: bandwidth_optimized_strategy,
            AllocationStrategy.HYBRID: hybrid_strategy,
        }

        # 策略性能历史
        self.strategy_history = {s: [] for s in self.strategies}

    def analyze_scenario(self, requests: List[ServiceRequest]) -> Dict:
        """细粒度场景特征分析"""
        n = len(requests)

        # 基础统计
        type_counts = defaultdict(int)
        priority_dist = defaultdict(int)
        urgency_counts = defaultdict(int)

        total_lat_weight = 0
        total_bw_weight = 0
        total_bw_req = 0
        total_lat_req = 0

        # 细分统计
        latency_critical_count = 0
        bandwidth_heavy_count = 0
        balanced_count = 0

        high_urgency_count = 0
        high_priority_count = 0

        for req in requests:
            type_counts[req.service_type] += 1
            priority_dist[req.priority] += 1
            urgency_counts[req.urgency_hint] += 1

            total_lat_weight += req.latency_weight
            total_bw_weight += req.bandwidth_weight
            total_bw_req += req.bandwidth_req
            total_lat_req += req.latency_req

            # 服务类型分类
            if req.service_type in DecoupledEnv.LATENCY_CRITICAL:
                latency_critical_count += 1
            elif req.service_type in DecoupledEnv.BANDWIDTH_HEAVY:
                bandwidth_heavy_count += 1
            else:
                balanced_count += 1

            if req.urgency_hint == "urgent":
                high_urgency_count += 1
            if req.priority >= 2:
                high_priority_count += 1

        # 计算特征指标
        features = {
            # 多样性
            'diversity': len(type_counts) / len(DecoupledEnv.SERVICE_CONFIGS),

            # 敏感度
            'avg_latency_sensitivity': total_lat_weight / n,
            'avg_bandwidth_sensitivity': total_bw_weight / n,

            # 优先级分布
            'high_priority_ratio': high_priority_count / n,
            'priority_variance': np.var([req.priority for req in requests]),

            # 紧急度
            'urgency_ratio': high_urgency_count / n,

            # 服务类型分布
            'latency_critical_ratio': latency_critical_count / n,
            'bandwidth_heavy_ratio': bandwidth_heavy_count / n,
            'balanced_ratio': balanced_count / n,

            # 需求特征
            'avg_bw_req': total_bw_req / n,
            'avg_lat_req': total_lat_req / n,
            'bw_variance': np.var([req.bandwidth_req for req in requests]),
            'lat_variance': np.var([req.latency_req for req in requests]),
        }

        return features

    def compute_strategy_scores(self, features: Dict) -> Dict[AllocationStrategy, float]:
        """计算每个策略的适配分数"""
        scores = {}

        # Priority-First: 高优先级集中 + 低多样性时表现好
        scores[AllocationStrategy.PRIORITY_FIRST] = (
            features['high_priority_ratio'] * 3.0 +
            features['priority_variance'] * 2.0 +
            (1 - features['diversity']) * 1.5
        )

        # Latency-Optimized: 时延敏感场景
        scores[AllocationStrategy.LATENCY_OPTIMIZED] = (
            features['latency_critical_ratio'] * 4.0 +
            features['avg_latency_sensitivity'] * 3.0 +
            features['urgency_ratio'] * 2.0
        )

        # Bandwidth-Optimized: 带宽密集场景
        scores[AllocationStrategy.BANDWIDTH_OPTIMIZED] = (
            features['bandwidth_heavy_ratio'] * 4.0 +
            features['avg_bandwidth_sensitivity'] * 3.0 +
            features['bw_variance'] * 0.5
        )

        # Fairness-Aware: 需求差异大时表现好
        scores[AllocationStrategy.FAIRNESS_AWARE] = (
            features['diversity'] * 2.0 +
            features['bw_variance'] * 0.3 +
            features['lat_variance'] * 0.3 +
            (1 - features['high_priority_ratio']) * 1.5
        )

        # Hybrid: 复杂混合场景
        scores[AllocationStrategy.HYBRID] = (
            features['diversity'] * 2.5 +
            features['urgency_ratio'] * 2.0 +
            abs(features['latency_critical_ratio'] - features['bandwidth_heavy_ratio']) * 1.0
        )

        return scores

    def select_strategy(self, requests: List[ServiceRequest]) -> AllocationStrategy:
        """选择最优策略"""
        features = self.analyze_scenario(requests)
        scores = self.compute_strategy_scores(features)

        # 选择得分最高的策略
        best_strategy = max(scores, key=scores.get)
        return best_strategy

    def get_action(self, env) -> Dict:
        """获取动作"""
        strategy = self.select_strategy(env.current_requests)
        return self.strategies[strategy](env)

    def update_history(self, strategy: AllocationStrategy, reward: float):
        """更新策略历史"""
        self.strategy_history[strategy].append(reward)


# ============== 自适应策略选择器 ==============

class AdaptiveStrategySelector:
    """
    自适应策略选择器

    特点:
    1. 根据历史表现动态调整权重
    2. 支持在线学习
    """

    def __init__(self, learning_rate=0.1):
        self.strategies = {
            AllocationStrategy.PRIORITY_FIRST: priority_first_strategy,
            AllocationStrategy.FAIRNESS_AWARE: fairness_aware_strategy,
            AllocationStrategy.LATENCY_OPTIMIZED: latency_optimized_strategy,
            AllocationStrategy.BANDWIDTH_OPTIMIZED: bandwidth_optimized_strategy,
            AllocationStrategy.HYBRID: hybrid_strategy,
        }

        # 初始权重
        self.weights = {s: 1.0 for s in self.strategies}
        self.learning_rate = learning_rate

        # 历史记录
        self.performance = {s: [] for s in self.strategies}
        self.selection_counts = {s: 0 for s in self.strategies}

    def analyze_scenario(self, requests: List[ServiceRequest]) -> Dict:
        """场景分析"""
        n = len(requests)

        latency_critical = sum(1 for r in requests if r.service_type in DecoupledEnv.LATENCY_CRITICAL)
        bandwidth_heavy = sum(1 for r in requests if r.service_type in DecoupledEnv.BANDWIDTH_HEAVY)
        high_priority = sum(1 for r in requests if r.priority >= 2)

        return {
            'latency_critical_ratio': latency_critical / n,
            'bandwidth_heavy_ratio': bandwidth_heavy / n,
            'high_priority_ratio': high_priority / n,
            'diversity': len(set(r.service_type for r in requests)) / len(DecoupledEnv.SERVICE_CONFIGS),
        }

    def select_strategy(self, requests: List[ServiceRequest]) -> AllocationStrategy:
        """选择策略"""
        features = self.analyze_scenario(requests)

        # 基于场景特征的初始得分
        base_scores = {
            AllocationStrategy.PRIORITY_FIRST: features['high_priority_ratio'],
            AllocationStrategy.LATENCY_OPTIMIZED: features['latency_critical_ratio'],
            AllocationStrategy.BANDWIDTH_OPTIMIZED: features['bandwidth_heavy_ratio'],
            AllocationStrategy.FAIRNESS_AWARE: features['diversity'],
            AllocationStrategy.HYBRID: 0.5,  # 中性
        }

        # 结合历史权重
        final_scores = {
            s: base_scores[s] * self.weights[s]
            for s in self.strategies
        }

        return max(final_scores, key=final_scores.get)

    def update(self, strategy: AllocationStrategy, reward: float):
        """更新权重"""
        self.performance[strategy].append(reward)
        self.selection_counts[strategy] += 1

        # 指数移动平均更新
        avg = np.mean(self.performance[strategy])
        self.weights[strategy] = 0.9 * self.weights[strategy] + 0.1 * (reward / 60)  # 归一化

    def get_action(self, env) -> Dict:
        """获取动作"""
        strategy = self.select_strategy(env.current_requests)
        return self.strategies[strategy](env)


# ============== 实验函数 ==============

def run_method(env, selector, horizon=50, adaptive=False):
    """运行方法"""
    env.reset()
    total_r = 0
    sat_list = []

    for step in range(horizon):
        action = selector.get_action(env)
        _, r, done, info = env.step(action)
        total_r += r
        sat_list.append(info['satisfaction'])

        if adaptive and hasattr(selector, 'update'):
            strategy = selector.select_strategy(env.current_requests)
            selector.update(strategy, r)

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


def run_experiment_v6():
    """运行V6实验: 智能策略选择器"""
    print("=" * 70)
    print("LLM-SRAF V6 - Smart Strategy Selector")
    print("=" * 70)

    methods = {
        'Smart-Selector': lambda e: run_method(e, SmartStrategySelector()),
        'Adaptive-Selector': lambda e: run_method(e, AdaptiveStrategySelector(), adaptive=True),
        'Priority-First': lambda e: run_fixed_strategy(e, priority_first_strategy),
        'Latency-Opt': lambda e: run_fixed_strategy(e, latency_optimized_strategy),
        'Bandwidth-Opt': lambda e: run_fixed_strategy(e, bandwidth_optimized_strategy),
        'Hybrid': lambda e: run_fixed_strategy(e, hybrid_strategy),
    }

    # 多场景测试
    scenarios = {
        'latency_critical': {
            'desc': '时延敏感场景',
            'weights': {'gaming': 4, 'voice_call': 4, 'video_conference': 2}
        },
        'bandwidth_heavy': {
            'desc': '带宽密集场景',
            'weights': {'video_streaming': 4, 'file_transfer': 4, 'live_stream': 2}
        },
        'mixed': {
            'desc': '混合场景',
            'weights': None
        },
        'high_priority': {
            'desc': '高优先级场景',
            'weights': {'gaming': 5, 'video_conference': 5, 'voice_call': 5}
        }
    }

    all_results = {}

    for scenario_name, scenario_info in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_info['desc']}")
        print("=" * 70)

        results = {m: {'rewards': [], 'sats': []} for m in methods}

        for seed in range(5):
            np.random.seed(seed*100)
            random.seed(seed*100)

            env = DecoupledEnv(
                num_sats=15, num_users=60, seed=seed*100,
                service_weights=scenario_info['weights']
            )

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
        smart_r = results['Smart-Selector']['rewards']
        priority_r = results['Priority-First']['rewards']

        t_stat, p_val = scipy_stats.ttest_ind(smart_r, priority_r)
        gain = (np.mean(smart_r) - np.mean(priority_r)) / np.mean(priority_r) * 100

        print(f"\n  Smart-Selector vs Priority-First: {gain:+.2f}% (p={p_val:.4f})")

    # 汇总分析
    print(f"\n{'='*70}")
    print("Cross-Scenario Summary")
    print("=" * 70)

    print(f"\n{'Scenario':20s} {'Best Selector':>15s} {'vs Best Fixed':>15s}")
    print("-" * 55)

    selector_wins = 0
    total_scenarios = len(scenarios)

    for scenario_name, results in all_results.items():
        smart_r = np.mean(results['Smart-Selector']['rewards'])
        adaptive_r = np.mean(results['Adaptive-Selector']['rewards'])
        best_selector = max(smart_r, adaptive_r)
        best_selector_name = 'Smart' if smart_r >= adaptive_r else 'Adaptive'

        # 找最佳固定策略
        fixed_methods = ['Priority-First', 'Latency-Opt', 'Bandwidth-Opt', 'Hybrid']
        best_fixed = max(fixed_methods, key=lambda m: np.mean(results[m]['rewards']))
        best_fixed_r = np.mean(results[best_fixed]['rewards'])

        gain = (best_selector - best_fixed_r) / best_fixed_r * 100

        if gain > 0:
            selector_wins += 1

        print(f"{scenario_name:20s} {best_selector_name:>15s} {gain:+14.2f}%")

    print(f"\n{'='*70}")
    print(f"Selector wins {selector_wins}/{total_scenarios} scenarios")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    results = run_experiment_v6()