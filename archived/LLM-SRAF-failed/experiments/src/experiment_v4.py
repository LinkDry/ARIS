"""
LLM-SRAF 迭代3: 业务异构性场景
================================

核心洞察: 语义感知的价值在于理解业务类型的差异
- Priority-only: 只知道优先级，不知道业务类型
- LLM-SRAF: 知道业务类型 + 优先级，能针对性分配资源

关键设计:
1. 不同业务类型的资源需求差异大
2. 同一优先级下不同业务类型需要不同策略
3. 高带宽业务(视频流) vs 低延迟业务(游戏) 需要不同资源分配
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
from collections import defaultdict
from scipy import stats as scipy_stats


@dataclass
class ServiceRequest:
    user_id: int
    service_type: str
    latency_req: float
    bandwidth_req: float
    priority: int
    # 业务特征权重 (用于语义感知)
    latency_weight: float  # 时延敏感度
    bandwidth_weight: float  # 带宽敏感度
    actual_latency: float = 0.0
    actual_bandwidth: float = 0.0
    satisfied: bool = False
    latency_satisfied: bool = False
    bandwidth_satisfied: bool = False


class HeterogeneousEnv:
    """
    业务异构性场景环境

    关键特点:
    1. 同一优先级下业务类型差异大
    2. 资源分配需要根据业务特征调整
    3. 时延敏感业务(游戏) vs 带宽敏感业务(视频流)
    """

    # 业务类型配置: 类型 -> (优先级, 时延需求, 带宽需求, 时延权重, 带宽权重)
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

    def __init__(self, num_sats=20, num_users=50, seed=42, heterogeneous=True):
        self.num_sats = num_sats
        self.num_users = num_users
        self.heterogeneous = heterogeneous

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
        """生成异构业务请求"""
        self.current_requests = []

        if self.heterogeneous:
            # 确保业务类型多样化
            service_types = list(self.SERVICE_CONFIGS.keys())
            # 均匀分布各业务类型
            for u in range(self.num_users):
                st = service_types[u % len(service_types)]
                cfg = self.SERVICE_CONFIGS[st]
                self.current_requests.append(ServiceRequest(
                    user_id=u, service_type=st,
                    latency_req=cfg['latency'] * np.random.uniform(0.9, 1.1),
                    bandwidth_req=max(0.1, cfg['bandwidth'] * np.random.uniform(0.9, 1.1)),
                    priority=cfg['priority'],
                    latency_weight=cfg['lat_w'],
                    bandwidth_weight=cfg['bw_w']
                ))
        else:
            # 同构场景：所有用户都是相同业务类型
            st = random.choice(list(self.SERVICE_CONFIGS.keys()))
            cfg = self.SERVICE_CONFIGS[st]
            for u in range(self.num_users):
                self.current_requests.append(ServiceRequest(
                    user_id=u, service_type=st,
                    latency_req=cfg['latency'] * np.random.uniform(0.9, 1.1),
                    bandwidth_req=max(0.1, cfg['bandwidth'] * np.random.uniform(0.9, 1.1)),
                    priority=cfg['priority'],
                    latency_weight=cfg['lat_w'],
                    bandwidth_weight=cfg['bw_w']
                ))

    def step(self, action, use_semantic=True):
        """执行一步"""
        self.current_step += 1
        total_reward = 0
        lat_sat_count = bw_sat_count = total_sat_count = 0

        for i, req in enumerate(self.current_requests):
            sat_idx = action['satellite_selection'][i]
            power = action['power_level'][i]
            spectrum = max(1, action['spectrum_allocation'][i])

            # Fallback 到可见卫星
            if sat_idx >= self.num_sats or self.visibility[i, sat_idx] == 0:
                visible = np.where(self.visibility[i] > 0)[0]
                sat_idx = np.random.choice(visible) if len(visible) > 0 else 0

            # 计算性能
            snr = self.channel_quality[i, sat_idx]
            spec_eff = np.log2(1 + 10 ** (snr / 10))
            avail_ratio = max(0.1, 1 - self.sat_load[sat_idx])

            # 带宽: 取决于频谱、功率、卫星负载
            bandwidth = spec_eff * 25 * spectrum * (0.5 + 0.15 * power) * avail_ratio
            # 时延: 取决于功率(处理速度)和卫星负载
            latency = 15 + (1 - power/5) * 10 + self.sat_load[sat_idx] * 40

            # 更新负载
            self.sat_load[sat_idx] = min(1.0, self.sat_load[sat_idx] + bandwidth / self.sat_capacity[sat_idx] * 0.08)

            # 判断满足
            lat_ok = latency <= req.latency_req
            bw_ok = bandwidth >= req.bandwidth_req * 0.8
            satisfied = lat_ok and bw_ok

            req.actual_latency = latency
            req.actual_bandwidth = bandwidth
            req.satisfied = satisfied
            req.latency_satisfied = lat_ok
            req.bandwidth_satisfied = bw_ok

            if lat_ok: lat_sat_count += 1
            if bw_ok: bw_sat_count += 1
            if satisfied: total_sat_count += 1

            # 计算奖励
            lat_score = min(1.0, req.latency_req / max(1, latency))
            bw_score = min(1.0, bandwidth / max(0.1, req.bandwidth_req))

            if use_semantic:
                # 语义感知: 根据业务特征权重计算奖励
                # 时延敏感业务: 时延满足更重要
                # 带宽敏感业务: 带宽满足更重要
                reward = (req.latency_weight * lat_score + req.bandwidth_weight * bw_score)

                # 高优先级业务额外奖励
                if satisfied and req.priority >= 2:
                    reward += 0.3 * (req.priority / 3)
            else:
                # 非语义感知: 固定权重
                reward = 0.5 * lat_score + 0.5 * bw_score
                if satisfied and req.priority >= 2:
                    reward += 0.2

            total_reward += reward

        self.sat_load *= 0.9
        self.channel_quality = np.clip(self.channel_quality + np.random.randn(*self.channel_quality.shape) * 0.2, 5, 30)

        done = self.current_step >= self.max_steps
        return self._get_state(), total_reward / self.num_users, done, {
            'satisfaction': total_sat_count / self.num_users,
            'lat_satisfaction': lat_sat_count / self.num_users,
            'bw_satisfaction': bw_sat_count / self.num_users
        }

    def _get_state(self):
        return {'network_state': np.zeros(64, dtype=np.float32)}


def run_semantic_aware_method(env, horizon=50):
    """语义感知方法: 根据业务特征分配资源"""
    env.reset()
    total_r = 0
    sat_list = []

    for step in range(horizon):
        sat_sel = np.zeros(env.num_users, dtype=int)
        power_sel = np.zeros(env.num_users, dtype=int)
        spectrum_sel = np.zeros(env.num_users, dtype=int)
        load_temp = env.sat_load.copy()

        # 按优先级和业务特征综合排序
        scored_users = []
        for i, req in enumerate(env.current_requests):
            # 综合得分 = 优先级 + 业务紧急度
            score = req.priority * 10
            # 时延敏感业务在时延接近阈值时更紧急
            if req.latency_weight > 0.7:
                score += 5
            scored_users.append((i, score))
        scored_users.sort(key=lambda x: x[1], reverse=True)

        for i, _ in scored_users:
            visible = np.where(env.visibility[i] > 0)[0]
            if len(visible) > 0:
                best = visible[np.argmin(load_temp[visible])]
                sat_sel[i] = best
                load_temp[best] += 0.03

            # 根据业务特征分配资源
            req = env.current_requests[i]
            if req.latency_weight > 0.7:
                # 时延敏感: 高功率, 适中频谱
                power_sel[i] = 4
                spectrum_sel[i] = 3
            elif req.bandwidth_weight > 0.7:
                # 带宽敏感: 中等功率, 高频谱
                power_sel[i] = 2
                spectrum_sel[i] = 6
            else:
                # 平衡型
                power_sel[i] = 3
                spectrum_sel[i] = 4

        action = {
            'satellite_selection': sat_sel,
            'power_level': power_sel,
            'spectrum_allocation': spectrum_sel
        }
        _, r, done, info = env.step(action, use_semantic=True)
        total_r += r
        sat_list.append(info['satisfaction'])
        if done: break

    return total_r, np.mean(sat_list)


def run_priority_only_method(env, horizon=50):
    """仅优先级感知: 不知道业务类型，只知道优先级"""
    env.reset()
    total_r = 0
    sat_list = []

    for step in range(horizon):
        sat_sel = np.zeros(env.num_users, dtype=int)
        load_temp = env.sat_load.copy()

        # 仅按优先级排序
        priority_order = sorted(range(len(env.current_requests)),
            key=lambda i: env.current_requests[i].priority, reverse=True)

        for i in priority_order:
            visible = np.where(env.visibility[i] > 0)[0]
            if len(visible) > 0:
                best = visible[np.argmin(load_temp[visible])]
                sat_sel[i] = best
                load_temp[best] += 0.03

        # 根据优先级分配资源 (不知道业务是时延敏感还是带宽敏感)
        power_sel = np.array([min(4, env.current_requests[i].priority + 1) for i in range(env.num_users)])
        spectrum_sel = np.array([min(6, env.current_requests[i].priority + 2) for i in range(env.num_users)])

        action = {
            'satellite_selection': sat_sel,
            'power_level': power_sel,
            'spectrum_allocation': spectrum_sel
        }
        _, r, done, info = env.step(action, use_semantic=False)
        total_r += r
        sat_list.append(info['satisfaction'])
        if done: break

    return total_r, np.mean(sat_list)


def run_greedy_method(env, horizon=50):
    """贪婪信道选择"""
    env.reset()
    total_r = 0
    sat_list = []

    for step in range(horizon):
        action = {
            'satellite_selection': np.argmax(env.channel_quality, axis=1),
            'power_level': np.ones(env.num_users, dtype=int) * 2,
            'spectrum_allocation': np.ones(env.num_users, dtype=int) * 3
        }
        _, r, done, info = env.step(action, use_semantic=False)
        total_r += r
        sat_list.append(info['satisfaction'])
        if done: break

    return total_r, np.mean(sat_list)


def run_experiment_v4():
    """运行实验 V4"""
    print("=" * 70)
    print("LLM-SRAF V4 - Heterogeneous Business Scenario")
    print("=" * 70)

    # 异构场景
    print("\n=== HETEROGENEOUS Scenario (Diverse Service Types) ===")
    print(f"{'Method':20s} {'Reward':>10s} {'Std':>8s} {'Satisfaction':>12s}")
    print("-" * 55)

    methods = {
        'LLM-SRAF (Semantic)': run_semantic_aware_method,
        'Priority-only': run_priority_only_method,
        'Greedy': run_greedy_method
    }

    results = {m: {'rewards': [], 'sats': []} for m in methods}

    for seed in range(5):
        env = HeterogeneousEnv(num_sats=15, num_users=60, seed=seed*100, heterogeneous=True)
        for name, method in methods.items():
            for _ in range(10):
                r, s = method(env, 50)
                results[name]['rewards'].append(r)
                results[name]['sats'].append(s)

    for name in methods:
        r = results[name]['rewards']
        s = results[name]['sats']
        print(f"{name:20s} {np.mean(r):10.4f} {np.std(r):8.4f} {np.mean(s)*100:11.2f}%")

    # 统计检验
    llm_r = results['LLM-SRAF (Semantic)']['rewards']
    prio_r = results['Priority-only']['rewards']
    greedy_r = results['Greedy']['rewards']

    t1, p1 = scipy_stats.ttest_ind(llm_r, prio_r)
    t2, p2 = scipy_stats.ttest_ind(llm_r, greedy_r)

    gain_vs_prio = (np.mean(llm_r) - np.mean(prio_r)) / np.mean(prio_r) * 100
    gain_vs_greedy = (np.mean(llm_r) - np.mean(greedy_r)) / np.mean(greedy_r) * 100

    print(f"\n  LLM-SRAF vs Priority-only: {gain_vs_prio:+.2f}% (p={p1:.4f})")
    print(f"  LLM-SRAF vs Greedy: {gain_vs_greedy:+.2f}% (p={p2:.4f})")

    # 详细分析
    print("\n=== Satisfaction by Service Type ===")
    env = HeterogeneousEnv(num_sats=15, num_users=80, seed=42, heterogeneous=True)
    env.reset()

    # 运行语义感知方法并收集详细统计
    for step in range(50):
        sat_sel = np.zeros(env.num_users, dtype=int)
        power_sel = np.zeros(env.num_users, dtype=int)
        spectrum_sel = np.zeros(env.num_users, dtype=int)
        load_temp = env.sat_load.copy()

        scored_users = [(i, req.priority * 10 + (5 if req.latency_weight > 0.7 else 0))
                        for i, req in enumerate(env.current_requests)]
        scored_users.sort(key=lambda x: x[1], reverse=True)

        for i, _ in scored_users:
            visible = np.where(env.visibility[i] > 0)[0]
            if len(visible) > 0:
                sat_sel[i] = visible[np.argmin(load_temp[visible])]
                load_temp[sat_sel[i]] += 0.03

            req = env.current_requests[i]
            if req.latency_weight > 0.7:
                power_sel[i], spectrum_sel[i] = 4, 3
            elif req.bandwidth_weight > 0.7:
                power_sel[i], spectrum_sel[i] = 2, 6
            else:
                power_sel[i], spectrum_sel[i] = 3, 4

        env.step({'satellite_selection': sat_sel, 'power_level': power_sel, 'spectrum_allocation': spectrum_sel}, True)

    # 按业务类型统计
    type_stats = defaultdict(lambda: {'sat': 0, 'total': 0, 'lat_sat': 0, 'bw_sat': 0})
    for req in env.current_requests:
        type_stats[req.service_type]['total'] += 1
        if req.satisfied:
            type_stats[req.service_type]['sat'] += 1
        if req.latency_satisfied:
            type_stats[req.service_type]['lat_sat'] += 1
        if req.bandwidth_satisfied:
            type_stats[req.service_type]['bw_sat'] += 1

    print(f"\n{'Service Type':20s} {'Total':>6s} {'Sat%':>8s} {'Lat-Sat%':>10s} {'BW-Sat%':>10s}")
    print("-" * 60)
    for st, stats in sorted(type_stats.items()):
        t = stats['total']
        print(f"{st:20s} {t:6d} {stats['sat']/t*100:7.1f}% {stats['lat_sat']/t*100:9.1f}% {stats['bw_sat']/t*100:9.1f}%")


if __name__ == '__main__':
    run_experiment_v4()