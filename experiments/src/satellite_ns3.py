#!/usr/bin/env python3
"""
NS-3 卫星网络仿真 Python 接口
================================

用于 LLM-SRAF 框架的卫星网络仿真包装器

使用方法:
    from satellite_ns3 import SatelliteEnvNS3

    env = SatelliteEnvNS3()
    state = env.reset()

    for step in range(100):
        action = model.predict(state)  # LLM 资源分配决策
        state, reward, done, info = env.step(action)
"""

import os
import sys
import json
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import tempfile


@dataclass
class NS3SatelliteConfig:
    """NS-3 卫星仿真配置"""
    # 星座参数
    num_satellites: int = 100
    num_orbital_planes: int = 10
    sats_per_plane: int = 10
    altitude_km: float = 550.0

    # 用户参数
    num_users: int = 50

    # 链路参数
    bandwidth_mbps: float = 400.0
    max_beams_per_sat: int = 8

    # 仿真参数
    sim_duration: float = 60.0
    time_step_ms: float = 100.0
    random_seed: int = 42

    # NS-3 路径
    ns3_home: str = "/home/AI4SCI/ns-3"
    scratch_dir: str = "scratch/satellite-llm-sraf"


class SatelliteEnvNS3:
    """
    NS-3 卫星网络仿真环境

    兼容 Gym 接口，用于 RL/LLM 训练
    """

    def __init__(self, config: Optional[NS3SatelliteConfig] = None):
        self.config = config or NS3SatelliteConfig()

        # 设置环境变量
        os.environ['NS3_HOME'] = self.config.ns3_home
        os.environ['LD_LIBRARY_PATH'] = f"{self.config.ns3_home}/build/lib"

        # 状态维度
        self.state_dim = 128
        self.action_dim = self.config.num_users

        # 仿真状态
        self.current_step = 0
        self.max_steps = int(self.config.sim_duration * 1000 / self.config.time_step_ms)

        # 结果缓存
        self._results: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        重置仿真环境

        Returns:
            初始状态向量
        """
        if seed is not None:
            self.config.random_seed = seed

        self.current_step = 0
        self._results = {}

        # 返回初始状态
        return self._get_initial_state()

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步仿真

        Args:
            action: 资源分配动作
                - beam_allocation: 波束分配矩阵
                - power_control: 功率控制向量
                - priority: 优先级向量

        Returns:
            state: 新状态
            reward: 奖励值
            done: 是否结束
            info: 附加信息
        """
        self.current_step += 1

        # 运行 NS-3 仿真 (简化版本)
        results = self._run_ns3_step(action)

        # 计算奖励
        reward = self._compute_reward(results)

        # 检查是否结束
        done = self.current_step >= self.max_steps

        # 获取新状态
        state = self._extract_state(results)

        info = {
            'throughput': results.get('throughput', 0.0),
            'latency': results.get('latency', 0.0),
            'sinr': results.get('sinr', 0.0),
        }

        return state, reward, done, info

    def run_full_simulation(self,
                           duration: Optional[float] = None,
                           output_file: Optional[str] = None) -> Dict:
        """
        运行完整的 NS-3 仿真

        Args:
            duration: 仿真时长 (秒)
            output_file: 输出文件路径

        Returns:
            仿真结果字典
        """
        if duration is not None:
            self.config.sim_duration = duration

        # 构建 NS-3 命令
        cmd = [
            os.path.join(self.config.ns3_home, "ns3"),
            "run", "satellite-llm-sraf", "--",
            f"--sats={self.config.num_satellites}",
            f"--users={self.config.num_users}",
            f"--duration={self.config.sim_duration}",
            f"--seed={self.config.random_seed}",
        ]

        # 运行仿真
        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.ns3_home,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout
        except subprocess.TimeoutExpired:
            output = "Simulation timeout"
        except Exception as e:
            output = f"Error: {str(e)}"

        # 解析结果
        results = self._parse_ns3_output(output)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        return results

    def _run_ns3_step(self, action: Dict) -> Dict:
        """运行一步 NS-3 仿真"""
        # 简化版本：返回模拟结果
        # 实际应与 NS-3 进行交互

        np.random.seed(self.config.random_seed + self.current_step)

        return {
            'throughput': np.random.uniform(50, 200, self.config.num_users).tolist(),
            'latency': np.random.uniform(20, 100, self.config.num_users).tolist(),
            'sinr': np.random.uniform(10, 25, self.config.num_users).tolist(),
            'handover_count': np.random.randint(0, 10),
        }

    def _parse_ns3_output(self, output: str) -> Dict:
        """解析 NS-3 输出"""
        results = {
            'raw_output': output,
            'throughput': 0.0,
            'num_satellites': self.config.num_satellites,
            'num_users': self.config.num_users,
        }

        # 解析吞吐量
        for line in output.split('\n'):
            if 'Total throughput' in line:
                try:
                    parts = line.split(':')
                    results['throughput'] = float(parts[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass

        return results

    def _get_initial_state(self) -> np.ndarray:
        """获取初始状态向量"""
        np.random.seed(self.config.random_seed)

        # 卫星状态
        sat_positions = np.random.randn(self.config.num_satellites, 3)
        sat_loads = np.zeros(self.config.num_satellites)

        # 用户状态
        user_positions = np.random.randn(self.config.num_users, 3)
        user_sinr = np.random.uniform(10, 20, self.config.num_users)

        # 组合状态向量
        state = np.concatenate([
            sat_positions.flatten()[:64],
            sat_loads,
            user_positions.flatten()[:32],
            user_sinr,
        ])

        # 补齐/截断到固定维度
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]

        return state.astype(np.float32)

    def _extract_state(self, results: Dict) -> np.ndarray:
        """从结果中提取状态向量"""
        throughput = np.array(results.get('throughput', [0] * self.config.num_users))
        latency = np.array(results.get('latency', [0] * self.config.num_users))
        sinr = np.array(results.get('sinr', [0] * self.config.num_users))

        # 计算统计特征
        features = np.array([
            np.mean(throughput),
            np.std(throughput),
            np.mean(latency),
            np.std(latency),
            np.mean(sinr),
            np.std(sinr),
            results.get('handover_count', 0) / 10.0,
        ])

        # 添加详细特征
        detailed = np.concatenate([
            throughput[:32] / 200.0 if len(throughput) >= 32 else np.pad(throughput, (0, 32 - len(throughput))) / 200.0,
            latency[:32] / 100.0 if len(latency) >= 32 else np.pad(latency, (0, 32 - len(latency))) / 100.0,
            sinr[:32] / 25.0 if len(sinr) >= 32 else np.pad(sinr, (0, 32 - len(sinr))) / 25.0,
        ])

        state = np.concatenate([features, detailed])

        # 补齐
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))

        return state[:self.state_dim].astype(np.float32)

    def _compute_reward(self, results: Dict) -> float:
        """
        计算奖励

        奖励 = 加权吞吐量 - 时延惩罚 - 公平性惩罚
        """
        throughput = np.array(results.get('throughput', [0]))
        latency = np.array(results.get('latency', [100]))

        # 吞吐量奖励 (归一化)
        throughput_reward = np.mean(throughput) / 200.0

        # 时延惩罚
        latency_penalty = np.mean(latency) / 100.0

        # 公平性惩罚 (Jain 公平性指数)
        if len(throughput) > 1 and np.sum(throughput) > 0:
            fairness = (np.sum(throughput) ** 2) / (len(throughput) * np.sum(throughput ** 2))
        else:
            fairness = 1.0

        # 综合奖励
        reward = throughput_reward * 0.5 - latency_penalty * 0.3 + fairness * 0.2

        return float(reward)

    def close(self):
        """关闭环境"""
        pass

    def get_network_state(self) -> Dict:
        """获取网络状态 (用于 LLM 输入)"""
        return {
            'num_satellites': self.config.num_satellites,
            'num_users': self.config.num_users,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
        }


# ==================== 测试代码 ====================

def test_ns3_interface():
    """测试 NS-3 接口"""
    print("=" * 60)
    print("NS-3 卫星网络仿真接口测试")
    print("=" * 60)

    # 创建环境
    config = NS3SatelliteConfig(
        num_satellites=10,
        num_users=5,
        sim_duration=10.0,
    )

    env = SatelliteEnvNS3(config)

    # 测试重置
    print("\n1. 测试重置...")
    state = env.reset()
    print(f"   初始状态维度: {state.shape}")
    print(f"   状态范围: [{state.min():.3f}, {state.max():.3f}]")

    # 测试步进
    print("\n2. 测试步进...")
    action = {
        'beam_allocation': np.random.randint(0, config.num_satellites, config.num_users),
    }

    for i in range(3):
        state, reward, done, info = env.step(action)
        print(f"   Step {i+1}: reward={reward:.4f}, done={done}")

    # 测试完整仿真
    print("\n3. 测试完整仿真...")
    results = env.run_full_simulation(duration=5.0)
    print(f"   吞吐量: {results.get('throughput', 0):.2f} Mbps")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_ns3_interface()