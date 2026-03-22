"""
LLM-SRAF Evaluation Script
===========================

评估指标:
- 语义理解准确率
- 系统吞吐量
- 时延满足率
- 资源利用率
- 端到端推理时延
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import ModelConfig, LLMSRAF, create_model
from data import (
    SemanticPairDataset,
    SatelliteEnv,
    generate_synthetic_data,
    save_dataset,
)


class Evaluator:
    """LLM-SRAF 评估器"""

    def __init__(self, model: LLMSRAF, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_semantic_understanding(self,
                                        dataloader,
                                        verbose: bool = True) -> Dict[str, float]:
        """
        评估语义理解能力

        Returns:
            准确率指标字典
        """
        correct = {
            'latency': 0,
            'bandwidth': 0,
            'reliability': 0,
            'priority': 0,
        }
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating semantic", disable=not verbose):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model.semantic_module(input_ids, attention_mask)

                # 计算各分类准确率
                for key in ['latency', 'bandwidth', 'reliability']:
                    pred = outputs[f'{key}_class'].argmax(dim=1)
                    labels = batch[f'{key}_class'].to(self.device)
                    correct[key] += (pred == labels).sum().item()

                # 优先级准确率
                pred_priority = outputs['priority'].argmax(dim=1)
                labels_priority = batch['priority'].to(self.device)
                correct['priority'] += (pred_priority == labels_priority).sum().item()

                total += input_ids.size(0)

        metrics = {
            'latency_acc': correct['latency'] / total,
            'bandwidth_acc': correct['bandwidth'] / total,
            'reliability_acc': correct['reliability'] / total,
            'priority_acc': correct['priority'] / total,
            'overall_acc': sum(correct.values()) / (total * 4),
        }

        return metrics

    def evaluate_resource_allocation(self,
                                     env: SatelliteEnv,
                                     num_episodes: int = 100,
                                     verbose: bool = True) -> Dict[str, float]:
        """
        评估资源分配性能

        Returns:
            性能指标字典
        """
        throughput_list = []
        latency_satisfaction_list = []
        resource_utilization_list = []
        rewards = []

        with torch.no_grad():
            for ep in tqdm(range(num_episodes), desc="Evaluating RL", disable=not verbose):
                state = env.reset()
                episode_reward = 0
                episode_throughput = []
                episode_latency = []

                for step in range(100):
                    # 获取状态向量并确保正确维度
                    network_state_vec = env.get_network_state_vector()
                    orbit_info_vec = env.get_orbit_info_vector()

                    # Ensure correct dimensions
                    if len(network_state_vec) < 64:
                        network_state_vec = np.pad(network_state_vec, (0, 64 - len(network_state_vec)))
                    network_state_vec = network_state_vec[:64]

                    if len(orbit_info_vec) < 32:
                        orbit_info_vec = np.pad(orbit_info_vec, (0, 32 - len(orbit_info_vec)))
                    orbit_info_vec = orbit_info_vec[:32]

                    network_state = torch.tensor(
                        network_state_vec,
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

                    orbit_info = torch.tensor(
                        orbit_info_vec,
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

                    # 假输入 (实际应使用真实业务描述)
                    input_ids = torch.zeros(1, 128, dtype=torch.long).to(self.device)
                    attention_mask = torch.ones(1, 128, dtype=torch.long).to(self.device)

                    # 获取动作
                    outputs = self.model(input_ids, attention_mask, network_state, orbit_info)
                    actions, _, _ = self.model.decision_module.select_action(outputs['fused'])

                    # 执行动作
                    next_state, reward, done, info = env.step(actions)

                    episode_reward += reward
                    episode_throughput.append(info.get('throughput', 0))
                    episode_latency.append(info.get('latency_satisfaction', 0))

                    if done:
                        break
                    state = next_state

                rewards.append(episode_reward)
                throughput_list.append(np.mean(episode_throughput))
                latency_satisfaction_list.append(np.mean(episode_latency))
                resource_utilization_list.append(info.get('resource_utilization', 0.5))

        metrics = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_throughput': np.mean(throughput_list),
            'mean_latency_satisfaction': np.mean(latency_satisfaction_list),
            'mean_resource_utilization': np.mean(resource_utilization_list),
        }

        return metrics

    def evaluate_inference_latency(self,
                                   num_samples: int = 1000,
                                   warmup: int = 100) -> Dict[str, float]:
        """
        评估推理时延

        Returns:
            时延指标字典 (ms)
        """
        # 准备输入
        input_ids = torch.zeros(1, 128, dtype=torch.long).to(self.device)
        attention_mask = torch.ones(1, 128, dtype=torch.long).to(self.device)
        network_state = torch.randn(1, 64, dtype=torch.float32).to(self.device)
        orbit_info = torch.randn(1, 32, dtype=torch.float32).to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_ids, attention_mask, network_state, orbit_info)

        # 同步
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # 测量
        latencies = []
        with torch.no_grad():
            for _ in range(num_samples):
                start = time.perf_counter()

                _ = self.model(input_ids, attention_mask, network_state, orbit_info)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        metrics = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
        }

        return metrics


def run_ablation_study(model: LLMSRAF,
                       env: SatelliteEnv,
                       device: torch.device,
                       num_episodes: int = 50) -> Dict[str, float]:
    """
    消融实验: 评估各模块贡献

    Returns:
        消融结果字典
    """
    evaluator = Evaluator(model, device)

    results = {}

    # 完整模型
    print("Evaluating full model...")
    results['full_model'] = evaluator.evaluate_resource_allocation(env, num_episodes)

    # 去除语义模态 (使用零向量)
    print("Evaluating without semantic...")
    # 需要修改模型以支持消融

    # 单独 RL (无语义)
    print("Evaluating RL-only baseline...")
    # 使用随机语义向量

    return results


def compare_with_baselines(model: LLMSRAF,
                          env: SatelliteEnv,
                          device: torch.device,
                          num_episodes: int = 100) -> Dict[str, Dict]:
    """
    与基线方法对比

    Returns:
        对比结果字典
    """
    results = {}

    # LLM-SRAF
    print("Evaluating LLM-SRAF...")
    evaluator = Evaluator(model, device)
    results['LLM-SRAF'] = evaluator.evaluate_resource_allocation(env, num_episodes)

    # Random policy
    print("Evaluating Random policy...")
    random_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for _ in range(100):
            action = {
                'beam': np.random.randint(0, 8),
                'spectrum': np.random.randint(0, 16),
                'power': np.random.randint(0, 5),
                'priority': np.random.randint(0, 4),
            }
            _, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        random_rewards.append(episode_reward)

    results['Random'] = {
        'mean_reward': np.mean(random_rewards),
        'std_reward': np.std(random_rewards),
    }

    # Round-robin policy
    print("Evaluating Round-robin policy...")
    rr_rewards = []
    beam_idx = 0
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for _ in range(100):
            action = {
                'beam': beam_idx % 8,
                'spectrum': beam_idx % 16,
                'power': 2,
                'priority': 2,
            }
            beam_idx += 1
            _, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        rr_rewards.append(episode_reward)

    results['RoundRobin'] = {
        'mean_reward': np.mean(rr_rewards),
        'std_reward': np.std(rr_rewards),
    }

    return results


def generate_report(results: Dict, output_path: str):
    """生成评估报告"""
    report = []
    report.append("# LLM-SRAF Evaluation Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 语义理解指标
    if 'semantic' in results:
        report.append("## Semantic Understanding\n\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        for key, value in results['semantic'].items():
            report.append(f"| {key} | {value:.4f} |\n")
        report.append("\n")

    # 资源分配性能
    if 'resource_allocation' in results:
        report.append("## Resource Allocation Performance\n\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        for key, value in results['resource_allocation'].items():
            report.append(f"| {key} | {value:.4f} |\n")
        report.append("\n")

    # 推理时延
    if 'latency' in results:
        report.append("## Inference Latency\n\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        for key, value in results['latency'].items():
            report.append(f"| {key} | {value:.2f} ms |\n")
        report.append("\n")

    # 基线对比
    if 'baselines' in results:
        report.append("## Baseline Comparison\n\n")
        report.append("| Method | Mean Reward | Std Reward |\n")
        report.append("|--------|-------------|------------|\n")
        for method, metrics in results['baselines'].items():
            report.append(f"| {method} | {metrics['mean_reward']:.4f} | {metrics['std_reward']:.4f} |\n")
        report.append("\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM-SRAF')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--data_path', type=str, default=None, help='Path to test data')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes for RL eval')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}...")
    config = ModelConfig()
    model = create_model(config).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create evaluator
    evaluator = Evaluator(model, device)

    results = {}

    # 1. Evaluate semantic understanding
    print("\n[1/4] Evaluating semantic understanding...")
    test_data = SemanticPairDataset(data_path=args.data_path) if args.data_path else SemanticPairDataset()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    results['semantic'] = evaluator.evaluate_semantic_understanding(test_loader)

    # 2. Evaluate resource allocation
    print("\n[2/4] Evaluating resource allocation...")
    env = SatelliteEnv(num_users=100)
    results['resource_allocation'] = evaluator.evaluate_resource_allocation(env, args.num_episodes)

    # 3. Evaluate inference latency
    print("\n[3/4] Evaluating inference latency...")
    results['latency'] = evaluator.evaluate_inference_latency()

    # 4. Compare with baselines
    print("\n[4/4] Comparing with baselines...")
    results['baselines'] = compare_with_baselines(model, env, device, args.num_episodes)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print("\nSemantic Understanding:")
    for key, value in results['semantic'].items():
        print(f"  {key}: {value:.4f}")

    print("\nResource Allocation:")
    for key, value in results['resource_allocation'].items():
        print(f"  {key}: {value:.4f}")

    print("\nInference Latency:")
    for key, value in results['latency'].items():
        print(f"  {key}: {value:.2f} ms")

    print("\nBaseline Comparison:")
    for method, metrics in results['baselines'].items():
        print(f"  {method}: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in value.items()
                }
        json.dump(serializable_results, f, indent=2)

    # Generate report
    report_path = os.path.join(args.output_dir, 'EVALUATION_REPORT.md')
    generate_report(results, report_path)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()