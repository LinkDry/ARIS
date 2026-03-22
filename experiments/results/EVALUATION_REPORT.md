# LLM-SRAF Evaluation Report
Generated: 2026-03-22 13:55:24

## Semantic Understanding

| Metric | Value |
|--------|-------|
| latency_acc | 0.9109 |
| bandwidth_acc | 0.8901 |
| reliability_acc | 0.9244 |
| priority_acc | 0.4709 |
| overall_acc | 0.7991 |

## Resource Allocation Performance

| Metric | Value |
|--------|-------|
| mean_reward | 74.9617 |
| std_reward | 0.8764 |
| mean_throughput | 901.5849 |
| mean_latency_satisfaction | 0.8746 |
| mean_resource_utilization | 0.6902 |

## Inference Latency

| Metric | Value |
|--------|-------|
| mean_latency_ms | 1.25 ms |
| std_latency_ms | 0.29 ms |
| p50_latency_ms | 1.14 ms |
| p95_latency_ms | 2.18 ms |
| p99_latency_ms | 2.33 ms |

## Baseline Comparison

| Method | Mean Reward | Std Reward |
|--------|-------------|------------|
| LLM-SRAF | 75.1133 | 0.9149 |
| Random | 75.2102 | 0.7535 |
| RoundRobin | 74.8578 | 0.8405 |

