# 归档说明：LLM-SRAF 项目

**归档时间**: 2026-03-22
**归档原因**: 研究假设被证伪，项目失败

---

## 项目概述

**原始假设**: LLM 的语义理解能力可以提升卫星网络资源分配效率

**实际结论**:
- 语义感知在结构化资源分配问题中价值有限
- 简单的 Priority-First 启发式方法优于复杂方法
- 该问题不适合使用语义感知方法

---

## 失败原因分析

### 核心问题

| 问题 | 说明 |
|------|------|
| 问题类型错误 | 资源分配是结构化决策问题，不需要语义理解 |
| 信息冗余 | 优先级参数已充分表达需求，语义嵌入是冗余 |
| LLM不必要 | 简单规则方法表现更优 |

### 教训

1. **语义必要性检验**: 选择研究方向前，必须检验是否真正需要语义理解
2. **简单基线优先**: 先验证简单方法，再尝试复杂方法
3. **问题-方法匹配**: 方法复杂度需与问题复杂度匹配

---

## 实验记录

### 6轮迭代结果

| 迭代 | 方法 | vs Priority-First | 结论 |
|------|------|------------------|------|
| 1-3 | LLM-SRAF直接耦合 | -3.3% ~ -4.35% | 失败 |
| 4 | 解耦架构 | -0.1% ~ +0.13% | 持平 |
| 5 | 智能选择器 | -1.78% ~ -2.23% | 更差 |
| 6 | DRL验证 | Priority-First最优 | 确认简单方法最优 |

---

## 文件结构

```
LLM-SRAF-failed/
├── experiments/                    # 实验代码
│   ├── src/
│   │   ├── experiment_v3.py       # 迭代2
│   │   ├── experiment_v4.py       # 迭代3
│   │   ├── experiment_v5_decoupled.py  # 迭代4
│   │   ├── experiment_v6_smart_selector.py  # 迭代5
│   │   └── experiment_quick_validation.py  # 迭代6
│   └── results/
│       ├── FINAL_ITERATION_REPORT.md
│       ├── NEGATIVE_RESULT_ANALYSIS.md
│       └── DRL_VALIDATION_RESULTS.md
└── project_state/                  # 研究流水线状态（旧）
```

---

## 后续方向

新的研究方向 **SatFault-LLM** 已在 `projects/SatFault-LLM/` 启动：
- 卫星网络多源异构日志融合的故障根因智能定位
- 该方向真正需要LLM的语义理解能力

---

*归档人: ARIS Framework*
*教训已记录，避免重复犯错*