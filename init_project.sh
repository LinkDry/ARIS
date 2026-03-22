#!/bin/bash
# LLM-SRAF 项目初始化脚本
# 用法: bash init_project.sh

set -e

# 配置
PROJECT_NAME="P1-LLM卫星语义资源分配"
PROJECT_DIR="e:/Projects/AI4Sci/projects/${PROJECT_NAME}"
CONDA_ENV="llm-sraf"

echo "=== LLM-SRAF 项目初始化 ==="

# 1. 创建项目目录结构
echo "[1/4] 创建项目目录结构..."
mkdir -p "${PROJECT_DIR}"/experiments/{configs,scripts,results,logs,models,checkpoints}
mkdir -p "${PROJECT_DIR}"/paper/{sections,figures}
mkdir -p "${PROJECT_DIR}"/refine-logs
mkdir -p "${PROJECT_DIR}"/data/{raw,processed}

# 2. 复制 ARIS 产出文件
echo "[2/4] 复制 ARIS 产出文件..."
cp project_state/stage_1_literature/LITERATURE_REPORT.md "${PROJECT_DIR}/01-调研与选题.md"
cp project_state/stage_2_ideas/IDEA_REPORT.md "${PROJECT_DIR}/03-Idea与创新点.md"
cp project_state/stage_3_novelty/NOVELTY_REPORT.md "${PROJECT_DIR}/02-新颖性验证.md"
cp project_state/stage_4_proposal/FINAL_PROPOSAL.md "${PROJECT_DIR}/04-方法设计.md"
cp project_state/stage_5_experiment/EXPERIMENT_PLAN.md "${PROJECT_DIR}/05-实验设计.md"

# 3. 创建 README
echo "[3/4] 创建项目 README..."
cat > "${PROJECT_DIR}/README.md" << 'EOF'
---
title: LLM增强卫星语义资源分配框架
date: 2026-03-22
stage: 方法设计
target_venue: IEEE JSAC / TWC
deadline: TBD
status: 进行中
---

## 项目概述

利用大语言模型的语义理解能力，将用户自然语言描述的业务需求转化为卫星网络资源分配策略。

## 当前进度

- [x] 阶段1：调研与选题
- [x] 阶段2：Idea生成与验证
- [x] 阶段3：新颖性验证
- [x] 阶段4：方法设计
- [ ] 阶段5：实验实现
- [ ] 阶段6：论文写作

## 关键文件

- [方法设计](04-方法设计.md)
- [实验设计](05-实验设计.md)
- [实验代码](experiments/)

## 核心贡献

1. LLM首次用于卫星资源管理的语义理解环节
2. 语义感知的资源分配框架
3. 自然语言交互的资源管理系统
EOF

# 4. 创建配置文件
echo "[4/4] 创建配置文件..."
cat > "${PROJECT_DIR}/experiments/configs/model_config.yaml" << 'EOF'
# LLM-SRAF 模型配置
llm:
  model_name: "Qwen/Qwen2-7B"
  max_length: 512
  device: "cuda"

semantic_encoder:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.1

fusion:
  type: "gated"  # gated, attention, concat
  hidden_dim: 256

actor:
  state_dim: 256
  action_dim: 16
  hidden_dims: [128, 64]

critic:
  state_dim: 256
  action_dim: 16
  hidden_dims: [128, 64]
EOF

cat > "${PROJECT_DIR}/experiments/configs/train_config.yaml" << 'EOF'
# 训练配置
seed: 42
device: "cuda"

pretrain:
  epochs: 50
  batch_size: 32
  lr: 1e-4
  weight_decay: 1e-5

rl:
  algorithm: "PPO"
  total_timesteps: 1000000
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95

logging:
  use_wandb: true
  wandb_project: "llm-sraf"
  tensorboard: true
  save_interval: 1000
EOF

echo ""
echo "=== 初始化完成 ==="
echo "项目目录: ${PROJECT_DIR}"
echo ""
echo "下一步:"
echo "  1. conda activate ${CONDA_ENV}"
echo "  2. cd ${PROJECT_DIR}/experiments"
echo "  3. 开始编写实验代码"