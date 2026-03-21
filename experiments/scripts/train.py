"""
LLM-SRAF Training Scripts
=========================

- train_semantic.py: Train semantic understanding module
- train_rl.py: Train RL-based resource decision module
- evaluate.py: Evaluation script
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import ModelConfig, LLMSRAF, create_model
from data import (
    SemanticPairDataset,
    SatelliteEnv,
    create_dataloader,
    generate_synthetic_data,
    save_dataset,
)


def train_semantic(args):
    """
    Train semantic understanding module with supervised learning
    """
    print("=" * 60)
    print("Training Semantic Understanding Module")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    config = ModelConfig()
    model = create_model(config).to(device)

    # Load data
    print("Loading dataset...")
    train_dataset = SemanticPairDataset(data_path=args.train_data) if args.train_data else SemanticPairDataset()
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)

    val_dataset = SemanticPairDataset(data_path=args.val_data) if args.val_data else SemanticPairDataset()
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # Training loop
    best_val_acc = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            latency_labels = batch['latency_class'].to(device)
            bandwidth_labels = batch['bandwidth_class'].to(device)
            reliability_labels = batch['reliability_class'].to(device)
            priority_labels = batch['priority'].to(device)
            resource_targets = batch['resource_vector'].to(device)

            # Forward pass
            outputs = model.semantic_module(input_ids, attention_mask)

            # Calculate losses
            loss = 0

            # Classification losses
            loss += ce_loss(outputs['latency_class'], latency_labels)
            loss += ce_loss(outputs['bandwidth_class'], bandwidth_labels)
            loss += ce_loss(outputs['reliability_class'], reliability_labels)
            loss += ce_loss(outputs['priority'], priority_labels)

            # Regression loss
            loss += mse_loss(outputs['resource_vec'], resource_targets) * 0.5

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        val_acc = evaluate_semantic(model, val_loader, device)

        # Logging
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/accuracy', val_acc, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))

        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))

    writer.close()
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

    return model


def evaluate_semantic(model, dataloader, device):
    """Evaluate semantic understanding accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['latency_class'].to(device)

            outputs = model.semantic_module(input_ids, attention_mask)
            predictions = outputs['latency_class'].argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_rl(args):
    """
    Train resource decision module with PPO
    """
    print("=" * 60)
    print("Training Resource Decision Module with PPO")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    config = ModelConfig()
    model = create_model(config).to(device)

    # Load pretrained semantic module
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.semantic_module.load_state_dict(
                {k.replace('semantic_module.', ''): v
                 for k, v in checkpoint['model_state_dict'].items()
                 if k.startswith('semantic_module.')}
            )
        else:
            model.load_state_dict(checkpoint)

    # Create environment
    env = SatelliteEnv(num_users=100)

    # PPO parameters
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_coef = 0.5

    # Optimizer
    optimizer = optim.Adam(model.decision_module.parameters(), lr=args.rl_lr)

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # Training loop
    global_step = 0
    best_reward = 0

    for episode in range(args.num_episodes):
        # Collect trajectories
        trajectories = collect_trajectories(model, env, device, args.horizon)

        # Update policy
        losses = update_ppo(model, optimizer, trajectories, device,
                           gamma, lambda_gae, clip_epsilon,
                           entropy_coef, value_coef, args.ppo_epochs)

        # Logging
        mean_reward = np.mean([t['reward'] for t in trajectories])
        writer.add_scalar('train/reward', mean_reward, episode)
        writer.add_scalar('train/policy_loss', losses['policy_loss'], episode)
        writer.add_scalar('train/value_loss', losses['value_loss'], episode)

        if episode % args.log_interval == 0:
            print(f"Episode {episode}: Reward={mean_reward:.4f}")

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(model.state_dict(),
                      os.path.join(args.output_dir, 'best_rl_model.pt'))

        global_step += args.horizon

    writer.close()
    print(f"RL training complete. Best reward: {best_reward:.4f}")

    return model


def collect_trajectories(model, env, device, horizon):
    """Collect trajectories from environment"""
    trajectories = []
    state = env.reset()

    for _ in range(horizon):
        # Get state vectors
        network_state = torch.tensor(env.get_network_state_vector(),
                                     dtype=torch.float32).unsqueeze(0).to(device)
        orbit_info = torch.tensor(env.get_orbit_info_vector(),
                                  dtype=torch.float32).unsqueeze(0).to(device)

        # Dummy input for semantic (will be replaced with actual descriptions)
        input_ids = torch.zeros(1, 128, dtype=torch.long).to(device)
        attention_mask = torch.ones(1, 128, dtype=torch.long).to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask, network_state, orbit_info)

        # Sample action
        actions, log_prob, value = model.decision_module.select_action(outputs['fused'])

        # Step environment
        next_state, reward, done, info = env.step(actions)

        # Store transition
        trajectories.append({
            'state': outputs['fused'].detach(),
            'action': actions,
            'log_prob': log_prob.detach(),
            'value': value.detach(),
            'reward': reward,
            'done': done,
        })

        if done:
            state = env.reset()
        else:
            state = next_state

    return trajectories


def update_ppo(model, optimizer, trajectories, device,
               gamma, lambda_gae, clip_epsilon,
               entropy_coef, value_coef, epochs):
    """Update policy with PPO"""
    # Calculate returns and advantages
    returns = []
    advantages = []

    with torch.no_grad():
        for i, traj in enumerate(trajectories):
            # Calculate returns
            G = 0
            for j in range(len(trajectories) - 1, i - 1, -1):
                G = trajectories[j]['reward'] + gamma * G * (1 - trajectories[j]['done'])
            returns.append(G)

            # Calculate advantage (simplified)
            advantage = G - traj['value'].item()
            advantages.append(advantage)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Stack states
    states = torch.cat([t['state'] for t in trajectories])

    losses = {'policy_loss': 0, 'value_loss': 0}

    for _ in range(epochs):
        # Forward pass
        action_probs, values = model.decision_module(states)
        values = values.squeeze()

        # Policy loss (simplified)
        policy_loss = -torch.mean(advantages)

        # Value loss
        value_loss = nn.MSELoss()(values, returns)

        # Total loss
        loss = policy_loss + value_coef * value_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses['policy_loss'] += policy_loss.item()
        losses['value_loss'] += value_loss.item()

    losses['policy_loss'] /= epochs
    losses['value_loss'] /= epochs

    return losses


def main():
    parser = argparse.ArgumentParser(description='Train LLM-SRAF')

    # Mode
    parser.add_argument('--mode', type=str, default='semantic',
                       choices=['semantic', 'rl', 'all'],
                       help='Training mode')

    # Data
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rl_lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # RL
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--horizon', type=int, default=256)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--pretrained', type=str, default=None)

    # Output
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='logs/tensorboard')
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    if args.mode == 'semantic':
        train_semantic(args)
    elif args.mode == 'rl':
        train_rl(args)
    else:
        train_semantic(args)
        args.pretrained = os.path.join(args.output_dir, 'best_model.pt')
        train_rl(args)


if __name__ == '__main__':
    main()