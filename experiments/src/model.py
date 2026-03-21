"""
LLM-SRAF Model Components
=========================

Core modules:
1. SemanticUnderstandingModule - LLM-based semantic encoder
2. CrossModalFusion - Gated fusion for multi-modal inputs
3. ResourceDecisionModule - PPO-based resource allocation
4. LLMSRAF - Complete end-to-end framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration"""
    # Semantic encoder
    semantic_hidden_dim: int = 512
    semantic_output_dim: int = 128

    # State encoder
    state_input_dim: int = 64  # CSI + queue state
    state_hidden_dim: int = 256
    state_output_dim: int = 128

    # Orbit encoder
    orbit_input_dim: int = 32  # Satellite position, velocity
    orbit_hidden_dim: int = 128
    orbit_output_dim: int = 64

    # Fusion
    fusion_output_dim: int = 256

    # Actor-Critic
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    # Action space
    num_beams: int = 8
    num_spectrum_blocks: int = 16
    num_power_levels: int = 5
    num_priorities: int = 4

    # Resource output
    resource_dim: int = 5  # latency, bandwidth, power, spectrum, priority


class SemanticEncoder(nn.Module):
    """
    Semantic encoder using pre-trained LLM embeddings
    For Qwen-7B or similar models
    """

    def __init__(self, config: ModelConfig, pretrained_embedding: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # LLM embedding layer (can be replaced with actual LLM)
        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            # Placeholder: learnable embedding
            self.embedding = nn.Embedding(50000, config.semantic_hidden_dim)

        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(config.semantic_hidden_dim, config.semantic_hidden_dim),
            nn.LayerNorm(config.semantic_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.semantic_hidden_dim, config.semantic_output_dim),
        )

        # Resource prediction heads
        self.resource_heads = nn.ModuleDict({
            'latency_class': nn.Linear(config.semantic_output_dim, 3),      # low/medium/high
            'bandwidth_class': nn.Linear(config.semantic_output_dim, 3),    # low/medium/high
            'reliability_class': nn.Linear(config.semantic_output_dim, 3),  # low/medium/high
            'priority': nn.Linear(config.semantic_output_dim, 4),           # 1-4
        })

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with semantic embedding and resource predictions
        """
        # Get embeddings
        embeddings = self.embedding(input_ids)  # [batch, seq, hidden]

        # Pool (mean pooling with attention mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = embeddings.mean(dim=1)

        # Project
        semantic_vec = self.projection(pooled)

        # Predict resource requirements
        outputs = {'semantic_vec': semantic_vec}
        for name, head in self.resource_heads.items():
            outputs[name] = head(semantic_vec)

        return outputs


class StateEncoder(nn.Module):
    """
    Network state encoder (CSI, queue states, etc.)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.state_input_dim, config.state_hidden_dim),
            nn.LayerNorm(config.state_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
            nn.LayerNorm(config.state_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.state_hidden_dim, config.state_output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode network state"""
        return self.encoder(state)


class OrbitEncoder(nn.Module):
    """
    Satellite orbit information encoder
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.orbit_input_dim, config.orbit_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.orbit_hidden_dim, config.orbit_output_dim),
        )

    def forward(self, orbit_info: torch.Tensor) -> torch.Tensor:
        """Encode orbit information"""
        return self.encoder(orbit_info)


class SemanticUnderstandingModule(nn.Module):
    """
    Complete semantic understanding module
    Combines LLM encoder with resource mapping
    """

    def __init__(self, config: ModelConfig, pretrained_embedding: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        self.semantic_encoder = SemanticEncoder(config, pretrained_embedding)
        self.resource_mapper = nn.Sequential(
            nn.Linear(config.semantic_output_dim, config.semantic_output_dim * 2),
            nn.ReLU(),
            nn.Linear(config.semantic_output_dim * 2, config.resource_dim),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process natural language description

        Returns:
            semantic_vec: Semantic embedding
            resource_vec: Resource requirement vector
            class_predictions: Classification predictions
        """
        outputs = self.semantic_encoder(input_ids, attention_mask)

        # Map to resource vector
        outputs['resource_vec'] = self.resource_mapper(outputs['semantic_vec'])

        return outputs


class CrossModalFusion(nn.Module):
    """
    Gated Cross-Modal Fusion Layer

    Fuses semantic, network state, and orbit information
    using a gated mechanism for adaptive weighting.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        semantic_dim = config.semantic_output_dim
        state_dim = config.state_output_dim
        orbit_dim = config.orbit_output_dim
        total_dim = semantic_dim + state_dim + orbit_dim

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 gates for 3 modalities
            nn.Softmax(dim=-1),
        )

        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(total_dim, config.fusion_output_dim),
            nn.LayerNorm(config.fusion_output_dim),
            nn.ReLU(),
        )

    def forward(self, semantic_vec: torch.Tensor,
                state_vec: torch.Tensor,
                orbit_vec: torch.Tensor) -> torch.Tensor:
        """
        Gated fusion of multi-modal inputs

        Args:
            semantic_vec: Semantic embedding [batch, semantic_dim]
            state_vec: Network state encoding [batch, state_dim]
            orbit_vec: Orbit encoding [batch, orbit_dim]

        Returns:
            Fused representation [batch, fusion_output_dim]
        """
        # Concatenate all modalities
        concat = torch.cat([semantic_vec, state_vec, orbit_vec], dim=-1)

        # Compute gates
        gates = self.gate_net(concat)  # [batch, 3]

        # Weighted fusion
        weighted_semantic = gates[:, 0:1] * semantic_vec
        weighted_state = gates[:, 1:2] * state_vec
        weighted_orbit = gates[:, 2:3] * orbit_vec

        # Concatenate weighted features
        fused = torch.cat([weighted_semantic, weighted_state, weighted_orbit], dim=-1)

        # Project to output dimension
        output = self.fusion_proj(fused)

        return output


class ActorNetwork(nn.Module):
    """
    Actor network for PPO
    Outputs resource allocation actions
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.fusion_output_dim, config.actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        self.beam_head = nn.Linear(config.actor_hidden_dim, config.num_beams)
        self.spectrum_head = nn.Linear(config.actor_hidden_dim, config.num_spectrum_blocks)
        self.power_head = nn.Linear(config.actor_hidden_dim, config.num_power_levels)
        self.priority_head = nn.Linear(config.actor_hidden_dim, config.num_priorities)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get action probabilities"""
        features = self.net(x)

        return {
            'beam_probs': F.softmax(self.beam_head(features), dim=-1),
            'spectrum_probs': F.softmax(self.spectrum_head(features), dim=-1),
            'power_probs': F.softmax(self.power_head(features), dim=-1),
            'priority_probs': F.softmax(self.priority_head(features), dim=-1),
        }

    def get_action(self, x: torch.Tensor) -> Tuple[Dict[str, int], torch.Tensor]:
        """Sample action and return log probability"""
        probs = self.forward(x)

        actions = {}
        log_probs = []

        for name, prob in probs.items():
            dist = Categorical(prob)
            action = dist.sample()
            actions[name] = action
            log_probs.append(dist.log_prob(action))

        return actions, torch.stack(log_probs).sum(dim=0)


class CriticNetwork(nn.Module):
    """
    Critic network for PPO
    Estimates state value
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.fusion_output_dim, config.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.critic_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value estimate"""
        return self.net(x)


class ResourceDecisionModule(nn.Module):
    """
    Resource decision module using PPO
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.actor = ActorNetwork(config)
        self.critic = CriticNetwork(config)

    def forward(self, fused_state: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get action probabilities and value estimate"""
        action_probs = self.actor(fused_state)
        value = self.critic(fused_state)
        return action_probs, value

    def select_action(self, fused_state: torch.Tensor) -> Tuple[Dict[str, int], torch.Tensor, torch.Tensor]:
        """Select action for environment interaction"""
        action_probs, value = self.forward(fused_state)

        actions = {}
        log_probs = []

        for name, prob in action_probs.items():
            dist = Categorical(prob)
            action = dist.sample()
            actions[name] = action.item()
            log_probs.append(dist.log_prob(action))

        return actions, torch.stack(log_probs).sum(dim=0), value


class LLMSRAF(nn.Module):
    """
    Complete LLM-SRAF Framework

    End-to-end semantic resource allocation system
    """

    def __init__(self, config: ModelConfig, pretrained_embedding: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # Core modules
        self.semantic_module = SemanticUnderstandingModule(config, pretrained_embedding)
        self.state_encoder = StateEncoder(config)
        self.orbit_encoder = OrbitEncoder(config)
        self.fusion = CrossModalFusion(config)
        self.decision_module = ResourceDecisionModule(config)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                network_state: torch.Tensor,
                orbit_info: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass

        Args:
            input_ids: Tokenized business description
            attention_mask: Attention mask
            network_state: Network state (CSI, queue, etc.)
            orbit_info: Satellite orbit information

        Returns:
            Dictionary with all outputs
        """
        # Semantic understanding
        semantic_outputs = self.semantic_module(input_ids, attention_mask)

        # Encode state and orbit
        state_vec = self.state_encoder(network_state)
        orbit_vec = self.orbit_encoder(orbit_info)

        # Fusion
        fused = self.fusion(
            semantic_outputs['semantic_vec'],
            state_vec,
            orbit_vec
        )

        # Decision
        action_probs, value = self.decision_module(fused)

        # Combine outputs
        outputs = {
            **semantic_outputs,
            'state_vec': state_vec,
            'orbit_vec': orbit_vec,
            'fused': fused,
            'action_probs': action_probs,
            'value': value,
        }

        return outputs

    def allocate(self,
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor],
                 network_state: torch.Tensor,
                 orbit_info: torch.Tensor) -> Dict[str, int]:
        """
        Get resource allocation decision

        Returns:
            Action dictionary with beam, spectrum, power, priority
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, network_state, orbit_info)

            actions = {}
            for name, probs in outputs['action_probs'].items():
                actions[name] = probs.argmax(dim=-1).item()

            return actions


def create_model(config: Optional[ModelConfig] = None,
                  pretrained_embedding: Optional[nn.Module] = None) -> LLMSRAF:
    """Factory function to create LLM-SRAF model"""
    if config is None:
        config = ModelConfig()
    return LLMSRAF(config, pretrained_embedding)