"""
LLM-SRAF Unit Tests
===================

Tests for core model components and data processing.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import (
    ModelConfig,
    SemanticEncoder,
    StateEncoder,
    OrbitEncoder,
    CrossModalFusion,
    ActorNetwork,
    CriticNetwork,
    ResourceDecisionModule,
    LLMSRAF,
    create_model,
)
from data import (
    SemanticPairDataset,
    SatelliteEnv,
    generate_synthetic_data,
    SemanticPair,
)


class TestModelConfig:
    """Test model configuration"""

    def test_default_config(self):
        config = ModelConfig()
        assert config.semantic_hidden_dim == 512
        assert config.semantic_output_dim == 128
        assert config.fusion_output_dim == 256

    def test_custom_config(self):
        config = ModelConfig(semantic_hidden_dim=256, semantic_output_dim=64)
        assert config.semantic_hidden_dim == 256
        assert config.semantic_output_dim == 64


class TestSemanticEncoder:
    """Test semantic encoder"""

    def test_forward(self):
        config = ModelConfig()
        encoder = SemanticEncoder(config)

        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        outputs = encoder(input_ids)

        assert 'semantic_vec' in outputs
        assert outputs['semantic_vec'].shape == (batch_size, config.semantic_output_dim)

    def test_resource_prediction(self):
        config = ModelConfig()
        encoder = SemanticEncoder(config)

        input_ids = torch.randint(0, 1000, (2, 16))
        outputs = encoder(input_ids)

        assert 'latency_class' in outputs
        assert outputs['latency_class'].shape == (2, 3)


class TestStateEncoder:
    """Test state encoder"""

    def test_forward(self):
        config = ModelConfig()
        encoder = StateEncoder(config)

        batch_size = 4
        state = torch.randn(batch_size, config.state_input_dim)

        output = encoder(state)

        assert output.shape == (batch_size, config.state_output_dim)


class TestOrbitEncoder:
    """Test orbit encoder"""

    def test_forward(self):
        config = ModelConfig()
        encoder = OrbitEncoder(config)

        batch_size = 4
        orbit_info = torch.randn(batch_size, config.orbit_input_dim)

        output = encoder(orbit_info)

        assert output.shape == (batch_size, config.orbit_output_dim)


class TestCrossModalFusion:
    """Test cross-modal fusion"""

    def test_forward(self):
        config = ModelConfig()
        fusion = CrossModalFusion(config)

        batch_size = 4
        semantic_vec = torch.randn(batch_size, config.semantic_output_dim)
        state_vec = torch.randn(batch_size, config.state_output_dim)
        orbit_vec = torch.randn(batch_size, config.orbit_output_dim)

        output = fusion(semantic_vec, state_vec, orbit_vec)

        assert output.shape == (batch_size, config.fusion_output_dim)


class TestActorCritic:
    """Test actor-critic networks"""

    def test_actor_forward(self):
        config = ModelConfig()
        actor = ActorNetwork(config)

        batch_size = 4
        x = torch.randn(batch_size, config.fusion_output_dim)

        outputs = actor(x)

        assert 'beam_probs' in outputs
        assert 'spectrum_probs' in outputs
        assert 'power_probs' in outputs
        assert 'priority_probs' in outputs

    def test_critic_forward(self):
        config = ModelConfig()
        critic = CriticNetwork(config)

        batch_size = 4
        x = torch.randn(batch_size, config.fusion_output_dim)

        value = critic(x)

        assert value.shape == (batch_size, 1)


class TestLLMSRAF:
    """Test complete LLM-SRAF model"""

    def test_forward(self):
        config = ModelConfig()
        model = LLMSRAF(config)

        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        attention_mask = torch.ones(batch_size, 32, dtype=torch.long)
        network_state = torch.randn(batch_size, config.state_input_dim)
        orbit_info = torch.randn(batch_size, config.orbit_input_dim)

        outputs = model(input_ids, attention_mask, network_state, orbit_info)

        assert 'semantic_vec' in outputs
        assert 'fused' in outputs
        assert 'action_probs' in outputs
        assert 'value' in outputs

    def test_allocate(self):
        config = ModelConfig()
        model = LLMSRAF(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)
        network_state = torch.randn(1, config.state_input_dim)
        orbit_info = torch.randn(1, config.orbit_input_dim)

        actions = model.allocate(input_ids, attention_mask, network_state, orbit_info)

        assert 'beam' in actions
        assert 'spectrum' in actions
        assert 'power' in actions
        assert 'priority' in actions


class TestSemanticPairDataset:
    """Test semantic pair dataset"""

    def test_create_dataset(self):
        samples = generate_synthetic_data(100)
        dataset = SemanticPairDataset(samples=samples)

        assert len(dataset) == 100

    def test_getitem(self):
        samples = generate_synthetic_data(100)
        dataset = SemanticPairDataset(samples=samples)

        item = dataset[0]

        assert 'input_ids' in item
        assert 'resource_vector' in item
        assert item['input_ids'].shape[0] == 128


class TestSatelliteEnv:
    """Test satellite environment"""

    def test_init(self):
        env = SatelliteEnv(num_users=50)

        assert env.num_users == 50
        assert env.satellite_positions.shape == (env.num_satellites, 3)

    def test_reset(self):
        env = SatelliteEnv()
        state = env.reset()

        assert 'csi' in state
        assert 'queue_states' in state

    def test_step(self):
        env = SatelliteEnv()
        env.reset()

        action = {
            'beam': 0,
            'spectrum': 5,
            'power': 2,
            'priority': 3,
        }

        next_state, reward, done, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert 'throughput' in info

    def test_get_vectors(self):
        env = SatelliteEnv()
        env.reset()

        network_state = env.get_network_state_vector()
        orbit_info = env.get_orbit_info_vector()

        assert network_state.shape[0] == 64
        assert orbit_info.shape[0] == 32


class TestIntegration:
    """Integration tests"""

    def test_full_pipeline(self):
        """Test complete forward pipeline"""
        config = ModelConfig()
        model = create_model(config)
        model.eval()

        env = SatelliteEnv()
        env.reset()

        # Get environment state
        network_state = torch.tensor(env.get_network_state_vector()).unsqueeze(0)
        orbit_info = torch.tensor(env.get_orbit_info_vector()).unsqueeze(0)

        # Create dummy text input
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)

        # Get allocation
        with torch.no_grad():
            actions = model.allocate(input_ids, attention_mask, network_state, orbit_info)

        # Step environment
        next_state, reward, done, info = env.step(actions)

        assert isinstance(actions, dict)
        assert reward >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])