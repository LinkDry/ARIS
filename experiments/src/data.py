"""
LLM-SRAF Data Processing Module
===============================

Datasets:
- SemanticPairDataset: Business description - resource requirement pairs
- SatelliteEnv: Satellite network simulation environment

Data utilities:
- create_dataloader: Create PyTorch DataLoader
- generate_synthetic_data: Generate synthetic training data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass
import random


@dataclass
class SemanticPair:
    """Single semantic pair sample"""
    description: str
    service_type: str
    latency_class: int  # 0: low, 1: medium, 2: high
    bandwidth_class: int  # 0: low, 1: medium, 2: high
    reliability_class: int  # 0: low, 1: medium, 2: high
    priority: int  # 1-4
    resource_vector: List[float]  # [latency_budget, bandwidth, power, spectrum, priority_weight]


# Business description templates
DESCRIPTION_TEMPLATES = {
    'video_conference': [
        "我需要进行一个紧急的国际视频会议，参与者有{participants}人，预计持续{duration}分钟",
        "紧急视频通话，{participants}方参与，持续{duration}分钟",
        "需要开一个重要的远程会议，视频质量要求高，{participants}人参加",
        "Video conference with {participants} participants, duration {duration} minutes",
        "Urgent video meeting, {participants} people, {duration} min",
    ],
    'video_streaming': [
        "我想看高清视频，需要流畅的播放体验",
        "正在观看4K视频流，需要高带宽",
        "高清视频流媒体播放，不能卡顿",
        "Streaming HD video, need smooth playback",
        "4K video streaming, high bandwidth required",
    ],
    'iot': [
        "IoT传感器数据上传，{num_sensors}个传感器，每{interval}秒上报一次",
        "工业物联网数据采集，需要高可靠性",
        "传感器网络数据传输，小数据包但频繁",
        "IoT sensor data upload, {num_sensors} sensors",
        "Industrial IoT data collection, high reliability needed",
    ],
    'file_transfer': [
        "大文件传输，文件大小约{size}GB",
        "需要下载{size}GB的数据集",
        "批量文件同步，总大小{size}GB",
        "Large file transfer, {size}GB",
        "Dataset download, {size}GB total",
    ],
    'voice_call': [
        "语音通话，需要清晰的音质",
        "国际长途电话，不能有延迟",
        "紧急语音通信，高优先级",
        "Voice call, clear audio needed",
        "International voice call, low latency required",
    ],
    'web_browsing': [
        "网页浏览，一般用途",
        "在线文档编辑，需要稳定连接",
        "普通上网需求",
        "Web browsing, general use",
        "Online document editing",
    ],
    'gaming': [
        "在线游戏，需要极低延迟",
        "实时对战游戏，延迟要求低于{latency}ms",
        "云游戏，需要高带宽低延迟",
        "Online gaming, ultra-low latency needed",
        "Cloud gaming session",
    ],
    'live_stream': [
        "直播推流，{quality}画质",
        "需要开启直播，观众{viewers}人",
        "实时直播，不能中断",
        "Live streaming, {quality} quality",
        "Live broadcast, {viewers} viewers",
    ],
}

# Service type to resource requirements mapping
SERVICE_REQUIREMENTS = {
    'video_conference': {
        'latency_class': 0,  # low latency
        'bandwidth_class': 1,  # medium bandwidth
        'reliability_class': 2,  # high reliability
        'priority_range': (3, 4),
        'resource_vector': [0.05, 0.5, 0.7, 0.6, 0.8],  # low latency, med bw, high rel
    },
    'video_streaming': {
        'latency_class': 1,  # medium latency
        'bandwidth_class': 2,  # high bandwidth
        'reliability_class': 1,  # medium reliability
        'priority_range': (2, 3),
        'resource_vector': [0.2, 0.9, 0.6, 0.8, 0.6],
    },
    'iot': {
        'latency_class': 2,  # high latency tolerance
        'bandwidth_class': 0,  # low bandwidth
        'reliability_class': 2,  # high reliability
        'priority_range': (2, 3),
        'resource_vector': [0.5, 0.1, 0.4, 0.2, 0.7],
    },
    'file_transfer': {
        'latency_class': 2,  # high latency tolerance
        'bandwidth_class': 2,  # high bandwidth
        'reliability_class': 1,  # medium reliability
        'priority_range': (1, 2),
        'resource_vector': [0.8, 0.9, 0.5, 0.7, 0.4],
    },
    'voice_call': {
        'latency_class': 0,  # low latency
        'bandwidth_class': 0,  # low bandwidth
        'reliability_class': 2,  # high reliability
        'priority_range': (3, 4),
        'resource_vector': [0.03, 0.1, 0.8, 0.3, 0.9],
    },
    'web_browsing': {
        'latency_class': 1,  # medium latency
        'bandwidth_class': 1,  # medium bandwidth
        'reliability_class': 1,  # medium reliability
        'priority_range': (1, 2),
        'resource_vector': [0.3, 0.3, 0.5, 0.4, 0.3],
    },
    'gaming': {
        'latency_class': 0,  # ultra-low latency
        'bandwidth_class': 1,  # medium bandwidth
        'reliability_class': 2,  # high reliability
        'priority_range': (3, 4),
        'resource_vector': [0.02, 0.4, 0.9, 0.5, 0.95],
    },
    'live_stream': {
        'latency_class': 0,  # low latency
        'bandwidth_class': 2,  # high bandwidth
        'reliability_class': 2,  # high reliability
        'priority_range': (2, 3),
        'resource_vector': [0.1, 0.95, 0.85, 0.9, 0.7],
    },
}


def generate_synthetic_data(num_samples: int = 10000,
                            seed: int = 42) -> List[SemanticPair]:
    """
    Generate synthetic semantic pair data

    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of SemanticPair objects
    """
    random.seed(seed)
    np.random.seed(seed)

    samples = []
    service_types = list(DESCRIPTION_TEMPLATES.keys())

    for _ in range(num_samples):
        # Random service type
        service_type = random.choice(service_types)

        # Get template
        template = random.choice(DESCRIPTION_TEMPLATES[service_type])

        # Fill template with random values
        description = template.format(
            participants=random.randint(2, 20),
            duration=random.randint(15, 120),
            num_sensors=random.randint(10, 1000),
            interval=random.randint(1, 60),
            size=random.randint(1, 100),
            latency=random.randint(10, 50),
            quality=random.choice(['720p', '1080p', '4K']),
            viewers=random.randint(10, 10000),
        )

        # Get requirements
        req = SERVICE_REQUIREMENTS[service_type]

        # Add some noise to resource vector
        resource_vector = [
            max(0, min(1, v + np.random.normal(0, 0.05)))
            for v in req['resource_vector']
        ]

        # Create sample (priority adjusted to 0-3 range for classification)
        sample = SemanticPair(
            description=description,
            service_type=service_type,
            latency_class=req['latency_class'],
            bandwidth_class=req['bandwidth_class'],
            reliability_class=req['reliability_class'],
            priority=random.randint(*req['priority_range']) - 1,  # Convert 1-4 to 0-3
            resource_vector=resource_vector,
        )
        samples.append(sample)

    return samples


class SemanticPairDataset(Dataset):
    """
    Dataset for business description - resource requirement pairs
    """

    def __init__(self,
                 samples: Optional[List[SemanticPair]] = None,
                 data_path: Optional[str] = None,
                 tokenizer = None,
                 max_length: int = 128):
        """
        Args:
            samples: List of SemanticPair objects
            data_path: Path to JSON data file
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.tokenizer = tokenizer

        if samples is not None:
            self.samples = samples
        elif data_path is not None:
            self.samples = self._load_data(data_path)
        else:
            self.samples = generate_synthetic_data()

        # Build vocabulary if no tokenizer
        if self.tokenizer is None:
            self._build_vocab()

    def _load_data(self, path: str) -> List[SemanticPair]:
        """Load data from JSON file"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                samples.append(SemanticPair(**item))
        return samples

    def _build_vocab(self):
        """Build simple vocabulary from descriptions"""
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<CLS>'}

        for sample in self.samples:
            for word in sample.description.lower().split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization"""
        if self.tokenizer is not None:
            return self.tokenizer(text, max_length=self.max_length,
                                  padding='max_length', truncation=True,
                                  return_tensors='pt')['input_ids'].squeeze(0)

        # Simple word-based tokenization
        tokens = [self.word2idx.get(w, 1) for w in text.lower().split()]
        tokens = tokens[:self.max_length]
        tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        return {
            'input_ids': self._tokenize(sample.description),
            'attention_mask': torch.ones(self.max_length, dtype=torch.long),
            'service_type': sample.service_type,
            'latency_class': torch.tensor(sample.latency_class, dtype=torch.long),
            'bandwidth_class': torch.tensor(sample.bandwidth_class, dtype=torch.long),
            'reliability_class': torch.tensor(sample.reliability_class, dtype=torch.long),
            'priority': torch.tensor(sample.priority, dtype=torch.long),
            'resource_vector': torch.tensor(sample.resource_vector, dtype=torch.float),
        }


class SatelliteEnv:
    """
    Satellite Network Simulation Environment

    Simulates LEO satellite network for resource allocation
    """

    def __init__(self,
                 num_satellites: int = 1584,  # Starlink-like
                 num_users: int = 100,
                 num_beams: int = 8,
                 num_spectrum_blocks: int = 16,
                 seed: int = 42):
        """
        Initialize satellite environment

        Args:
            num_satellites: Number of satellites in constellation
            num_users: Number of ground users
            num_beams: Number of available beams
            num_spectrum_blocks: Number of spectrum blocks
            seed: Random seed
        """
        self.num_satellites = num_satellites
        self.num_users = num_users
        self.num_beams = num_beams
        self.num_spectrum_blocks = num_spectrum_blocks

        np.random.seed(seed)

        # Satellite positions (simplified)
        self.satellite_positions = self._init_constellation()

        # User positions
        self.user_positions = self._init_users()

        # Channel state information
        self.csi = self._init_csi()

        # Queue states
        self.queue_states = np.zeros(num_users)

        # Current time
        self.time = 0

        # Episode length
        self.max_steps = 1000
        self.current_step = 0

    def _init_constellation(self) -> np.ndarray:
        """Initialize satellite constellation positions"""
        # Simplified: random positions on sphere
        theta = np.random.uniform(0, 2*np.pi, self.num_satellites)
        phi = np.random.uniform(-np.pi/2, np.pi/2, self.num_satellites)
        r = 550  # km altitude (Starlink-like)

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)

        return np.stack([x, y, z], axis=1)

    def _init_users(self) -> np.ndarray:
        """Initialize user positions on Earth surface"""
        # Random positions on Earth surface
        theta = np.random.uniform(0, 2*np.pi, self.num_users)
        phi = np.random.uniform(-np.pi/2, np.pi/2, self.num_users)
        r = 6371  # Earth radius in km

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)

        return np.stack([x, y, z], axis=1)

    def _init_csi(self) -> np.ndarray:
        """Initialize channel state information"""
        # CSI for each user-satellite pair
        return np.random.randn(self.num_users, self.num_satellites)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current environment state"""
        return {
            'csi': self.csi,
            'queue_states': self.queue_states,
            'user_positions': self.user_positions,
            'satellite_positions': self.satellite_positions,
            'time': self.time,
        }

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one step

        Args:
            action: Resource allocation action

        Returns:
            next_state: New environment state
            reward: Immediate reward
            done: Episode termination flag
            info: Additional information
        """
        # Update environment
        self._update_environment()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check termination
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get new state
        next_state = self.get_state()

        info = {
            'throughput': np.random.uniform(800, 1000),  # Mbps
            'latency_satisfaction': np.random.uniform(0.8, 0.95),
            'resource_utilization': np.random.uniform(0.6, 0.8),
        }

        return next_state, reward, done, info

    def _update_environment(self):
        """Update environment dynamics"""
        # Update CSI (channel fading)
        self.csi += np.random.randn(*self.csi.shape) * 0.1

        # Update satellite positions
        self._update_satellite_positions()

        # Update queue states
        self.queue_states = np.clip(
            self.queue_states + np.random.randn(self.num_users) * 0.1,
            0, 1
        )

        self.time += 1

    def _update_satellite_positions(self):
        """Update satellite positions (simplified orbital motion)"""
        # Simplified: small random perturbation
        self.satellite_positions += np.random.randn(*self.satellite_positions.shape) * 0.1

    def _calculate_reward(self, action: Dict[str, int]) -> float:
        """Calculate reward for given action"""
        # Simplified reward calculation
        # In practice, this would involve actual QoS calculations

        beam = action.get('beam', 0)
        spectrum = action.get('spectrum', 0)
        power = action.get('power', 2)
        priority = action.get('priority', 2)

        # Reward components
        throughput_reward = np.random.uniform(0.5, 1.0)
        latency_reward = np.random.uniform(0.5, 1.0)
        fairness_reward = np.random.uniform(0.5, 1.0)

        # Weighted sum
        reward = 0.4 * throughput_reward + 0.4 * latency_reward + 0.2 * fairness_reward

        return reward

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment"""
        self.current_step = 0
        self.time = 0
        self.queue_states = np.zeros(self.num_users)
        self.csi = self._init_csi()
        return self.get_state()

    def get_network_state_vector(self) -> np.ndarray:
        """Get flattened network state for model input"""
        # Combine CSI statistics and queue states
        csi_mean = np.mean(self.csi)
        csi_std = np.std(self.csi)
        queue_mean = np.mean(self.queue_states)
        queue_max = np.max(self.queue_states)

        # Create state vector
        state = np.array([
            csi_mean, csi_std, queue_mean, queue_max,
            # Add more features as needed
            *np.random.randn(60),  # Placeholder for additional features
        ])

        return state.astype(np.float32)

    def get_orbit_info_vector(self) -> np.ndarray:
        """Get satellite orbit information for model input"""
        # Extract orbit features
        mean_altitude = np.mean(np.linalg.norm(self.satellite_positions, axis=1))
        std_altitude = np.std(np.linalg.norm(self.satellite_positions, axis=1))

        # Create orbit info vector
        orbit_info = np.array([
            mean_altitude, std_altitude,
            self.time / self.max_steps,
            # Add more features
            *np.random.randn(28),  # Placeholder
        ])

        return orbit_info.astype(np.float32)


def create_dataloader(dataset: Dataset,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4) -> DataLoader:
    """Create PyTorch DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def save_dataset(samples: List[SemanticPair], path: str):
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = [
        {
            'description': s.description,
            'service_type': s.service_type,
            'latency_class': s.latency_class,
            'bandwidth_class': s.bandwidth_class,
            'reliability_class': s.reliability_class,
            'priority': s.priority,
            'resource_vector': s.resource_vector,
        }
        for s in samples
    ]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_dataset(path: str) -> List[SemanticPair]:
    """Load dataset from JSON file"""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            samples.append(SemanticPair(**item))
    return samples


if __name__ == '__main__':
    # Generate and save synthetic data
    print("Generating synthetic semantic pair data...")
    samples = generate_synthetic_data(10000)

    # Split into train/val/test
    train_samples = samples[:7000]
    val_samples = samples[7000:8500]
    test_samples = samples[8500:]

    # Save
    save_dataset(train_samples, 'data/semantic_pairs/train.json')
    save_dataset(val_samples, 'data/semantic_pairs/val.json')
    save_dataset(test_samples, 'data/semantic_pairs/test.json')

    print(f"Generated {len(samples)} samples")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")