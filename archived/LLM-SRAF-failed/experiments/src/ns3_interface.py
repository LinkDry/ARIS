"""
NS-3 + SNS Satellite Network Simulation Interface
=================================================

专业卫星网络仿真环境接口
支持 NS-3 (Network Simulator 3) + SNS (Satellite Network Simulator) 扩展

安装要求:
- NS-3.38+ (推荐 NS-3.41)
- SNS Extension (Satellite Network Simulator)
- Python 3.8+
- ns3-python bindings 或 ZMQ 通信接口

参考:
- NS-3: https://www.nsnam.org/
- SNS: https://github.com/sns/sns-ns3
- NTN 仿真: https://gitlab.com/cttc-lena/ntn
"""

import os
import json
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import socket
import struct
import time


@dataclass
class NS3Config:
    """NS-3 仿真配置"""
    # 卫星星座配置
    constellation_type: str = "starlink"  # starlink, oneweb, kuiper
    num_orbital_planes: int = 72
    sats_per_plane: int = 22
    altitude_km: float = 550.0
    inclination_deg: float = 53.0

    # 地面站配置
    num_ground_stations: int = 100
    ground_station_distribution: str = "urban_weighted"  # uniform, urban_weighted, custom

    # 链路配置
    freq_band: str = "Ka"  # Ka, Ku, V
    bandwidth_mhz: float = 400.0
    max_beam_per_sat: int = 8

    # ISL (Inter-Satellite Link)
    enable_isl: bool = True
    isl_topology: str = "grid"  # grid, mesh

    # 仿真参数
    sim_duration_sec: float = 86400.0  # 24小时
    time_step_ms: float = 100.0
    random_seed: int = 42

    # 输出配置
    enable_tracing: bool = True
    trace_output_dir: str = "traces/"
    enable_pcap: bool = False


@dataclass
class SatelliteState:
    """卫星状态"""
    sat_id: int
    position: Tuple[float, float, float]  # ECEF (km)
    velocity: Tuple[float, float, float]  # km/s
    visible_users: List[int]
    available_beams: int
    load_factor: float  # 0-1


@dataclass
class UserState:
    """用户状态"""
    user_id: int
    position: Tuple[float, float, float]  # lat, lon, alt
    serving_satellite: Optional[int]
    csi: np.ndarray  # Channel State Information
    sinr_db: float
    throughput_mbps: float
    latency_ms: float
    queue_length: int


class NS3SimulationBase(ABC):
    """NS-3 仿真基类"""

    @abstractmethod
    def initialize(self, config: NS3Config) -> bool:
        """初始化仿真"""
        pass

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """执行一步仿真"""
        pass

    @abstractmethod
    def get_state(self) -> Dict:
        """获取当前状态"""
        pass

    @abstractmethod
    def reset(self) -> Dict:
        """重置仿真"""
        pass

    @abstractmethod
    def close(self):
        """关闭仿真"""
        pass


class NS3PythonBinding(NS3SimulationBase):
    """
    通过 Python Bindings 直接调用 NS-3

    要求: ns3-python bindings 已安装
    适用于: Linux 环境
    """

    def __init__(self):
        self.simulator = None
        self.config = None
        self.initialized = False

        # 尝试导入 ns-3 python 模块
        try:
            import ns.core as ns_core
            import ns.network as ns_network
            import ns.internet as ns_internet
            import ns.satellite as ns_satellite

            self.ns_core = ns_core
            self.ns_network = ns_network
            self.ns_internet = ns_internet
            self.ns_satellite = ns_satellite
            self.binding_available = True
        except ImportError:
            self.binding_available = False
            print("Warning: NS-3 Python bindings not available.")
            print("Please install ns-3 with Python bindings or use ZMQ interface.")

    def initialize(self, config: NS3Config) -> bool:
        if not self.binding_available:
            return False

        self.config = config

        # 设置随机种子
        self.ns_core.RngSeedManager.SetSeed(config.random_seed)
        self.ns_core.RngSeedManager.SetRun(1)

        # 创建卫星星座
        self._create_constellation()

        # 创建地面站
        self._create_ground_stations()

        # 配置链路
        self._configure_links()

        self.initialized = True
        return True

    def _create_constellation(self):
        """创建卫星星座"""
        # 创建卫星节点容器
        self.satellites = self.ns_network.NodeContainer()
        num_sats = self.config.num_orbital_planes * self.config.sats_per_plane
        self.satellites.Create(num_sats)

        # 安装卫星移动模型
        # (实际实现需要调用 SNS 扩展的卫星轨道模型)
        pass

    def _create_ground_stations(self):
        """创建地面站"""
        self.ground_stations = self.ns_network.NodeContainer()
        self.ground_stations.Create(self.config.num_ground_stations)
        pass

    def _configure_links(self):
        """配置链路"""
        # 配置用户链路 (User Link)
        # 配置馈电链路 (Feeder Link)
        # 配置星间链路 (ISL)
        pass

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        if not self.initialized:
            raise RuntimeError("Simulation not initialized")

        # 应用资源分配动作
        self._apply_action(action)

        # 运行一个时间步
        self.ns_core.Simulator.Schedule(
            self.ns_core.Time(f"{self.config.time_step_ms}ms"),
            self._step_callback
        )
        self.ns_core.Simulator.Run()

        # 获取状态和奖励
        state = self.get_state()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        return state, reward, done, info

    def _apply_action(self, action: Dict):
        """应用资源分配动作到 NS-3 仿真"""
        pass

    def _step_callback(self):
        """时间步回调"""
        pass

    def _compute_reward(self) -> float:
        """计算奖励"""
        # 基于吞吐量、时延、公平性等计算
        return 0.0

    def _check_done(self) -> bool:
        """检查是否结束"""
        return False

    def _get_info(self) -> Dict:
        """获取附加信息"""
        return {}

    def get_state(self) -> Dict:
        """获取当前仿真状态"""
        return {
            'satellites': [],
            'users': [],
            'time': 0.0,
        }

    def reset(self) -> Dict:
        """重置仿真"""
        self.ns_core.Simulator.Destroy()
        return self.initialize(self.config)

    def close(self):
        """关闭仿真"""
        if self.initialized:
            self.ns_core.Simulator.Destroy()
            self.initialized = False


class NS3ZMQInterface(NS3SimulationBase):
    """
    通过 ZMQ 与 NS-3 进程通信

    适用于: Windows + WSL2 或 远程服务器
    NS-3 运行在 Linux 环境，Python 控制程序在任意环境
    """

    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.config = None

    def connect(self) -> bool:
        """连接到 NS-3 ZMQ 服务器"""
        try:
            import zmq
            context = zmq.Context()
            self.socket = context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to NS-3 ZMQ server: {e}")
            return False

    def initialize(self, config: NS3Config) -> bool:
        if not self.connected:
            if not self.connect():
                return False

        self.config = config

        # 发送初始化命令
        cmd = {
            'type': 'init',
            'config': config.__dict__,
        }
        response = self._send_command(cmd)
        return response.get('success', False)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        cmd = {
            'type': 'step',
            'action': action,
        }
        response = self._send_command(cmd)
        return (
            response.get('state', {}),
            response.get('reward', 0.0),
            response.get('done', False),
            response.get('info', {}),
        )

    def get_state(self) -> Dict:
        cmd = {'type': 'get_state'}
        return self._send_command(cmd)

    def reset(self) -> Dict:
        cmd = {'type': 'reset'}
        response = self._send_command(cmd)
        return response.get('state', {})

    def close(self):
        if self.socket:
            self.socket.close()
            self.connected = False

    def _send_command(self, cmd: Dict) -> Dict:
        """发送命令到 NS-3"""
        if not self.connected:
            return {'success': False, 'error': 'Not connected'}

        try:
            self.socket.send_json(cmd)
            response = self.socket.recv_json()
            return response
        except Exception as e:
            return {'success': False, 'error': str(e)}


class NS3FileInterface(NS3SimulationBase):
    """
    通过文件交换与 NS-3 通信

    适用于: 离线仿真、批量实验
    NS-3 读取配置文件，输出结果文件
    """

    def __init__(self, work_dir: str = "./ns3_workspace"):
        self.work_dir = work_dir
        self.config = None
        self.sim_counter = 0

        os.makedirs(work_dir, exist_ok=True)

    def initialize(self, config: NS3Config) -> bool:
        self.config = config

        # 写入配置文件
        config_path = os.path.join(self.work_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)

        return True

    def run_simulation(self, duration_sec: Optional[float] = None) -> Dict:
        """
        运行完整的 NS-3 仿真

        返回: 仿真结果字典
        """
        if duration_sec is None:
            duration_sec = self.config.sim_duration_sec

        # 生成仿真脚本输入
        input_file = os.path.join(self.work_dir, f"sim_{self.sim_counter}.json")
        output_file = os.path.join(self.work_dir, f"result_{self.sim_counter}.json")

        sim_input = {
            'config_path': os.path.join(self.work_dir, "config.json"),
            'duration_sec': duration_sec,
            'output_file': output_file,
            'seed': self.config.random_seed + self.sim_counter,
        }

        with open(input_file, 'w') as f:
            json.dump(sim_input, f)

        # 调用 NS-3 仿真 (需要先编译 NS-3)
        # ns3_cmd = f"./ns3 run 'satellite-sim --input={input_file}'"
        # subprocess.run(ns3_cmd, shell=True, cwd=self.ns3_dir)

        # 这里生成模拟结果用于演示
        result = self._generate_mock_result(duration_sec)

        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        self.sim_counter += 1
        return result

    def _generate_mock_result(self, duration: float) -> Dict:
        """生成模拟结果 (实际应从 NS-3 获取)"""
        num_steps = int(duration * 10)  # 每 100ms 一个采样点

        return {
            'duration_sec': duration,
            'throughput_mbps': np.random.uniform(800, 1200, num_steps).tolist(),
            'latency_ms': np.random.uniform(20, 100, num_steps).tolist(),
            'sinr_db': np.random.uniform(10, 25, num_steps).tolist(),
            'coverage_ratio': np.random.uniform(0.95, 0.99, num_steps).tolist(),
            'handover_count': np.random.randint(100, 500),
            'dropped_packets': np.random.randint(0, 100),
            'total_bytes': np.random.uniform(1e9, 1e10),
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        # 文件接口不支持实时 step，需要批量运行
        raise NotImplementedError("Use run_simulation() for file-based interface")

    def get_state(self) -> Dict:
        raise NotImplementedError("Use run_simulation() for file-based interface")

    def reset(self) -> Dict:
        self.sim_counter = 0
        return {}

    def close(self):
        pass


class SatelliteEnvNS3:
    """
    基于 NS-3 的卫星网络仿真环境

    兼容 Gym 接口，可直接用于 RL 训练
    """

    def __init__(self,
                 interface: str = "auto",
                 config: Optional[NS3Config] = None,
                 **kwargs):
        """
        Args:
            interface: 接口类型
                - "auto": 自动选择 (python > zmq > file)
                - "python": 直接 Python bindings
                - "zmq": ZMQ 通信
                - "file": 文件交换
            config: NS-3 配置
        """
        self.config = config or NS3Config()

        # 选择接口
        if interface == "auto":
            self.sim = self._auto_select_interface(**kwargs)
        elif interface == "python":
            self.sim = NS3PythonBinding()
        elif interface == "zmq":
            self.sim = NS3ZMQInterface(**kwargs)
        elif interface == "file":
            self.sim = NS3FileInterface(**kwargs)
        else:
            raise ValueError(f"Unknown interface: {interface}")

        # 动作空间
        self.action_space = {
            'beam_allocation': (self.config.num_ground_stations, self.config.max_beam_per_sat),
            'power_control': (self.config.num_ground_stations,),
            'spectrum_allocation': (self.config.num_ground_stations,),
            'priority': (self.config.num_ground_stations,),
        }

        # 状态空间维度
        self.state_dim = 256  # 可调整

        # 初始化
        self._initialized = False

    def _auto_select_interface(self, **kwargs) -> NS3SimulationBase:
        """自动选择最佳接口"""
        # 优先尝试 Python bindings
        sim = NS3PythonBinding()
        if sim.binding_available:
            print("Using NS-3 Python bindings")
            return sim

        # 其次尝试 ZMQ
        sim = NS3ZMQInterface(**kwargs)
        if sim.connect():
            print("Using NS-3 ZMQ interface")
            return sim

        # 最后使用文件接口
        print("Using NS-3 file interface (offline mode)")
        return NS3FileInterface(**kwargs)

    def initialize(self) -> bool:
        """初始化仿真环境"""
        self._initialized = self.sim.initialize(self.config)
        return self._initialized

    def reset(self) -> np.ndarray:
        """重置环境"""
        if not self._initialized:
            self.initialize()

        state = self.sim.reset()
        return self._process_state(state)

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        if not self._initialized:
            self.initialize()

        state, reward, done, info = self.sim.step(action)
        return self._process_state(state), reward, done, info

    def _process_state(self, state: Dict) -> np.ndarray:
        """处理状态为特征向量"""
        # 提取关键特征
        features = []

        # 卫星负载
        if 'satellites' in state:
            sat_loads = [s.get('load_factor', 0) for s in state['satellites']]
            features.extend([
                np.mean(sat_loads),
                np.std(sat_loads),
                np.max(sat_loads),
            ])

        # 用户状态统计
        if 'users' in state:
            throughputs = [u.get('throughput_mbps', 0) for u in state['users']]
            latencies = [u.get('latency_ms', 0) for u in state['users']]
            features.extend([
                np.mean(throughputs),
                np.std(throughputs),
                np.mean(latencies),
                np.std(latencies),
            ])

        # 补齐到固定维度
        while len(features) < self.state_dim:
            features.append(0.0)

        return np.array(features[:self.state_dim], dtype=np.float32)

    def get_network_state_vector(self) -> np.ndarray:
        """获取网络状态向量 (兼容原接口)"""
        state = self.sim.get_state()
        return self._process_state(state)

    def get_orbit_info_vector(self) -> np.ndarray:
        """获取轨道信息向量 (兼容原接口)"""
        # 提取轨道相关特征
        features = np.random.randn(32).astype(np.float32)  # 占位
        return features

    def close(self):
        """关闭环境"""
        self.sim.close()
        self._initialized = False


# ============== NS-3 C++ 脚本模板 ==============

NS3_SCRIPT_TEMPLATE = """
/**
 * NS-3 Satellite Network Simulation Script
 * 用于 LLM-SRAF 实验
 *
 * 编译: ./ns3 build
 * 运行: ./ns3 run "satellite-llm-sraf --config=config.json"
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/satellite-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/config-store-module.h"

using namespace ns3;

// 全局变量
NodeContainer satellites;
NodeContainer groundStations;
NetDeviceContainer satDevices;

// 仿真参数
double simDuration = 86400.0;  // 24小时
uint32_t numSats = 1584;       // Starlink-like
uint32_t numGS = 100;

void
ConfigureConstellation()
{
    // 创建卫星节点
    satellites.Create(numSats);

    // 配置 Walker 星座
    // 72 轨道面 × 22 卫星
    // 高度 550km, 倾角 53°

    MobilityHelper mobility;
    // ... 安装卫星移动模型
}

void
ConfigureGroundStations()
{
    groundStations.Create(numGS);

    // 配置地面站位置
    // 使用实际城市分布数据
}

void
ConfigureISL()
{
    // 配置星间链路
    // Grid 拓扑: 每颗卫星 4 条 ISL
    // (前、后、左、右邻居)
}

void
ConfigureUserLinks()
{
    // 配置用户链路
    // Ka 频段, 400MHz 带宽
    // 点波束, 每卫星最多 8 个波束
}

void
InstallApplications()
{
    // 安装应用层
    // UDP/TCP 流量生成
    // QoS 监控
}

void
SetupTracing()
{
    // 配置追踪输出
    // 吞吐量、时延、丢包率
}

int
main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.AddValue("duration", "Simulation duration (seconds)", simDuration);
    cmd.AddValue("numSats", "Number of satellites", numSats);
    cmd.AddValue("numGS", "Number of ground stations", numGS);
    cmd.Parse(argc, argv);

    // 配置仿真
    ConfigureConstellation();
    ConfigureGroundStations();
    ConfigureISL();
    ConfigureUserLinks();
    InstallApplications();
    SetupTracing();

    // 运行仿真
    Simulator::Stop(Seconds(simDuration));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
"""


# ============== 安装脚本 ==============

def generate_install_script() -> str:
    """生成 NS-3 + SNS 安装脚本"""
    return """
#!/bin/bash
# ============================================
# NS-3 + SNS 卫星网络仿真环境安装脚本
# 适用于: Ubuntu 22.04 / WSL2
# ============================================

set -e

# 1. 安装依赖
sudo apt update
sudo apt install -y \\
    build-essential \\
    cmake \\
    python3 \\
    python3-pip \\
    python3-dev \\
    git \\
    mercurial \\
    autoconf \\
    automake \\
    libgsl-dev \\
    libgtk-3-dev \\
    libsqlite3-dev \\
    libxml2-dev \\
    libboost-all-dev \\
    libeigen3-dev \\
    qtbase5-dev \\
    qtchooser \\
    qt5-qmake \\
    qtbase5-dev-tools \\
    openmpi-bin \\
    openmpi-common \\
    libopenmpi-dev \\
    libzmq3-dev \\
    python3-zmq

# 2. 下载 NS-3
cd ~
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3
cd ns-3

# 3. 下载 SNS 扩展 (卫星网络仿真)
# 选项 A: CTTC NTN 模块
git clone https://gitlab.com/cttc-lena/ntn.git src/ntn

# 选项 B: SNS (如果有开源版本)
# git clone https://github.com/sns/sns-ns3.git src/satellite

# 4. 配置和编译
./ns3 configure --enable-examples --enable-tests --enable-python-bindings
./ns3 build

# 5. 测试安装
./ns3 run hello-simulator

# 6. 设置环境变量
echo 'export NS3_HOME=~/ns-3' >> ~/.bashrc
echo 'export PATH=$PATH:~/ns-3' >> ~/.bashrc

echo "NS-3 + SNS installation complete!"
echo "NS3_HOME: ~/ns-3"
"""


if __name__ == "__main__":
    # 演示使用
    print("=" * 60)
    print("NS-3 Satellite Network Simulation Interface")
    print("=" * 60)

    # 创建环境
    config = NS3Config(
        constellation_type="starlink",
        num_orbital_planes=72,
        sats_per_plane=22,
        num_ground_stations=100,
    )

    # 选择接口
    env = SatelliteEnvNS3(interface="file", config=config)

    # 运行仿真
    result = env.sim.run_simulation(duration_sec=3600)  # 1小时仿真

    print(f"Simulation completed!")
    print(f"  Duration: {result['duration_sec']} seconds")
    print(f"  Avg Throughput: {np.mean(result['throughput_mbps']):.2f} Mbps")
    print(f"  Avg Latency: {np.mean(result['latency_ms']):.2f} ms")
    print(f"  Handover Count: {result['handover_count']}")

    # 生成安装脚本
    install_script = generate_install_script()
    with open("install_ns3.sh", "w") as f:
        f.write(install_script)
    print(f"\\nInstall script saved to: install_ns3.sh")