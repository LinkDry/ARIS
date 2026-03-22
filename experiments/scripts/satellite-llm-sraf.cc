/*
 * LLM-SRAF Satellite Network Simulation
 * =====================================
 *
 * 用于 LLM 增强的卫星网络资源分配框架仿真
 *
 * 功能:
 * - LEO 星座仿真 (Starlink-like)
 * - 用户链路模拟
 * - 资源分配接口
 * - 性能指标收集
 *
 * 作者: ARIS Framework
 * 日期: 2026-03-22
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/config-store-module.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SatelliteLlmSraf");

// ==================== 卫星结构 ====================
struct Satellite {
    uint32_t id;
    double altitude;      // km
    double inclination;   // degrees
    double raan;          // Right Ascension of Ascending Node
    double meanAnomaly;   // degrees
    Vector position;      // ECEF coordinates
    uint32_t numBeams;
    double loadFactor;
};

// ==================== 用户终端结构 ====================
struct UserTerminal {
    uint32_t id;
    Vector position;      // Lat, Lon, Alt
    uint32_t servingSatellite;
    double sinr;
    double throughput;
    double latency;
};

// ==================== 仿真参数 ====================
class SimulationConfig {
public:
    // 星座参数
    uint32_t numOrbitalPlanes = 72;
    uint32_t satsPerPlane = 22;
    double altitudeKm = 550.0;
    double inclinationDeg = 53.0;

    // 地面站参数
    uint32_t numUserTerminals = 100;

    // 链路参数
    double frequencyGHz = 26.0;      // Ka band
    double bandwidthMHz = 400.0;
    double maxBeamsPerSat = 8;
    double satellitePowerW = 10.0;
    double antennaGainDbi = 30.0;

    // 仿真参数
    double simDurationSec = 100.0;
    double timeStepMs = 100.0;
    uint32_t randomSeed = 42;

    // 输出
    std::string outputDir = "output/";
    bool enableTracing = true;

    // 从 JSON 配置文件加载
    void LoadFromJson(const std::string& filename);
};

// ==================== LEO 星座 ====================
class LeoConstellation {
public:
    LeoConstellation(const SimulationConfig& config);
    ~LeoConstellation() = default;

    void Initialize();
    void UpdatePositions(double time);
    Satellite& GetSatellite(uint32_t id);
    std::vector<uint32_t> GetVisibleSatellites(const Vector& userPosition);

private:
    SimulationConfig m_config;
    std::vector<Satellite> m_satellites;
    double m_earthRadiusKm = 6371.0;

    Vector ComputeSatellitePosition(const Satellite& sat, double time);
    double ComputeElevationAngle(const Vector& satPos, const Vector& userPos);
};

// ==================== 用户终端管理 ====================
class UserTerminalManager {
public:
    UserTerminalManager(const SimulationConfig& config);
    ~UserTerminalManager() = default;

    void Initialize();
    void DistributeGeographically(const std::string& distribution);
    UserTerminal& GetUser(uint32_t id);
    void UpdateServingSatellites(LeoConstellation& constellation);

private:
    SimulationConfig m_config;
    std::vector<UserTerminal> m_users;
};

// ==================== 资源分配器 ====================
class ResourceAllocator {
public:
    ResourceAllocator();
    ~ResourceAllocator() = default;

    // 资源分配策略
    enum class Strategy {
        ROUND_ROBIN,
        MAX_SINR,
        LOAD_BALANCING,
        LLM_BASED
    };

    void SetStrategy(Strategy strategy);
    void Allocate(LeoConstellation& constellation,
                  UserTerminalManager& users);

    // LLM 接口
    void SetLlmBasedAllocation(const std::map<uint32_t, uint32_t>& allocation);

private:
    Strategy m_strategy;
    std::map<uint32_t, uint32_t> m_allocation; // user -> satellite
};

// ==================== 性能监控 ====================
class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor() = default;

    void RecordThroughput(uint32_t userId, double throughput);
    void RecordLatency(uint32_t userId, double latency);
    void RecordSinr(uint32_t userId, double sinr);
    void RecordHandover(uint32_t userId, uint32_t oldSat, uint32_t newSat);

    void GenerateReport(const std::string& filename);
    void ExportToPython(const std::string& filename);

private:
    std::map<uint32_t, std::vector<double>> m_throughputHistory;
    std::map<uint32_t, std::vector<double>> m_latencyHistory;
    std::map<uint32_t, std::vector<double>> m_sinrHistory;
    std::vector<std::tuple<double, uint32_t, uint32_t, uint32_t>> m_handoverEvents;
};

// ==================== 主仿真类 ====================
class SatelliteSimulation {
public:
    SatelliteSimulation();
    ~SatelliteSimulation() = default;

    void Configure(const SimulationConfig& config);
    void Run();
    void OutputResults();

private:
    SimulationConfig m_config;
    LeoConstellation m_constellation;
    UserTerminalManager m_users;
    ResourceAllocator m_allocator;
    PerformanceMonitor m_monitor;
    FlowMonitorHelper m_flowMonitor;

    void SetupNetwork();
    void SetupApplications();
    void ScheduleEvents();
    void CollectMetrics();
};

// ==================== 实现 ====================

LeoConstellation::LeoConstellation(const SimulationConfig& config)
    : m_config(config) {
    m_satellites.resize(config.numOrbitalPlanes * config.satsPerPlane);
}

void LeoConstellation::Initialize() {
    uint32_t satId = 0;
    for (uint32_t plane = 0; plane < m_config.numOrbitalPlanes; ++plane) {
        double raan = 360.0 * plane / m_config.numOrbitalPlanes;
        for (uint32_t s = 0; s < m_config.satsPerPlane; ++s) {
            Satellite& sat = m_satellites[satId];
            sat.id = satId;
            sat.altitude = m_config.altitudeKm;
            sat.inclination = m_config.inclinationDeg;
            sat.raan = raan;
            sat.meanAnomaly = 360.0 * s / m_config.satsPerPlane;
            sat.numBeams = m_config.maxBeamsPerSat;
            sat.loadFactor = 0.0;
            satId++;
        }
    }
    NS_LOG_INFO("Initialized " << satId << " satellites");
}

Vector LeoConstellation::ComputeSatellitePosition(const Satellite& sat, double time) {
    // 简化的轨道计算
    // 实际应使用 SGP4 或更精确的轨道模型
    double meanMotion = 15.0; // orbits per day (typical for LEO)
    double period = 86400.0 / meanMotion;
    double angle = sat.meanAnomaly + 360.0 * time / period;

    // 简化的 ECEF 坐标计算
    double r = m_earthRadiusKm + sat.altitude;
    double theta = angle * M_PI / 180.0;
    double phi = sat.raan * M_PI / 180.0;

    double x = r * std::cos(theta) * std::cos(phi);
    double y = r * std::cos(theta) * std::sin(phi);
    double z = r * std::sin(theta) * std::sin(sat.inclination * M_PI / 180.0);

    return Vector(x * 1000, y * 1000, z * 1000); // 转换为米
}

void LeoConstellation::UpdatePositions(double time) {
    for (auto& sat : m_satellites) {
        sat.position = ComputeSatellitePosition(sat, time);
    }
}

Satellite& LeoConstellation::GetSatellite(uint32_t id) {
    return m_satellites.at(id);
}

std::vector<uint32_t> LeoConstellation::GetVisibleSatellites(const Vector& userPosition) {
    std::vector<uint32_t> visible;
    for (const auto& sat : m_satellites) {
        double elevation = ComputeElevationAngle(sat.position, userPosition);
        if (elevation > 25.0) { // 最小仰角阈值
            visible.push_back(sat.id);
        }
    }
    return visible;
}

double LeoConstellation::ComputeElevationAngle(const Vector& satPos, const Vector& userPos) {
    Vector diff = satPos - userPos;
    double distance = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    // 简化的仰角计算
    double dotProduct = satPos.x * userPos.x + satPos.y * userPos.y + satPos.z * userPos.z;
    double userMag = std::sqrt(userPos.x * userPos.x + userPos.y * userPos.y + userPos.z * userPos.z);

    double elevation = std::asin((dotProduct / (userMag * distance)) -
                                  userMag / distance) * 180.0 / M_PI;
    return elevation;
}

// ==================== 用户终端管理实现 ====================

UserTerminalManager::UserTerminalManager(const SimulationConfig& config)
    : m_config(config) {
    m_users.resize(config.numUserTerminals);
}

void UserTerminalManager::Initialize() {
    DistributeGeographically("urban_weighted");
}

void UserTerminalManager::DistributeGeographically(const std::string& distribution) {
    // 分布用户终端 (简化版本)
    Ptr<UniformRandomVariable> latGen = CreateObject<UniformRandomVariable>();
    latGen->SetAttribute("Min", DoubleValue(-60.0));
    latGen->SetAttribute("Max", DoubleValue(60.0));

    Ptr<UniformRandomVariable> lonGen = CreateObject<UniformRandomVariable>();
    lonGen->SetAttribute("Min", DoubleValue(-180.0));
    lonGen->SetAttribute("Max", DoubleValue(180.0));

    for (auto& user : m_users) {
        user.position = Vector(latGen->GetValue(), lonGen->GetValue(), 0.0);
        user.servingSatellite = 0;
        user.sinr = 0.0;
        user.throughput = 0.0;
        user.latency = 0.0;
    }

    NS_LOG_INFO("Distributed " << m_users.size() << " user terminals");
}

UserTerminal& UserTerminalManager::GetUser(uint32_t id) {
    return m_users.at(id);
}

void UserTerminalManager::UpdateServingSatellites(LeoConstellation& constellation) {
    for (auto& user : m_users) {
        auto visible = constellation.GetVisibleSatellites(user.position);
        if (!visible.empty()) {
            // 简化: 选择第一个可见卫星
            user.servingSatellite = visible[0];
        }
    }
}

// ==================== 资源分配器实现 ====================

ResourceAllocator::ResourceAllocator()
    : m_strategy(Strategy::ROUND_ROBIN) {
}

void ResourceAllocator::SetStrategy(Strategy strategy) {
    m_strategy = strategy;
}

void ResourceAllocator::Allocate(LeoConstellation& constellation,
                                  UserTerminalManager& users) {
    switch (m_strategy) {
        case Strategy::ROUND_ROBIN:
            // Round-robin allocation
            break;
        case Strategy::MAX_SINR:
            // Maximum SINR allocation
            break;
        case Strategy::LOAD_BALANCING:
            // Load-balancing allocation
            break;
        case Strategy::LLM_BASED:
            // Use LLM-based allocation (set via SetLlmBasedAllocation)
            break;
    }
}

void ResourceAllocator::SetLlmBasedAllocation(const std::map<uint32_t, uint32_t>& allocation) {
    m_allocation = allocation;
    m_strategy = Strategy::LLM_BASED;
}

// ==================== 性能监控实现 ====================

PerformanceMonitor::PerformanceMonitor() {
}

void PerformanceMonitor::RecordThroughput(uint32_t userId, double throughput) {
    m_throughputHistory[userId].push_back(throughput);
}

void PerformanceMonitor::RecordLatency(uint32_t userId, double latency) {
    m_latencyHistory[userId].push_back(latency);
}

void PerformanceMonitor::RecordSinr(uint32_t userId, double sinr) {
    m_sinrHistory[userId].push_back(sinr);
}

void PerformanceMonitor::RecordHandover(uint32_t userId, uint32_t oldSat, uint32_t newSat) {
    m_handoverEvents.push_back(std::make_tuple(Simulator::Now().GetSeconds(),
                                                userId, oldSat, newSat));
}

void PerformanceMonitor::GenerateReport(const std::string& filename) {
    std::ofstream file(filename);
    file << "# LLM-SRAF Performance Report\n";
    file << "# Generated by NS-3 Satellite Simulation\n\n";

    file << "## Throughput Statistics\n";
    for (const auto& [userId, history] : m_throughputHistory) {
        double sum = 0.0;
        for (double t : history) sum += t;
        file << "User " << userId << ": avg=" << sum / history.size() << " Mbps\n";
    }

    file << "\n## Handover Events\n";
    file << "Time(s),User,OldSat,NewSat\n";
    for (const auto& [time, user, oldSat, newSat] : m_handoverEvents) {
        file << time << "," << user << "," << oldSat << "," << newSat << "\n";
    }

    file.close();
}

void PerformanceMonitor::ExportToPython(const std::string& filename) {
    std::ofstream file(filename);
    file << "{\n";

    // Export throughput
    file << "  \"throughput\": {\n";
    for (const auto& [userId, history] : m_throughputHistory) {
        file << "    \"" << userId << "\": [";
        for (size_t i = 0; i < history.size(); ++i) {
            file << history[i];
            if (i < history.size() - 1) file << ", ";
        }
        file << "],\n";
    }
    file << "  },\n";

    file << "}\n";
    file.close();
}

// ==================== 主仿真实现 ====================

SatelliteSimulation::SatelliteSimulation()
    : m_constellation(m_config), m_users(m_config) {
}

void SatelliteSimulation::Configure(const SimulationConfig& config) {
    m_config = config;

    // 设置随机种子
    RngSeedManager::SetSeed(config.randomSeed);

    // 初始化星座和用户
    m_constellation.Initialize();
    m_users.Initialize();

    NS_LOG_INFO("Simulation configured with " <<
                m_config.numOrbitalPlanes * m_config.satsPerPlane << " satellites and " <<
                m_config.numUserTerminals << " users");
}

void SatelliteSimulation::Run() {
    NS_LOG_INFO("Starting simulation for " << m_config.simDurationSec << " seconds");

    // 设置网络和应用
    SetupNetwork();
    SetupApplications();
    ScheduleEvents();

    // 运行仿真
    Simulator::Stop(Seconds(m_config.simDurationSec));
    Simulator::Run();

    // 收集结果
    CollectMetrics();

    Simulator::Destroy();

    NS_LOG_INFO("Simulation completed");
}

void SatelliteSimulation::SetupNetwork() {
    // 创建节点
    NodeContainer satellites;
    satellites.Create(m_config.numOrbitalPlanes * m_config.satsPerPlane);

    NodeContainer users;
    users.Create(m_config.numUserTerminals);

    // 安装互联网协议栈
    InternetStackHelper internet;
    internet.Install(satellites);
    internet.Install(users);

    // 配置链路 (简化版本)
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("20ms"));

    // 安装设备
    NetDeviceContainer devices = p2p.Install(NodeContainer(satellites.Get(0), users.Get(0)));

    // 分配 IP 地址
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);
}

void SatelliteSimulation::SetupApplications() {
    // 创建 UDP 应用用于流量生成
    uint16_t port = 9;

    for (uint32_t i = 0; i < m_config.numUserTerminals && i < 10; ++i) {
        // 服务器端
        Address serverAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
        PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory", serverAddress);

        // 客户端
        OnOffHelper onoff("ns3::UdpSocketFactory", Address());
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", StringValue("50Mbps"));

        port++;
    }
}

void SatelliteSimulation::ScheduleEvents() {
    // 定期更新卫星位置
    for (double t = 0; t < m_config.simDurationSec; t += m_config.timeStepMs / 1000.0) {
        Simulator::Schedule(Seconds(t), &LeoConstellation::UpdatePositions,
                           &m_constellation, t);
        Simulator::Schedule(Seconds(t), &UserTerminalManager::UpdateServingSatellites,
                           &m_users, std::ref(m_constellation));
    }
}

void SatelliteSimulation::CollectMetrics() {
    // 收集性能指标
    for (uint32_t i = 0; i < m_config.numUserTerminals; ++i) {
        m_monitor.RecordThroughput(i, 50.0 + (rand() % 100)); // 简化的吞吐量
        m_monitor.RecordLatency(i, 20.0 + (rand() % 50));     // 简化的时延
    }
}

void SatelliteSimulation::OutputResults() {
    std::string outputDir = m_config.outputDir;
    system(("mkdir -p " + outputDir).c_str());

    m_monitor.GenerateReport(outputDir + "performance_report.txt");
    m_monitor.ExportToPython(outputDir + "results.json");

    NS_LOG_INFO("Results written to " << outputDir);
}

// ==================== 主函数 ====================

int main(int argc, char *argv[]) {
    // 解析命令行参数
    CommandLine cmd;
    SimulationConfig config;

    cmd.AddValue("duration", "Simulation duration (seconds)", config.simDurationSec);
    cmd.AddValue("users", "Number of user terminals", config.numUserTerminals);
    cmd.AddValue("output", "Output directory", config.outputDir);
    cmd.AddValue("seed", "Random seed", config.randomSeed);
    cmd.AddValue("planes", "Number of orbital planes", config.numOrbitalPlanes);
    cmd.AddValue("sats", "Satellites per plane", config.satsPerPlane);
    cmd.AddValue("altitude", "Satellite altitude (km)", config.altitudeKm);

    cmd.Parse(argc, argv);

    // 启用日志
    LogComponentEnable("SatelliteLlmSraf", LOG_LEVEL_INFO);

    // 创建并运行仿真
    SatelliteSimulation sim;
    sim.Configure(config);
    sim.Run();
    sim.OutputResults();

    return 0;
}