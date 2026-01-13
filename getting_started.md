# NetHeal: Getting Started Guide

> Note: This guide has moved. See the canonical docs:
> - `docs/getting-started.md` (quickstart)
> - `docs/reference/environment.md` (API)
> - `docs/webapp.md` (Web App)
> - `docs/guides/training-sb3.md` (training)

**NetHeal** is a reinforcement learning environment designed to train AI agents to systematically troubleshoot and "heal" network problems. This guide will get you up to speed quickly on the project architecture, core concepts, and how to use it effectively.

## 🎯 **What is NetHeal?**

NetHeal simulates realistic network troubleshooting scenarios where an RL agent must:
1. **Discover** the network topology through exploration.
2. **Diagnose** problems using a variety of network tools (ping, traceroute, etc.).
3. **Make an accurate final diagnosis** to "heal" the network.

Think of it as training a virtual network engineer that can systematically identify and fix network issues.

## 🏗️ **Project Architecture**

```
netheal/
├── netheal/                    # Core package
│   ├── network/               # Network representation & topology generation
│   │   ├── graph.py          # NetworkGraph class (NetworkX-based)
│   │   └── topology.py       # TopologyGenerator (star, mesh, hierarchical, etc.)
│   ├── faults/               # Fault injection system
│   │   └── injector.py       # FaultInjector (device/link failures, performance issues)
│   ├── tools/                # Diagnostic tools simulation
│   │   └── simulator.py      # ToolSimulator (ping, traceroute, status checks)
│   ├── environment/          # RL Environment (Gymnasium compatible)
│   │   ├── env.py           # Main NetworkTroubleshootingEnv class
│   │   ├── actions.py       # Structured action space
│   │   ├── observation.py   # Graph-aware observations
│   │   └── rewards.py       # Sparse reward system
│   └── utils/               # Utilities and helpers
├── examples/                 # Usage examples and training data
├── tests/                   # Comprehensive test suite
└── requirements.txt         # Python dependencies
```

## 🌐 **Core Concepts**

### **Network Representation**
- **NetworkX-based graphs** with device nodes and connection edges
- **Device types**: Router, Switch, Server, Firewall, Host
- **Connection properties**: Bandwidth, latency, status
- **Topology types**: Linear, Star, Mesh, Hierarchical, Random

### **Fault Types**
- **Device Failure**: Complete device outage
- **Link Failure**: Connection disruption between devices
- **Performance Degradation**: Increased latency/reduced bandwidth
- **Misconfiguration**: Blocked ports, routing issues

### **Diagnostic Tools**
- **Ping**: Test connectivity with realistic latency simulation
- **Traceroute**: Path discovery and hop-by-hop analysis
- **Status Check**: Device operational state verification
- **Interface Check**: Connection status and properties

## 🤖 **How the RL Environment Works**

### **Observation Space** (What the agent sees)
The observation is a dictionary containing:
- **`discovery_matrix`**: An adjacency matrix representing the agent's current knowledge of the network topology.
- **`device_status`**: A matrix containing known properties and statuses of discovered devices.
- **`recent_diagnostics`**: A matrix storing the results of recent diagnostic actions.
- **`episode_metadata`**: A vector with progress indicators, like the current step count.

### **Action Space** (What the agent can do)
Actions are organized into 3 categories:

1.  **Topology Discovery**: Actions to discover the network structure (e.g., `scan_network`, `discover_neighbors`).
2.  **Diagnostic Actions**: Tools to investigate the network's health (e.g., `ping`, `traceroute`, `check_status`).
3.  **Final Diagnosis**: Terminal actions to diagnose the fault (e.g., `device_failure`, `link_failure`).

### **Reward System** (How the agent learns)
NetHeal uses a sparse, dynamic reward system:

- **Step Penalty**: A small penalty (`-0.1`) for every action to encourage efficiency.
- **Dynamic Final Diagnosis**: The reward for a correct or incorrect diagnosis is scaled based on the network size. This means that solving larger, more complex networks yields a proportionally larger reward (or penalty), providing a better learning signal.

## 📊 **Sample Network Scenarios**

### **Star Topology with Device Failure**
```
    edge_4 (Server)
         |
    edge_2 (Host) --- center (Router) --- edge_0 (Switch)
         |                    |
    edge_1 (Host)         edge_3 (Server)
```
**Fault**: `center` router fails → All inter-device communication breaks

### **Hierarchical Network with Link Failure**
```
Layer 0:     [Router_A] ---- [Router_B]
                |               |
Layer 1:   [Switch_1]      [Switch_2]
              |               |
Layer 2:   [Host_1]        [Host_2]
```
**Fault**: Link between `Router_A` and `Switch_1` fails → `Host_1` isolated


## 🚀 **Quick Start**

See the main README.md for installation instructions, basic usage examples, and training integration code.

## 🧪 **Testing & Development**

See README.md for testing instructions and development setup.

## 🎯 **Use Cases & Research Applications**

### **Network Operations**
- **Automated Troubleshooting**: Train agents to diagnose network faults
- **Network Monitoring**: Proactive fault detection and isolation
- **Incident Response**: Systematic approach to network problem resolution

### **Reinforcement Learning Research**
- **Graph-aware RL**: Learning on network topologies
- **Structured Action Spaces**: Hierarchical decision making
- **Multi-objective Rewards**: Balancing exploration vs exploitation
- **Outcome-Driven Learning**: Training agents to focus on long-term results.

### **Educational Applications**
- **Network Engineering Training**: Teach systematic troubleshooting
- **Simulation Environment**: Safe space to practice network diagnosis
- **Benchmarking**: Compare different troubleshooting strategies

## 🔬 **Advanced Features**

See README.md for advanced usage examples including custom network topologies, fault injection, and direct tool usage.

## 📈 **Performance Metrics**

### **Success Rates by Strategy**
- **Systematic**: ~15% success rate (methodical approach)
- **Targeted**: ~12% success rate (focused on likely faults)
- **Exploratory**: ~8% success rate (comprehensive search)

### **Episode Statistics**
- **Average Episode Length**: 10-13 steps
- **Network Sizes**: 3-8 devices

## 📚 **Next Steps**

1. **Read the README**: Complete installation and usage instructions
2. **Try the examples**: Start with `examples/basic_usage.py`
3. **Explore the data**: Run `examples/analyze_heuristic_data.py`
4. **Train your agent**: Use the RL training examples in README
5. **Customize**: Create your own network topologies and fault scenarios

---

**NetHeal** provides a rich, realistic environment for training AI agents to become expert network troubleshooters. The combination of graph-aware observations, a structured action space, and a sparse reward system creates a challenging and realistic testbed for developing intelligent network operations systems.

Happy troubleshooting! 🚀
