# netheal-ai-agent-benchmark

netheal-ai-agent-benchmark is a reinforcement learning environment for training agents to systematically troubleshoot network problems through topology discovery and diagnostic testing.

## Documentation

- Full docs: `docs/index.md`
- Getting Started: `docs/getting-started.md`
- Environment API: `docs/reference/environment.md`
- Web App Guide: `docs/webapp.md`
- Training (SB3): `docs/guides/training-sb3.md`
- REST API: `docs/reference/web-api.md`
- Running Assessments: `docs/tutorials/running-assessments.md`

## Overview

netheal-ai-agent-benchmark provides a comprehensive simulation environment where RL agents learn to diagnose network faults using realistic troubleshooting methodologies. The environment features graph-aware observations, a structured action space, and a sparse reward system that encourages efficient, outcome-focused problem-solving.

### Key Capabilities

- **Interactive Network Discovery**: Agents build topology knowledge through exploration.
- **Structured Troubleshooting**: A clear, two-phase approach (discovery → diagnostics → diagnosis).
- **Realistic Fault Scenarios**: Device failures, link failures, performance degradation, and misconfigurations.
- **Sparse Rewards**: A reward system that promotes efficient, outcome-driven troubleshooting.

## Features

### Network Simulation
- **NetworkX-based Graphs**: Directed graph representation with device nodes and connection edges
- **Multiple Topologies**: Linear, star, mesh, hierarchical, and random network generation
- **Device Types**: Routers, switches, servers, firewalls, and hosts with realistic properties
- **Connection Properties**: Bandwidth, latency, and status modeling

### Fault Injection System
- **Device Failures**: Simulate complete device outages
- **Link Failures**: Network connection disruptions
- **Performance Degradation**: Latency increases and bandwidth limitations
- **Misconfigurations**: Blocked ports and routing issues

### Diagnostic Tools
- **Ping**: Connectivity testing with realistic latency simulation
- **Traceroute**: Path discovery and hop-by-hop analysis
- **Status Checks**: Device operational state verification
- **Interface Monitoring**: Connection status and properties inspection

### RL Environment
- **Structured Observations**: Graph-aware state representation with a topology discovery matrix and diagnostic history.
- **Hierarchical Actions**: Categorized action space (topology discovery, diagnostics, diagnosis).
- **Sparse Rewards**: A simple, outcome-based reward system.
- **Episode Management**: Configurable episode length and termination conditions

## Installation

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Setup

1. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
- `networkx>=3.1` - Network graph representation
- `gymnasium>=0.29.0` - RL environment interface
- `numpy>=1.24.0` - Numerical computations
- `matplotlib>=3.7.0` - Visualization (optional)
- `pytest>=7.4.0` - Testing framework
- `openai>=1.30.0` - Optional, for Azure OpenAI powered user hints
- `fastapi>=0.111.0` - Web API and A2A server
- `uvicorn>=0.30.0` - ASGI server
- `mcp>=1.0.0` - MCP protocol support
- `typer>=0.12.3` - CLI framework

## Quick Start

### Basic Usage

```python
from netheal import NetworkTroubleshootingEnv

# Create environment with custom parameters
env = NetworkTroubleshootingEnv(
    max_devices=8,
    max_episode_steps=20,
    topology_types=["star", "mesh", "hierarchical"],
    render_mode="text"
)

# Reset for new episode
observation, info = env.reset(seed=42)
print(f"Network size: {info['network_size']} devices")
print(f"Ground truth fault: {info['ground_truth_fault']}")
print(f"User hint: {info.get('user_hint')}")

# Interactive troubleshooting loop
for step in range(20):
    # Get valid actions based on current knowledge
    valid_actions = env.get_valid_actions()
    
    # Take action (replace with your agent's policy)
    action = valid_actions[0] if valid_actions else env.action_space.sample()
    
    # Execute action
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: Action {action}, Reward: {reward:.2f}")
    
    # Display current state
    env.render()
    
    if terminated or truncated:
        success = "SUCCESS" if reward > 0 else "FAILURE"
        print(f"Episode ended: {success}")
        break

env.close()
```

## Web Demo (FastAPI)

An interactive web UI is included to showcase netheal-ai-agent-benchmark episodes, hints, actions, and live observations.

### Run the Web Backend

1. Ensure your virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the FastAPI app with Uvicorn:

```bash
uvicorn webapp.backend.app.main:app --reload
```

3. Open the UI at:

- http://127.0.0.1:8000/

The UI lets you:

- Generate a new episode (configure seed, max devices/steps, hint mode)
- View the non-leaky user hint
- See valid actions and take steps
- Watch observations update (discovery matrix, recent diagnostics, metadata)

### API Endpoints

- `GET /api/health` – service health
- `POST /api/env/reset` – start a new episode (body: seed, max_devices, etc.)
- `GET /api/env/state` – current observation/info/valid actions
- `GET /api/env/actions` – valid action IDs and meanings
- `POST /api/env/step` – execute an action by ID

### API Tests

Run web API tests with FastAPI's TestClient:

```bash
pytest tests/test_web_api.py -q
```

## AAA Protocol Integration (Agent-to-Agent Assessment)

netheal-ai-agent-benchmark follows the AAA (Agentified Agent Assessment) format for standardized agent-to-agent evaluation. It provides a **green agent** (evaluator) that orchestrates assessments and exposes diagnostic tools to **purple agents** (solvers) via MCP.

### Quick Start with Docker

```bash
# Build the Docker image
docker build --platform linux/amd64 -t netheal-green-agent .

# Run the green agent
docker run -p 9020:9020 netheal-green-agent \
  --host 0.0.0.0 --port 9020 --card-url http://localhost:9020

# Test the agent card
curl http://localhost:9020/.well-known/agent.json

# Create an assessment task
curl -X POST http://localhost:9020/tasks \
  -H "Content-Type: application/json" \
  -d '{"config": {"num_episodes": 3, "max_devices": 6}}'
```

### A2A Server Endpoints

The green agent exposes AAA-compatible endpoints:

- `GET /.well-known/agent.json` - Agent card with capabilities
- `POST /tasks` - Create a new assessment task
- `GET /tasks/{id}` - Get task status and results
- `GET /tasks/{id}/stream` - SSE stream for real-time updates

### CLI Options

```bash
python -m netheal.aaa.cli --help

Options:
  --host       Host address to bind (default: 0.0.0.0)
  --port       Port number (default: 9020)
  --card-url   URL advertised in agent card
  --log-level  Logging level (default: info)
```

### Running with Docker Compose

```bash
# Start the green agent
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### MCP Tool Server

During assessments, the green agent starts a per-episode MCP server exposing diagnostic tools to purple agents:

- `scan_network` - Discover network devices
- `discover_neighbors` - Find device connections
- `ping` - Test connectivity between devices
- `traceroute` - Trace network path
- `check_status` - Check device status
- `check_interfaces` - Inspect device interfaces
- `submit_diagnosis` - Submit final fault diagnosis

### Running the Demo

```bash
# Basic demo
python -m netheal.aaa.demo --seed 42 --devices 6

# Step-by-step mode
python -m netheal.aaa.demo --seed 42 --devices 6 --step --verbose
```

### Environment Variables (Optional)

For LLM-powered hints, set Azure OpenAI credentials:

```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
export AZURE_OPENAI_DEPLOYMENT=gpt-5
```

### Advanced Usage with Direct Components

```python
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType
from netheal.tools.simulator import ToolSimulator

# Create custom network
network = TopologyGenerator.generate_hierarchical_topology(
    num_layers=3, 
    devices_per_layer=[2, 3, 4]
)

# Inject specific fault
fault_injector = FaultInjector(network)
fault = fault_injector.inject_device_failure("L1_D0")

# Use diagnostic tools directly
tool_sim = ToolSimulator(network)
result = tool_sim.ping("L0_D0", "L2_D3")
print(f"Ping result: {result}")
```

## Architecture

### Project Structure

```
netheal/
├── netheal/
│   ├── __init__.py                    # Main package exports
│   ├── network/                       # Network representation
│   │   ├── __init__.py
│   │   ├── graph.py                   # NetworkGraph class with device/connection modeling
│   │   └── topology.py                # TopologyGenerator for various network types
│   ├── faults/                        # Fault injection system
│   │   ├── __init__.py
│   │   └── injector.py                # FaultInjector with multiple fault types
│   ├── tools/                         # Diagnostic tools simulation
│   │   ├── __init__.py
│   │   └── simulator.py               # ToolSimulator with realistic tool behavior
│   ├── environment/                   # RL environment components
│   │   ├── __init__.py
│   │   ├── env.py                     # Main NetworkTroubleshootingEnv class
│   │   ├── actions.py                 # Structured action space management
│   │   ├── observation.py             # Graph-aware observation system
│   │   └── rewards.py                 # Multi-component reward calculation
│   ├── hints/                         # Natural language hint providers
│   │   ├── __init__.py
│   │   └── provider.py                # Azure GPT + heuristic fallback
│   ├── aaa/                           # AAA protocol integration
│   │   ├── __init__.py
│   │   ├── cli.py                     # CLI entrypoint for Docker
│   │   ├── server.py                  # A2A FastAPI server
│   │   ├── green_agent.py             # Green agent orchestrator
│   │   ├── mcp_server.py              # MCP tool server for purple agents
│   │   ├── schemas.py                 # Pydantic models for A2A protocol
│   │   ├── demo.py                    # Interactive demo runner
│   │   └── dummy_agent.py             # Baseline purple agent
│   └── evaluation/                    # Metrics and evaluation
│       ├── __init__.py
│       ├── metrics.py                 # CompetitionEvaluator
│       ├── wrapper.py                 # MetricsCollectorWrapper
│       └── aaa.py                     # AAA payload builder
├── webapp/                            # Web demo
│   ├── backend/
│   │   ├── __init__.py
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py               # FastAPI app and API endpoints
│   │       ├── manager.py            # EnvManager for singleton demo env
│   │       └── schemas.py            # Pydantic request models
│   └── frontend/
│       ├── index.html                # Minimal, modern UI
│       ├── style.css                 # Styles
│       └── app.js                    # Frontend logic
├── examples/                          # Usage examples and demos
│   ├── basic_usage.py                 # Getting started examples
│   └── ...
├── tests/                             # Unit tests
│   ├── test_environment.py            # Environment testing
│   ├── test_actions.py                # Action space testing
│   └── ...
├── scenarios/                         # AAA scenario definitions
│   └── netheal/scenario.toml
├── Dockerfile                         # Docker build for AAA
├── docker-compose.yml                 # Local testing orchestration
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Core Components

#### 1. Network Representation (`network/`)
- **NetworkGraph**: Directed graph with device nodes and connection edges
- **TopologyGenerator**: Creates realistic network topologies (linear, star, mesh, hierarchical, random)
- **Device Types**: Router, switch, server, firewall, host with properties
- **Connection Properties**: Bandwidth, latency, status tracking

#### 2. Fault Injection (`faults/`)
- **FaultInjector**: Programmatic fault introduction
- **Fault Types**: Device failure, link failure, performance degradation, misconfiguration
- **Fault Management**: Active fault tracking and restoration capabilities

#### 3. Diagnostic Tools (`tools/`)
- **ToolSimulator**: Realistic network diagnostic tool simulation
- **Available Tools**: ping, traceroute, check_status, check_interfaces
- **Cost Modeling**: Each tool has associated execution costs

#### 4. RL Environment (`environment/`)
- **NetworkTroubleshootingEnv**: Main Gymnasium-compatible environment
- **Structured Actions**: Hierarchical action space with categories
- **Graph-Aware Observations**: Topology discovery matrix and diagnostic memory.
- **Sparse Rewards**: Outcome-based reward system focused on efficiency and accuracy.

## Action Space

The environment provides a structured action space organized into three categories:

### 1. Topology Discovery
- `scan_network`: Broad network discovery.
- `discover_neighbors`: Find connections from a specific device.

### 2. Diagnostic Actions
- `ping`: Test connectivity between two devices.
- `traceroute`: Trace the network path between two devices.
- `check_status`: Verify the operational state of a single device.
- `check_interfaces`: Examine the network interfaces of a single device.

### 3. Final Diagnosis
- `device_failure(device)`: Diagnose a device failure.
- `link_failure(device_a, device_b)`: Diagnose a link failure.
- `misconfiguration(device)`: Diagnose a device misconfiguration.
- `performance_degradation(link)`: Diagnose a performance issue on a link.

## Observation Space

The environment provides a structured observation as a dictionary containing the following components:

- **Discovery Matrix**: An adjacency matrix representing the agent's current knowledge of the network topology.
- **Device Status**: A matrix containing known properties and operational statuses of discovered devices.
- **Diagnostic History**: A matrix storing the results of the most recent diagnostic actions.
- **Episode Metadata**: A vector with progress indicators, such as the current step count.

## Reward System

netheal-ai-agent-benchmark uses a sparse, dynamic reward system to encourage efficient and accurate troubleshooting. The reward is scaled based on the complexity of the network to provide a more calibrated learning signal.

- **Step Penalty**: A small, constant penalty (`-0.1`) is applied for every action taken. This incentivizes the agent to solve the problem in the fewest steps possible.
- **Dynamic Final Diagnosis**: The reward for the final diagnosis is scaled based on the number of devices in the network. A correct diagnosis yields a positive reward, while an incorrect one yields a penalty. This ensures that solving more complex problems is appropriately incentivized.

This outcome-focused approach challenges the agent to learn effective long-term strategies rather than optimizing for intermediate goals.

## User Hints (Natural Language)

- The environment can provide a short, non-leaky natural language hint at reset to help agents start with a direction (without revealing the answer).
- Hints are returned in `info['user_hint']` from `env.reset()` and are also included in subsequent `info` dicts.

### Configuration

```python
env = NetworkTroubleshootingEnv(
    enable_user_hints=True,          # default True
    hint_provider_mode="auto",       # "auto" | "azure" | "heuristic"
    user_context={"access_point": "Guest-WiFi"}  # optional context for hint wording
)
```

### Azure OpenAI (optional)

If Azure OpenAI environment variables are set, `hint_provider_mode="auto"` uses GPT models via Azure to generate hints; otherwise a deterministic heuristic is used.

Required environment variables:

```bash
export AZURE_OPENAI_ENDPOINT=...           # e.g., https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
export AZURE_OPENAI_DEPLOYMENT=gpt-5  # or your deployment name
```

Security note: Ground truth is provided to the hint generator but sanitized; hints avoid explicit fault types (e.g., "misconfiguration") and internal IDs.

## Examples

### Running Examples

```bash
# Basic usage demonstration
python examples/basic_usage.py
```

### Evaluation & Metrics

Wrap the environment with `netheal.evaluation.MetricsCollectorWrapper` to collect episode traces, compute metrics (DSR, macro F1, TTD, tool cost index, topology coverage, evidence sufficiency, redundancy, discovery efficiency), and export AAA payloads without modifying the base env:

```python
from netheal import NetworkTroubleshootingEnv
from netheal.evaluation import MetricsCollectorWrapper, build_aaa_payload

env = MetricsCollectorWrapper(NetworkTroubleshootingEnv())
# ... agent loop
summary = env.evaluator.compute_summary()
payload = build_aaa_payload(env.evaluator, purple_agent_id="solver_v1")
```

See `tests/test_evaluation_metrics.py` for usage examples and regression tests.

### Training Integration

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = NetworkTroubleshootingEnv(max_devices=6, max_episode_steps=15)

# Train with Stable Baselines3
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate trained agent
obs, info = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=netheal

# Run specific test file
pytest tests/test_environment.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Research Applications

netheal-ai-agent-benchmark is designed for research in:
- **Network Troubleshooting Automation**: Training agents to systematically diagnose network faults
- **Reinforcement Learning**: Graph-aware RL with structured action spaces
- **Network Operations**: Developing AI-assisted network management tools
- **Fault Diagnosis**: Automated, evidence-based reasoning.

## License

Apache 2.0 License - see LICENSE file for details.

## Citation

If you use netheal-ai-agent-benchmark in your research, please cite:

```bibtex
@software{netheal2026,
  title={NetHeal AI Agent Benchmark},
  author={Ashkan Kazemi, Cisco AI},
  year={2026},
  url={https://github.com/cisco-open/netheal-ai-agent-benchmark}
}
```
