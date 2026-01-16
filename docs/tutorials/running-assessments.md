# Running NetHeal Assessments

This guide covers how to run NetHeal assessments using the AgentBeats AAA (Agentified Agent Assessment) protocol, either locally or on the AgentBeats platform.

## Architecture Overview

NetHeal implements the AAA protocol with two agent roles:

- **Green Agent (Evaluator)**: Orchestrates assessments, manages the RL environment, exposes diagnostic tools via MCP, and computes performance metrics
- **Purple Agent (Solver)**: Receives tasks from the green agent, uses MCP tools to diagnose network faults

```
┌─────────────────┐         ┌─────────────────┐
│   Green Agent   │◄───────►│  Purple Agent   │
│   (Evaluator)   │  A2A    │    (Solver)     │
│                 │         │                 │
│  ┌───────────┐  │         │  ┌───────────┐  │
│  │ MCP Server│◄─┼─────────┼──│ MCP Client│  │
│  │  (Tools)  │  │         │  │           │  │
│  └───────────┘  │         │  └───────────┘  │
└─────────────────┘         └─────────────────┘
```

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- LLM credentials (Azure/OpenAI/Anthropic/Bedrock) for GPT-powered solver or hints

## Quick Start

### 1. Build Docker Images

```bash
# Build green agent (evaluator)
docker build -f Dockerfile -t netheal-green-agent:latest .

# Build purple agent (solver)
docker build -f Dockerfile.purple -t netheal-purple-agent:latest .
```

### 2. Configure Environment

Optionally create a `.env` file with your LLM credentials (leave blank to use heuristic hints):

```bash
cat > .env <<'EOF'
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=
OPENAI_API_KEY=
OPENAI_MODEL=
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=
AWS_REGION=
BEDROCK_MODEL_ID=
EOF
```

### 3. Generate Docker Compose

```bash
python generate_compose.py --scenario scenarios/netheal/scenario.toml
```

### 4. Run Assessment

```bash
mkdir -p output
docker compose -f docker-compose.generated.yml up -d
```

### 5. Submit Task

```bash
curl -X POST http://localhost:9020/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "assessment-001",
    "participants": {
      "purple": {
        "role": "purple_agent",
        "endpoint": "http://netheal-solver:9030"
      }
    }
  }'
```

### 6. View Results

```bash
cat output/assessment_results.json | jq .summary
```

## Scenario Configuration

Scenarios are defined in TOML format in `scenarios/netheal/`:

| File | Solver | Description |
|------|--------|-------------|
| `scenario.toml` | GPT | Production configuration |
| `local_test.toml` | Dummy | Validation without API keys |

### Configuration Options

```toml
[config]
num_episodes = 5              # Number of assessment episodes
max_devices = 8               # Maximum network devices per episode
max_episode_steps = 25        # Tool call budget per episode
topology_types = ["star", "mesh", "hierarchical"]
timeout_seconds = 300         # Max wall-clock time
seed = 1234                   # Random seed for reproducibility
enable_user_hints = true      # Provide natural language hints
```

## Assessment Metrics

| Metric | Description |
|--------|-------------|
| `diagnosis_success_rate` | Fraction of correct diagnoses |
| `fault_type_macro_f1` | F1 score across fault types |
| `composite_episode_score` | Overall performance score |
| `tool_cost_index` | Normalized diagnostic cost |
| `topology_coverage` | Network exploration coverage |

## Building a Custom Solver

To create your own solver agent:

1. Implement the A2A `/tasks` endpoint to receive `EpisodeStart` messages
2. Connect to the MCP server URL provided in the episode context
3. Use MCP tools to explore and diagnose the network
4. Call `submit_diagnosis` with your answer

See `netheal/aaa/purple_server.py` for a reference implementation.

## Monitoring

```bash
# Check task status
curl http://localhost:9020/tasks/{task_id} | jq .status

# Stream real-time updates
curl -N http://localhost:9020/tasks/{task_id}/stream

# View logs
docker compose -f docker-compose.generated.yml logs -f
```

## Cleanup

```bash
docker compose -f docker-compose.generated.yml down
```

## Related Documentation

- [Architecture Overview](../explanation/architecture.md)
- [Environment API Reference](../reference/environment.md)
- [AgentBeats Platform](https://docs.agentbeats.dev/)
