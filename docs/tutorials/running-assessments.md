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
python generate_compose.py --scenario scenarios/netheal/local_test.toml
```

### 4. Run Assessment (AgentBeats-style)

```bash
mkdir -p output
docker compose -f docker-compose.generated.yml up --abort-on-container-exit
```

### 5. View Results

```bash
cat output/results.json | jq .results[0].summary
```

## Scenario Configuration

Scenarios are defined in TOML format in `scenarios/netheal/`:

| File | Solver | Description |
|------|--------|-------------|
| `local_test.toml` | Dummy | Validation without API keys |

Note: the AgentBeats submission scenario lives in the leaderboard repo
(`netheal-agentbeats-leaderboard/scenario.toml`).

### Configuration Options

```toml
[config]
num_episodes = 100            # Number of assessment episodes
max_devices = 8               # Maximum network devices per episode
max_episode_steps = 25        # Tool call budget per episode
topology_types = ["star", "mesh", "hierarchical"]
timeout_seconds = 1200        # Max wall-clock time
seed = 1234                   # Random seed for reproducibility
enable_user_hints = true      # Provide natural language hints
episode_concurrency = 8       # Run episodes in parallel
episode_retry_limit = 10      # Retries per episode on timeout/error
max_timeouts = 100            # Allow this many timeouts before failing
max_errors = 10               # Allow this many errors before failing
extra_env_options = { hint_provider_mode = "heuristic" }

# Snapshot mode for reproducible assessments
use_snapshots = true          # Use pre-generated snapshots instead of random episodes
snapshot_path = "snapshots/v1/"  # Path to snapshot directory
auto_detect_num_episodes = true  # Auto-detect num_episodes from snapshot count
                                 # Set to false to use explicit num_episodes value
```

### Snapshot Mode

When `use_snapshots = true`, episodes are replayed from pre-generated snapshots for reproducibility. This ensures all agents are evaluated on identical scenarios.

- `auto_detect_num_episodes = true` (default): Automatically sets `num_episodes` to match the snapshot count
- `auto_detect_num_episodes = false`: Uses the explicit `num_episodes` value (useful for testing subsets)

## Assessment Metrics

| Metric | Description |
|--------|-------------|
| `diagnosis_success_rate` | Fraction of correct diagnoses |
| `fault_type_macro_f1` | F1 score across fault types |
| `composite_episode_score` | Overall performance score |
| `tool_cost_index` | Normalized diagnostic cost |
| `topology_coverage` | Network exploration coverage |

## Reproducibility Metadata

Each per-episode result includes:

- `episode_seed`: the exact seed used for that episode.
- `scenario_fingerprint`: SHA256 of the starting network graph + ground truth.

You can re-run a specific episode locally by setting `seed = <episode_seed>` and
`num_episodes = 1` in your scenario.

## Building a Custom Solver

To create your own solver agent:

1. Implement the A2A `/tasks` endpoint to receive `EpisodeStart` messages
2. Connect to the MCP server URL provided in the episode context
3. Use MCP tools to explore and diagnose the network
4. Call `submit_diagnosis` with your answer

See `netheal/aaa/purple_server.py` for a reference implementation.

## Monitoring

```bash
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
