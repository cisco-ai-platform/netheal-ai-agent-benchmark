# Getting Started

This guide gets you running quickly with NetHeal for Python experiments.

## Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Core dependencies
pip install -r requirements.txt

# Optional: Add LLM support for intelligent hints
pip install -r requirements-llm.txt

# Optional: Add dev/test tools
pip install -r requirements-dev.txt
```

## Configuration (Optional)

NetHeal works out of the box with heuristic hints. To enable LLM-powered hints, set environment variables for your preferred provider:

### Azure OpenAI
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"  # Your deployment name
export AZURE_OPENAI_API_VERSION="2024-02-01"  # Optional
```

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4"  # Optional, defaults to gpt-4
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="your-api-key"
export ANTHROPIC_MODEL="claude-3-sonnet-20240229"  # Optional
```

### AWS Bedrock
```bash
export AWS_REGION="us-east-1"
export BEDROCK_MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0"
export AWS_ACCESS_KEY_ID="your-access-key"  # Or use AWS CLI config
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### Provider Selection
```bash
export LLM_PROVIDER="azure"  # Options: azure, openai, anthropic, bedrock
```

If `LLM_PROVIDER` is not set, NetHeal auto-detects based on which credentials are available. If no LLM is configured, heuristic hints are used automatically.

## Minimal loop

```python
from netheal import NetworkTroubleshootingEnv

env = NetworkTroubleshootingEnv(max_devices=6, max_episode_steps=15)
obs, info = env.reset(seed=0)

for step in range(20):
    valid = env.get_valid_actions()
    action = valid[0] if valid else env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Next steps

- Understand observations/actions: ./reference/environment.md
- Train an agent: ./guides/training-sb3.md
- Use the Web App: ./webapp.md
