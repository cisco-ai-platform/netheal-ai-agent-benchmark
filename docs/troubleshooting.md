# Troubleshooting

This guide covers common issues and their solutions when working with the NetHeal benchmark.

## Web Demo Issues

### Port Already in Use

**Symptom:** `Address already in use` or similar error when starting the server.

**Solution:** Run uvicorn on a different port:
```bash
uvicorn webapp.backend.app.main:app --port 8001
```

### CORS or 404 Errors

**Symptom:** Browser shows CORS errors or 404 when accessing the web demo.

**Solution:**
- Ensure you open `http://127.0.0.1:8000/` (not `localhost` or `0.0.0.0`)
- Verify `webapp/frontend/index.html` exists
- Check that the backend is running before accessing the frontend

### "Environment not initialized" Error

**Symptom:** API returns "Environment not initialized" when calling `/api/env/state`.

**Solution:** Call `/api/env/reset` first to initialize the environment before querying state.

---

## Environment & Observations

### Observation Shape Errors

**Symptom:** RL library complains about observation shape mismatch.

**Solution:** NetHeal observations are dictionaries with these keys:
- `discovery_matrix`: (max_devices, max_devices) adjacency matrix
- `device_status`: (max_devices, 10) device state array
- `recent_diagnostics`: (10, 6) recent tool results
- `episode_metadata`: (4,) step count and progress

Use dict-aware policies or flatten the observation manually. Example with Stable Baselines3:
```python
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation

env = FlattenObservation(NetworkTroubleshootingEnv())
```

### Action Errors / Invalid Action IDs

**Symptom:** `ValueError: Invalid action` when calling `env.step()`.

**Solution:** Always get valid actions before stepping:
```python
valid_actions = env.get_valid_actions()
action = random.choice(list(valid_actions.keys()))
obs, reward, terminated, truncated, info = env.step(action)
```

Action IDs are dynamically assigned based on network topology. Don't hardcode action IDs.

---

## LLM Hint Provider Issues

### Hints Not Generating (Fallback to Heuristic)

**Symptom:** Hints are generic/heuristic even though LLM is configured.

**Possible causes:**
1. Missing or invalid API credentials
2. Network connectivity issues
3. LLM provider library not installed

**Solution:** Check your environment variables are set correctly:
```bash
# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-5"

# Or OpenAI
export OPENAI_API_KEY="your-key"

# Or Anthropic
export ANTHROPIC_API_KEY="your-key"

# Or AWS Bedrock
export AWS_REGION="us-east-1"
export BEDROCK_MODEL_ID="anthropic.claude-3-sonnet"
```

Also ensure the provider library is installed:
```bash
pip install openai      # For Azure or OpenAI
pip install anthropic   # For Anthropic
pip install boto3       # For AWS Bedrock
```

### LLM API Key Errors

**Symptom:** `AuthenticationError` or `InvalidAPIKeyError`.

**Solution:**
- Verify your API key is correct and not expired
- Check you're using the right environment variable for your provider
- For Azure, ensure `AZURE_OPENAI_ENDPOINT` includes the full URL

### LLM Rate Limiting

**Symptom:** `RateLimitError` or 429 responses.

**Solution:**
- Reduce request frequency
- Consider using the heuristic provider for development/testing
- Check your API quota/billing

### LLM Connection Timeouts

**Symptom:** Requests hang or timeout.

**Solution:**
- Check network connectivity
- Verify firewall/proxy settings
- For Bedrock, ensure AWS credentials have necessary permissions

---

## Docker & Deployment

### Docker Build Fails

**Symptom:** Docker build fails with dependency errors.

**Solution:**
```bash
# Ensure you're building from the project root
docker build -t netheal-green-agent .

# If caching issues, force rebuild
docker build --no-cache -t netheal-green-agent .
```

### Docker Compose Port Conflicts

**Symptom:** Container fails to start due to port conflicts.

**Solution:** Edit `docker-compose.yml` or use the generator with offset:
```bash
python generate_compose.py --port-offset 100
```

### Container Can't Access Host Network

**Symptom:** Container can't reach services on the host machine.

**Solution:**
- Use `host.docker.internal` instead of `localhost`
- Or use `--network host` flag

---

## Testing Issues

### Tests Fail on Import

**Symptom:** `ModuleNotFoundError` when running pytest.

**Solution:** Install the package in development mode:
```bash
pip install -e .
# Or ensure the project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests Fail Due to Missing Dependencies

**Symptom:** Tests fail because optional dependencies aren't installed.

**Solution:** Install dev dependencies:
```bash
pip install -r requirements.txt
# Or if separated:
pip install -r requirements-dev.txt
```

### Flaky Tests (Non-Deterministic Failures)

**Symptom:** Some tests pass sometimes and fail other times.

**Solution:** Use explicit seeds for reproducibility:
```python
env = NetworkTroubleshootingEnv(seed=42)
```

---

## Scenario Import/Export

### Scenario Import Fails

**Symptom:** `ValueError` or schema validation error when importing a scenario.

**Solution:**
- Ensure the scenario file was exported from a compatible version
- Check the file isn't corrupted
- Verify JSON syntax is valid

### Deterministic Hints Not Working

**Symptom:** Imported scenario produces different hints than original.

**Solution:** The scenario snapshot includes RNG state for reproducibility. Ensure you're using the same hint provider mode:
```python
from netheal.scenario import import_scenario
env = import_scenario("path/to/scenario.json", deterministic_hints=True)
```

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Maximum devices exceeded` | Network has more devices than `max_devices` | Increase `max_devices` parameter |
| `No connections available` | Trying to inject link fault on disconnected network | Ensure network has connections |
| `Device not found` | Invalid device ID in tool call | Use IDs from topology discovery |
| `Environment not initialized` | Calling API before reset | Call `/api/env/reset` first |
| `Invalid action` | Using stale or invalid action ID | Get fresh valid actions before step |

---

## Getting More Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/cisco-ai-platform/netheal-ai-agent-benchmark/issues)
2. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. Run tests to verify your installation:
   ```bash
   pytest tests/ -v
   ```
