# FAQ

- The observation is a dict, not a numpy array. Use dict-aware policies (e.g., SB3 `MultiInputPolicy`).
- `info['user_hint']` is optional and non‑leaky; disable with `enable_user_hints=False`.
- Actions are dynamic per episode; always call `env.get_valid_actions()`.
- Positive final reward generally indicates correct diagnosis; step rewards are small negatives.
