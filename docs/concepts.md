# Core Concepts

- Network graph: Directed NetworkX graph with devices (router, switch, server, firewall, host) and links.
- Topologies: linear, star, mesh, hierarchical, random.
- Faults: device failure, link failure, performance degradation, misconfiguration.
- Tools: ping, traceroute, check_status, check_interfaces (simulated realistically).
- Actions (hierarchical): discovery → diagnostics → final diagnosis.
- Observations (dict):
  - discovery_matrix (known adjacency)
  - device_status (features per device)
  - recent_diagnostics (tool result memory)
  - episode_metadata (progress)
- Rewards: small per-step penalty; final positive/negative outcome on diagnosis, scaled by network size.
