#!/usr/bin/env python3
"""
Generate docker-compose.yml from AgentBeats scenario.toml.

Converts a scenario definition into a docker-compose configuration
for local testing of AgentBeats assessments.

Usage:
    pip install tomli tomli-w requests
    python generate_compose.py --scenario scenarios/netheal/scenario.toml
    cp .env.example .env  # Edit with your secrets
    mkdir -p output
    docker compose up --abort-on-container-exit

Reference: https://docs.agentbeats.dev/tutorial/#running-the-scenario--submitting-the-results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomli
except ImportError:
    print("Error: tomli not installed. Run: pip install tomli")
    sys.exit(1)

try:
    import tomli_w
except ImportError:
    tomli_w = None  # Optional for writing TOML


def load_scenario(scenario_path: str) -> Dict[str, Any]:
    """Load and parse scenario.toml file."""
    with open(scenario_path, "rb") as f:
        return tomli.load(f)


def resolve_agent_image(agent_config: Dict[str, Any], agent_type: str) -> str:
    """
    Resolve agent image from config.
    
    Priority:
    1. `image` field (for local testing)
    2. Fetch from AgentBeats API using `agentbeats_id` (requires network)
    """
    if "image" in agent_config:
        return agent_config["image"]
    
    agentbeats_id = agent_config.get("agentbeats_id")
    if agentbeats_id:
        # For CI/CD, the image should be fetched from AgentBeats registry
        # For local testing, fall back to default images
        print(f"Warning: agentbeats_id '{agentbeats_id}' specified for {agent_type}.")
        print(f"  Using default local image. Set 'image' in scenario.toml for local testing.")
        
        if agent_type == "green":
            return "netheal-green-agent:latest"
        else:
            return "netheal-purple-agent:latest"
    
    raise ValueError(f"No image or agentbeats_id specified for {agent_type} agent")


def build_env_section(env_config: Dict[str, str]) -> Dict[str, str]:
    """
    Build environment section for docker-compose.
    
    Converts ${VAR} references to docker-compose compatible format.
    """
    result = {}
    for key, value in env_config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # Keep variable reference as-is for docker-compose
            var_name = value[2:-1]
            result[key] = f"${{{var_name}:-}}"
        else:
            result[key] = value
    return result


def generate_compose(scenario: Dict[str, Any], output_path: str = "docker-compose.generated.yml") -> None:
    """Generate docker-compose.yml from scenario definition."""
    
    green_agent = scenario.get("green_agent", {})
    participants = scenario.get("participants", [])
    config = scenario.get("config", {})
    
    # Resolve images
    green_image = resolve_agent_image(green_agent, "green")
    
    # Build services
    services: Dict[str, Any] = {}
    
    # Green agent service
    green_env = build_env_section(green_agent.get("env", {}))
    green_env["PYTHONUNBUFFERED"] = "1"
    
    services["green-agent"] = {
        "image": green_image,
        "container_name": "netheal-green-agent",
        "ports": ["9020:9020"],
        "environment": green_env,
        "command": [
            "--host", "0.0.0.0",
            "--port", "9020",
            "--card-url", "http://green-agent:9020",
        ],
        "healthcheck": {
            "test": ["CMD", "python", "-c", 
                     "import urllib.request; urllib.request.urlopen('http://localhost:9020/.well-known/agent.json')"],
            "interval": "10s",
            "timeout": "5s",
            "retries": 3,
            "start_period": "10s",
        },
        "volumes": ["./output:/app/output"],
        "networks": ["netheal-network"],
    }
    
    # Add config as environment variables
    if config:
        for key, value in config.items():
            env_key = f"NETHEAL_{key.upper()}"
            if isinstance(value, list):
                services["green-agent"]["environment"][env_key] = ",".join(str(v) for v in value)
            else:
                services["green-agent"]["environment"][env_key] = str(value)
    
    # Purple agent services (participants)
    port_offset = 0
    participant_endpoints = []
    
    for i, participant in enumerate(participants):
        service_name = participant.get("name", f"purple-agent-{i}")
        service_name = service_name.replace("_", "-").lower()
        
        purple_image = resolve_agent_image(participant, f"participant[{i}]")
        purple_port = 9030 + port_offset
        port_offset += 1
        
        purple_env = build_env_section(participant.get("env", {}))
        purple_env["PYTHONUNBUFFERED"] = "1"
        
        solver_type = participant.get("solver", "dummy")
        
        services[service_name] = {
            "image": purple_image,
            "container_name": f"netheal-{service_name}",
            "ports": [f"{purple_port}:{purple_port}"],
            "environment": purple_env,
            "command": [
                "--host", "0.0.0.0",
                "--port", str(purple_port),
                "--card-url", f"http://{service_name}:{purple_port}",
                "--solver", solver_type,
            ],
            "healthcheck": {
                "test": ["CMD", "python", "-c",
                         f"import urllib.request; urllib.request.urlopen('http://localhost:{purple_port}/.well-known/agent.json')"],
                "interval": "10s",
                "timeout": "5s",
                "retries": 3,
                "start_period": "10s",
            },
            "depends_on": {
                "green-agent": {"condition": "service_healthy"},
            },
            "networks": ["netheal-network"],
        }
        
        participant_endpoints.append({
            "name": participant.get("name", f"solver_{i}"),
            "endpoint": f"http://{service_name}:{purple_port}",
        })
    
    # Build final compose structure
    compose = {
        "services": services,
        "networks": {
            "netheal-network": {
                "name": "netheal-network",
            },
        },
    }
    
    # Write docker-compose.yml
    import yaml
    
    # Custom representer for cleaner YAML output
    def str_representer(dumper, data):
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    
    yaml.add_representer(str, str_representer)
    
    with open(output_path, "w") as f:
        f.write("# Auto-generated from scenario.toml\n")
        f.write("# Do not edit directly - regenerate with: python generate_compose.py\n")
        f.write("#\n")
        f.write("# Participant endpoints for assessment request:\n")
        for ep in participant_endpoints:
            f.write(f"#   {ep['name']}: {ep['endpoint']}\n")
        f.write("\n")
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated: {output_path}")
    print(f"\nParticipant endpoints:")
    for ep in participant_endpoints:
        print(f"  {ep['name']}: {ep['endpoint']}")
    print(f"\nUsage:")
    print(f"  cp .env.example .env  # Add your secrets")
    print(f"  mkdir -p output")
    print(f"  docker compose -f {output_path} up --abort-on-container-exit")


def main():
    parser = argparse.ArgumentParser(
        description="Generate docker-compose.yml from AgentBeats scenario.toml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_compose.py --scenario scenarios/netheal/scenario.toml
  python generate_compose.py --scenario scenario.toml --output docker-compose.test.yml
        """,
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to scenario.toml file",
    )
    parser.add_argument(
        "--output",
        default="docker-compose.generated.yml",
        help="Output docker-compose file (default: docker-compose.generated.yml)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.scenario):
        print(f"Error: Scenario file not found: {args.scenario}")
        sys.exit(1)
    
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)
    
    scenario = load_scenario(args.scenario)
    generate_compose(scenario, args.output)


if __name__ == "__main__":
    main()
