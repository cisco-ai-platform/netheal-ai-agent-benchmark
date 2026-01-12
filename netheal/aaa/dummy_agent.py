# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Dummy solver agent for testing.

Connects to a NetHeal MCP server via HTTP endpoints and executes
random valid actions until submitting a diagnosis.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
from typing import Any, Dict, List, Optional

import httpx

LOGGER = logging.getLogger("netheal.dummy_agent")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


async def run_dummy_agent(
    base_url: str,
    min_steps: int = 4,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """Execute a random troubleshooting policy against the MCP server."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        steps = random.randint(min_steps, max_steps)
        LOGGER.info("Starting dummy session (%s steps).", steps)

        actual_steps = 0
        diagnosis_submitted = False

        for step_idx in range(steps):
            actions = await _get_valid_actions(client)
            if not actions:
                LOGGER.warning("No valid actions; aborting.")
                break

            final_step = step_idx == steps - 1
            action = _select_action(actions, prefer_diagnosis=final_step)
            if action is None:
                LOGGER.warning("Unable to select action; aborting.")
                break

            LOGGER.info(
                "Step %d/%d: %s (%s)",
                step_idx + 1,
                steps,
                action.get("description"),
                action.get("category"),
            )

            result = await _execute_action(client, action, force_diagnosis=final_step)
            actual_steps += 1
            LOGGER.debug("Result: %s", result)

            if action.get("category") == "diagnosis":
                LOGGER.info("Diagnosis submitted.")
                diagnosis_submitted = True
                break

        return {"steps": actual_steps, "diagnosis_submitted": diagnosis_submitted}


async def _get_valid_actions(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    response = await client.get("/actions")
    response.raise_for_status()
    payload = response.json()
    return payload.get("valid_actions", [])


def _select_action(
    actions: List[Dict[str, Any]],
    prefer_diagnosis: bool = False,
) -> Optional[Dict[str, Any]]:
    if not actions:
        return None

    diagnosis_actions = [a for a in actions if a.get("category") == "diagnosis"]
    non_diagnosis = [a for a in actions if a.get("category") != "diagnosis"]

    if prefer_diagnosis and diagnosis_actions:
        return random.choice(diagnosis_actions)

    pool = non_diagnosis or actions
    return random.choice(pool)


async def _execute_action(
    client: httpx.AsyncClient,
    action: Dict[str, Any],
    force_diagnosis: bool = False,
) -> Dict[str, Any]:
    category = action.get("category")
    action_type = action.get("action_type")
    parameters = action.get("parameters", {})

    if category == "diagnosis" or force_diagnosis:
        return await _submit_diagnosis(client, action, parameters)

    if category == "topology_discovery":
        if action_type == "scan_network":
            return await _post(client, "/tools/scan_network")
        if action_type == "discover_neighbors":
            return await _post(
                client, "/tools/discover_neighbors", params={"device": parameters["device"]}
            )

    if category == "diagnostic":
        if action_type == "ping":
            return await _post(
                client,
                "/tools/ping",
                params={"source": parameters["source"], "destination": parameters["destination"]},
            )
        if action_type == "traceroute":
            return await _post(
                client,
                "/tools/traceroute",
                params={"source": parameters["source"], "destination": parameters["destination"]},
            )
        if action_type == "check_status":
            return await _post(
                client, "/tools/check_status", params={"device": parameters["device"]}
            )
        if action_type == "check_interfaces":
            return await _post(
                client, "/tools/check_interfaces", params={"device": parameters["device"]}
            )

    LOGGER.warning("Unknown action; submitting fallback diagnosis.")
    return await _submit_random_diagnosis(client, actions=[action])


async def _submit_diagnosis(
    client: httpx.AsyncClient,
    action: Dict[str, Any],
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    fault_type = action.get("action_type")
    location = parameters.get("location") or "device_0"
    LOGGER.info("Submitting diagnosis: %s at %s", fault_type, location)
    return await _post(
        client,
        "/tools/submit_diagnosis",
        params={"fault_type": fault_type, "location": location},
    )


async def _submit_random_diagnosis(
    client: httpx.AsyncClient, actions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    options = [a for a in actions if a.get("category") == "diagnosis"] or actions
    choice = random.choice(options)
    return await _submit_diagnosis(client, choice, choice.get("parameters", {}))


async def _post(
    client: httpx.AsyncClient, path: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    response = await client.post(path, params=params or {})
    response.raise_for_status()
    return response.json()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dummy NetHeal solver agent.")
    parser.add_argument(
        "--mcp-url",
        required=True,
        help="Base URL of the MCP server (e.g., http://127.0.0.1:9025)",
    )
    parser.add_argument("--min-steps", type=int, default=4, help="Minimum steps before diagnosis.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps before diagnosis.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(run_dummy_agent(args.mcp_url, args.min_steps, args.max_steps))


if __name__ == "__main__":
    main()
