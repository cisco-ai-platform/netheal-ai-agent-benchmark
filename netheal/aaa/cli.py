# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI for the NetHeal Green Agent A2A server.

Provides command-line interface for starting the assessment server
with configurable host, port, and agent card URL.

Usage:
    python -m netheal.aaa.cli --host 0.0.0.0 --port 9020 --card-url http://localhost:9020

Docker execution:
    docker run <image> --host 0.0.0.0 --port 9020 --card-url http://green-agent:9020
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import typer
import uvicorn

from netheal.aaa import server as server_module
from netheal.hints.provider import _resolve_llm_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("netheal.aaa.cli")


def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host address to bind the server to.",
    ),
    port: int = typer.Option(
        9020,
        "--port",
        help="Port number to listen on.",
    ),
    card_url: Optional[str] = typer.Option(
        None,
        "--card-url",
        help="URL to advertise in the agent card (/.well-known/agent.json).",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Uvicorn log level (debug, info, warning, error, critical).",
    ),
) -> None:
    """
    Start the NetHeal green agent A2A server.

    Exposes AAA-compliant endpoints:
      - GET  /.well-known/agent.json  (agent card)
      - POST /tasks                    (create assessment)
      - GET  /tasks/{id}               (get task status)
      - GET  /tasks/{id}/stream        (SSE updates)
    """
    if card_url is None:
        card_url = f"http://{host}:{port}"
        LOGGER.info("No --card-url provided, defaulting to %s", card_url)

    server_module.set_card_url(card_url)

    LOGGER.info("Starting NetHeal green agent A2A server")
    LOGGER.info("  Host: %s", host)
    LOGGER.info("  Port: %d", port)
    LOGGER.info("  Card URL: %s", card_url)

    hint_provider = _resolve_llm_provider("auto")
    if hint_provider:
        LOGGER.info("  LLM hints: configured (%s)", hint_provider)
    else:
        LOGGER.info("  LLM hints: not configured (heuristic hints)")

    uvicorn.run(
        server_module.app,
        host=host,
        port=port,
        log_level=log_level,
    )


def main() -> None:
    """CLI entry point."""
    typer.run(serve)


if __name__ == "__main__":
    main()
