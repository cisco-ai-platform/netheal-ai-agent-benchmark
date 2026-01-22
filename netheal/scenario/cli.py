# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""CLI for NetHeal scenario snapshots."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from netheal.environment.env import NetworkTroubleshootingEnv
from netheal.faults.injector import FaultType

from .snapshot import export_snapshot, load_snapshot_episodes


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_env_config(args: argparse.Namespace) -> dict:
    config = {
        "min_devices": args.min_devices,
        "max_devices": args.max_devices,
        "max_episode_steps": args.max_episode_steps,
        "topology_types": _parse_list(args.topology_types),
        "enable_user_hints": args.enable_user_hints,
        "hint_provider_mode": args.hint_provider_mode,
        "reward_scaling_factor": args.reward_scaling_factor,
    }

    if args.fault_types:
        config["fault_types"] = [FaultType(ft) for ft in _parse_list(args.fault_types)]

    return config


def _generate_single(
    idx: int,
    seed: Optional[int],
    config: dict,
    output_dir: Path,
) -> Tuple[str, Path]:
    env = NetworkTroubleshootingEnv(**config)
    env.reset(seed=seed)
    snapshot = export_snapshot(env, seed=seed, metadata={"episode_index": idx})
    path = output_dir / f"{snapshot['snapshot_id']}.json"
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    env.close()
    return snapshot["snapshot_id"], path


def _handle_generate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _build_env_config(args)

    if args.clean:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()

    concurrency = max(1, args.concurrency)
    seeds = [
        (args.seed + idx) if args.seed is not None else None for idx in range(args.count)
    ]

    snapshot_ids: List[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_generate_single, idx, seeds[idx], config, output_dir)
            for idx in range(args.count)
        ]
        for future in as_completed(futures):
            snapshot_id, _ = future.result()
            snapshot_ids.append(snapshot_id)

    snapshot_ids.sort()
    if args.write_index:
        manifest = {
            "version": "v1",
            "snapshot_count": len(snapshot_ids),
            "snapshots": snapshot_ids,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for snapshot_id in snapshot_ids:
                snapshot_path = output_dir / f"{snapshot_id}.json"
                handle.write(snapshot_path.read_text(encoding="utf-8").strip() + "\n")

    print(f"Generated {args.count} snapshots in {output_dir}")
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    target = Path(args.path)
    snapshots = load_snapshot_episodes(target)
    print(f"Validated {len(snapshots)} snapshot(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NetHeal snapshot utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate episode snapshots")
    generate.add_argument("--count", type=int, default=10, help="Number of snapshots to generate")
    generate.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility")
    generate.add_argument("--output", type=str, required=True, help="Output directory for snapshots")
    generate.add_argument("--clean", action="store_true", help="Delete existing snapshots in output directory")
    generate.add_argument("--concurrency", type=int, default=1, help="Concurrent snapshot workers")
    generate.add_argument("--min-devices", type=int, default=3, help="Minimum devices per topology")
    generate.add_argument("--max-devices", type=int, default=15, help="Maximum devices per topology")
    generate.add_argument("--max-episode-steps", type=int, default=100, help="Max steps per episode")
    generate.add_argument("--topology-types", type=str, default=None, help="Comma-separated topology types")
    generate.add_argument("--fault-types", type=str, default=None, help="Comma-separated fault types")
    generate.add_argument(
        "--disable-user-hints",
        action="store_false",
        dest="enable_user_hints",
        default=True,
        help="Disable user hints",
    )
    generate.add_argument("--hint-provider-mode", type=str, default="auto", help="Hint provider mode")
    generate.add_argument("--reward-scaling-factor", type=float, default=10.0, help="Reward scaling factor")
    generate.add_argument("--write-index", action="store_true", help="Write manifest.json and episodes.jsonl")
    generate.set_defaults(func=_handle_generate)

    validate = subparsers.add_parser("validate", help="Validate snapshot file(s)")
    validate.add_argument("path", type=str, help="Snapshot JSON/JSONL file or directory")
    validate.set_defaults(func=_handle_validate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
