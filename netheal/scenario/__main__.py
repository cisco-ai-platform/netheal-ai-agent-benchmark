# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Entry point for python -m netheal.scenario."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
