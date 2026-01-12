# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Fault injection system for network troubleshooting simulation."""

from .injector import FaultInjector, FaultType

__all__ = ["FaultInjector", "FaultType"]
