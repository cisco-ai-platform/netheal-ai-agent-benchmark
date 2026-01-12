# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import re
from netheal import NetworkTroubleshootingEnv
from netheal.hints import HeuristicHintProvider


def _assert_non_leaky(text: str):
    assert text is not None and isinstance(text, str) and len(text) > 0
    # Should not contain explicit fault type names
    forbidden = [
        "device_failure",
        "link_failure",
        "misconfiguration",
        "performance_degradation",
    ]
    lower = text.lower()
    for term in forbidden:
        assert term not in lower
    # Should not contain raw device ids or connection arrows
    assert re.search(r"device_\d+", text, flags=re.IGNORECASE) is None
    assert "->" not in text


def test_user_hint_present_and_sanitized_with_heuristic():
    env = NetworkTroubleshootingEnv(
        max_devices=5,
        max_episode_steps=5,
        enable_user_hints=True,
        hint_provider=HeuristicHintProvider(),  # force deterministic
        user_context={"access_point": "Lab-AP"},
    )
    obs, info = env.reset(seed=123)
    hint = info.get("user_hint")
    _assert_non_leaky(hint)
    # Should include provided access point context
    assert "Lab-AP" in hint


def test_user_hint_absent_when_disabled():
    env = NetworkTroubleshootingEnv(
        max_devices=4,
        max_episode_steps=4,
        enable_user_hints=False,
        hint_provider=HeuristicHintProvider(),
    )
    obs, info = env.reset(seed=321)
    assert "user_hint" in info
    assert info["user_hint"] is None


def test_user_hint_persists_across_steps():
    env = NetworkTroubleshootingEnv(
        max_devices=4,
        max_episode_steps=4,
        enable_user_hints=True,
        hint_provider=HeuristicHintProvider(),
    )
    obs, info = env.reset(seed=42)
    hint0 = info.get("user_hint")
    _assert_non_leaky(hint0)

    # Take a valid or random action; hint should still be present
    action = env.get_valid_actions()[0] if env.get_valid_actions() else env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    hint1 = info.get("user_hint")
    assert hint1 == hint0
