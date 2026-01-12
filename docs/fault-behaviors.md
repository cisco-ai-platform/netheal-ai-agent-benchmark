# Fault Type Behaviors in CNTE

This document describes how each fault type manifests during a CNTE episode, including observable symptoms and diagnostic strategies.

---

## 1. DEVICE_FAILURE

**What happens:** A device's status is set to `'down'`

**Location format:** Single device ID (e.g., `"device_2"`)

**Observable symptoms:**
- **Ping to/from device** → Fails with error `"Destination device X is unreachable"` or `"Source device X is down"`
- **Traceroute** → Fails if device is on the path; shows partial path up to the failure point
- **Check status on device** → Returns `status: 'down'`
- **Check interfaces** → Fails with `"Device X is down"`
- **All paths through device** → Broken (no route to hosts beyond it)

**Diagnostic clues:**
- Multiple pings fail to/from the same device
- Traceroutes show the same device as the failure point
- Status check directly reveals the device is down

---

## 2. LINK_FAILURE

**What happens:** A connection's status is set to `'down'` (bidirectional)

**Location format:** Connection string (e.g., `"device_1->device_2"`)

**Observable symptoms:**
- **Ping across the link** → Fails with `"No route to host"` if no alternate path exists
- **Traceroute** → Shows partial path, fails at the broken link
- **Check interfaces on either endpoint** → Shows the specific interface as `status: 'down'`
- **Devices themselves remain up** → Status checks pass

**Diagnostic clues:**
- Both devices are up (status checks pass)
- Pings between specific device pairs fail
- Interface check reveals a down interface pointing to the other device
- Other paths not using this link still work

---

## 3. MISCONFIGURATION

**What happens:** A connection from a specific device is blocked (one direction only, simulating a firewall rule or blocked port)

**Location format:** Device ID where misconfiguration exists (e.g., `"device_3"`)

**Observable symptoms:**
- **Ping through the misconfigured device** → May fail in one direction
- **Check interfaces** → The blocked interface shows `status: 'down'`
- **Device status** → Shows as `'up'` (device is operational)
- **Selective connectivity** → Some destinations reachable, others not

**Diagnostic clues:**
- Device is up but specific connections from it fail
- Asymmetric behavior (may work in one direction but not the other)
- Interface check shows one specific interface down while others are up
- Similar to link failure but localized to one device's outbound connection

---

## 4. PERFORMANCE_DEGRADATION

**What happens:** A connection's latency is multiplied by 2-10x (bidirectional)

**Location format:** Connection string (e.g., `"device_0->device_1"`)

**Observable symptoms:**
- **Ping across the link** → **Succeeds** but with abnormally high `latency_ms`
- **Traceroute** → Succeeds but shows high latency on the affected hop
- **Check interfaces** → Shows `status: 'up'` but elevated `latency_ms`
- **All connectivity works** → Just slower

**Diagnostic clues:**
- Pings succeed but latency is 2-10x higher than normal (normal is ~0.5-5ms)
- Traceroute reveals which hop has the latency spike
- No failures, just slowness
- Interface check shows the high latency value

---

## Summary Table

| Fault Type | Location | Device Status | Link Status | Ping Result | Key Diagnostic |
|------------|----------|---------------|-------------|-------------|----------------|
| **DEVICE_FAILURE** | `device_X` | DOWN | N/A | Fails | `check_status` shows down |
| **LINK_FAILURE** | `A->B` | Both UP | DOWN | Fails (no route) | `check_interfaces` shows down interface |
| **MISCONFIGURATION** | `device_X` | UP | One direction DOWN | Selective failures | Interface check shows one down interface |
| **PERFORMANCE_DEGRADATION** | `A->B` | Both UP | UP | **Succeeds** with high latency | Latency 2-10x normal |

---

## Agent Strategy Implications

- **Device failure** → Use `check_status` on suspected devices
- **Link failure** → Use `check_interfaces` to find down interfaces, or `traceroute` to find where path breaks
- **Misconfiguration** → Similar to link failure but look for asymmetric behavior
- **Performance degradation** → The only fault where pings **succeed**; look for high latency values in ping/traceroute results

---

## Code References

- Fault injection: `netheal/faults/injector.py`
- Tool simulation: `netheal/tools/simulator.py`
- Observation updates: `netheal/environment/observation.py`
