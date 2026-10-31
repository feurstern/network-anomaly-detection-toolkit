"""Suggested investigation steps, keyed by metric family.

Steps are starting points for a human investigation and are phrased as checks, not conclusions:
an alert shows that metrics deviated together, not why. No step asserts a cause, and none labels
the behaviour as an attack - that needs independent evidence.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from network_anomaly.schemas import MetricFamily, metric_spec

PLAYBOOK: Mapping[MetricFamily, tuple[str, ...]] = MappingProxyType(
    {
        MetricFamily.TRAFFIC: (
            "Compare the request mix (endpoints, clients, user agents, source networks) with the "
            "same time last week.",
            "Check for deployments, campaigns, scheduled jobs, crawlers or client retries that "
            "change request volume.",
            "Check load balancer and DNS changes that could move traffic to or away from this "
            "host.",
        ),
        MetricFamily.ERRORS: (
            "Group application and web-server logs for the interval by endpoint and status code.",
            "Check recent releases and configuration changes, and the health of dependencies "
            "(databases, caches, upstream APIs).",
            "For 4xx increases, check for clients requesting unknown paths or using expired "
            "credentials.",
        ),
        MetricFamily.LATENCY: (
            "Find the slowest endpoints and queries in the interval (traces, slow-query logs).",
            "Check the host for saturation: CPU, memory, disk and connection counts at the same "
            "time.",
            "Check response times of dependencies (databases, caches, external APIs).",
        ),
        MetricFamily.CPU: (
            "Identify which processes used the CPU (e.g. top, pidstat) and compare with the usual "
            "workload.",
            "Check for runaway jobs, busy loops or garbage-collection pressure after recent "
            "changes.",
        ),
        MetricFamily.MEMORY: (
            "Check per-process resident memory and whether it grows steadily.",
            "Look for out-of-memory kills, swapping and unplanned restarts in system logs.",
        ),
        MetricFamily.DISK: (
            "Identify the processes doing I/O (e.g. iotop) and any backups or batch jobs scheduled "
            "in the interval.",
            "Check disk latency, queue depth and free space.",
        ),
        MetricFamily.NETWORK: (
            "Compare interface throughput with request volume: larger responses or bulk transfers "
            "raise bytes without more requests.",
            "Check for backups, replication or large downloads in the interval, and review flow "
            "records for unusual destinations or ports.",
        ),
        MetricFamily.CONNECTIONS: (
            "Check connection states (e.g. ss -s) for growth in established, TIME_WAIT or SYN_RECV "
            "connections.",
            "Review connection-pool, keep-alive and timeout settings, and whether clients retry "
            "failed requests.",
        ),
        MetricFamily.FLOWS: (
            "Group the window's flows by source, destination and destination port, and compare "
            "with the same time last week.",
            "Check whether connections complete (SYN without ACK, resets) and whether a few "
            "sources reach many destination ports or hosts.",
        ),
        MetricFamily.GENERIC: (
            "Compare the metric with related metrics on this host and with other hosts at the same "
            "time.",
        ),
    }
)

CONFIRMATION_STEP = (
    "Confirm with independent evidence (logs, traces, flow records, change history) before "
    "treating this as an incident."
)


def suggested_checks(metrics: Sequence[str], limit: int = 5) -> tuple[str, ...]:
    """Checks for the families of ``metrics`` (most relevant first), ending with confirmation.

    The first family contributes up to two steps and later families one each, so the list stays
    short and covers every family involved.
    """
    families: list[MetricFamily] = []
    for metric in metrics:
        family = metric_spec(metric).family
        if family not in families:
            families.append(family)
    steps: list[str] = []
    for rank, family in enumerate(families):
        for step in PLAYBOOK[family][: 2 if rank == 0 else 1]:
            if step not in steps:
                steps.append(step)
    return (*steps[: max(limit - 1, 0)], CONFIRMATION_STEP)
