"""
Metrics computation — accuracy, latency, per-scenario breakdown.
"""

from __future__ import annotations

from typing import Any


def compute_scenario_metrics(scenario_results: list[dict]) -> dict:
    """
    Compute per-scenario and overall metrics.

    Each item in scenario_results should have:
        name, scores (dict[str, bool]), latency_s, tokens_generated, time_to_first_token_s
    """
    per_scenario: list[dict] = []
    total_pass = 0
    total_checks = 0
    latencies: list[float] = []
    ttfts: list[float] = []
    tokens_per_sec: list[float] = []

    for res in scenario_results:
        scores = res.get("scores", {})
        passed = sum(1 for v in scores.values() if v)
        total = len(scores)
        total_pass += passed
        total_checks += total

        latency = res.get("latency_s", 0)
        tokens = res.get("tokens_generated", 0)
        ttft = res.get("time_to_first_token_s", 0)

        latencies.append(latency)
        if ttft > 0:
            ttfts.append(ttft)
        if latency > 0 and tokens > 0:
            tokens_per_sec.append(tokens / latency)

        scenario = {
            "name": res.get("name", "?"),
            "passed": passed,
            "total": total,
            "accuracy_pct": round(passed / total * 100, 1) if total else 0,
            "latency_s": round(latency, 2),
            "tokens_generated": tokens,
        }
        if ttft > 0:
            scenario["time_to_first_token_s"] = round(ttft, 3)
        if latency > 0 and tokens > 0:
            scenario["tokens_per_sec"] = round(tokens / latency, 1)

        scenario["checks"] = {k: "PASS" if v else "FAIL" for k, v in scores.items()}
        per_scenario.append(scenario)

    overall_accuracy = (total_pass / total_checks * 100) if total_checks else 0

    summary = {
        "overall_accuracy_pct": round(overall_accuracy, 1),
        "total_pass": total_pass,
        "total_checks": total_checks,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_s": round(
            sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2
        ),
        "avg_tokens_per_sec": (
            round(sum(tokens_per_sec) / len(tokens_per_sec), 1)
            if tokens_per_sec else 0
        ),
        "avg_ttft_s": (
            round(sum(ttfts) / len(ttfts), 3)
            if ttfts else 0
        ),
        "scenarios": per_scenario,
    }
    return summary


def format_comparison_table(
    current: dict,
    baselines: dict[str, dict] | None = None,
) -> str:
    """
    Build a human-readable comparison table.

    baselines: {"V4.7": {scenario_name: accuracy_pct, ...}, ...}
    """
    lines = []
    scenarios = current.get("scenarios", [])

    header = f"  {'Scenario':<25s}  {'Score':<8s}  {'Acc%':<6s}  {'Latency':<8s}"
    if baselines:
        for bname in baselines:
            header += f"  {bname:<6s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for sc in scenarios:
        row = (
            f"  {sc['name']:<25s}  "
            f"{sc['passed']}/{sc['total']:<6s}  "
            f"{sc['accuracy_pct']:>5.1f}%  "
            f"{sc['latency_s']:>6.2f}s"
        )
        if baselines:
            for bname, bdata in baselines.items():
                bval = bdata.get(sc["name"], "-")
                if isinstance(bval, (int, float)):
                    row += f"  {bval:>5.0f}%"
                else:
                    row += f"  {str(bval):>6s}"
        lines.append(row)

    lines.append("")
    lines.append(f"  Overall: {current['total_pass']}/{current['total_checks']} "
                 f"= {current['overall_accuracy_pct']:.1f}%")
    lines.append(f"  Avg latency: {current['avg_latency_s']:.2f}s  |  "
                 f"Avg tokens/sec: {current['avg_tokens_per_sec']:.1f}  |  "
                 f"Avg TTFT: {current['avg_ttft_s']:.3f}s")
    return "\n".join(lines)
