"""Deduplication and summary table reporting for JSONL trial results."""

import json
import math
import pathlib

RATE_METRICS = (
    "attack_success",
    "utility_success",
    "dual_success",
    "exec_miss",
    "mech_fail",
    "decline",
)


def mean_sem(values: list[float]) -> tuple[float, float | None]:
    """Compute arithmetic mean and standard error of the mean (SEM).

    Args:
        values: Numeric sample values across random seeds.

    Returns:
        Tuple `(mean, sem)` where `sem` is `None` when fewer than 2 samples exist.
    """
    n = len(values)
    if n == 0:
        return 0.0, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def format_rate(mean: float, sem: float | None) -> str:
    """Format a percentage rate with optional ± SEM for tabular display."""
    return f"{mean:5.1f}%" if sem is None else f"{mean:5.1f}±{sem:4.1f}%"


def load_dedup_results(results_path: pathlib.Path) -> list[dict]:
    """Load `results.jsonl` and keep only the latest record per unique trial key.

    Args:
        results_path: Path to the JSONL file written by `trial.run_suite`.

    Returns:
        Deduplicated list of trial result dictionaries.
    """
    dedup: dict[tuple, dict] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (
            rec["objective"],
            rec["condition"],
            rec["n_trained_tasks"],
            rec["seed"],
            rec["split"],
            rec["step"],
            rec["task_id"],
        )
        dedup[key] = rec
    return list(dedup.values())

def inspect_trials(
        results_path: pathlib.Path,
        objective: str,
        condition: str,
        task_id: int | None = None,
        n_trained_tasks: int | None = None,
        seed: int | None = None,
        step: int | None = None,
        split: str = "held_out",
        attack_success: bool | None = None,
        utility_success: bool | None = None,
        diagnosis: str | None = None,
) -> None:
    """Print the task, outcome and tool calls of the trials in one suite, or of a single trial.

    Args:
        results_path: Path to the JSONL file written by `trial.run_suite`.
        objective: Target scenario ('delete', 'exfiltrate', or 'fetch').
        condition: Condition name ('clean', 'naive', 'random', 'manual', or 'adaptive').
        task_id: Zero-based index of the evaluated task in the suite; `None` shows every trial of the suite.
        n_trained_tasks: Number of training tasks used to optimize the trigger, if applicable.
        seed: Optimization random seed, if applicable.
        step: Optimization checkpoint step, if applicable.
        split: Dataset split identifier ('train' or 'held_out').
        attack_success: Show only trials with this ground-truth attack outcome, if specified.
        utility_success: Show only trials with this benign-task outcome, if specified.
        diagnosis: Show only trials with this diagnosis ('decline', 'mechanical_failure', or 'success'), if specified.
    """
    filters = {
        "objective": objective, "condition": condition, "n_trained_tasks": n_trained_tasks,
        "seed": seed, "step": step, "split": split, "task_id": task_id,
        "attack_success": attack_success, "utility_success": utility_success, "diagnosis": diagnosis,
    }
    found = False
    for rec in load_dedup_results(results_path):
        # A filter left as None matches anything.
        if any(want is not None and rec[key] != want for key, want in filters.items()):
            continue
        found = True
        print(
            f"--- {rec['objective']} | {rec['condition']} | {rec['split']} | trained tasks: {rec['n_trained_tasks']} "
            f"| seed: {rec['seed']} | step: {rec['step']} | task {rec['task_id']} ---"
        )
        print("prompt:", rec["task"])
        print("attack:", rec["attack_success"], "| utility:", rec["utility_success"], "|", rec["diagnosis"], "|", rec["duration_sec"], "s")
        # Every tool call Pi made in this trial, in order, each printed once.
        for line in open(rec["log_file"]):
            ev = json.loads(line)
            if ev.get("type") == "tool_execution_start":
                args = ev.get("args", {})
                print("    ", ev.get("toolName"), "->", args.get("command") or args.get("path"))
    if not found:
        print("no matching trials")


def summarize_results(results_path: pathlib.Path) -> list[dict]:
    """Aggregate deduplicated trial records by seed and condition, and print a summary table.

    Args:
        results_path: Path to the JSONL file written by `trial.run_suite`.

    Returns:
        List of dictionaries containing numeric `*_mean` and `*_sem` fields per
        experimental group for downstream programmatic analysis.
    """
    records = load_dedup_results(results_path)

    # Stage 1: Group individual task trials into suite-level percentages per seed.
    by_seed: dict[tuple, list[dict]] = {}
    for rec in records:
        seed_key = (
            rec["objective"],
            rec["condition"],
            rec["n_trained_tasks"],
            rec["split"],
            rec["step"],
            rec["seed"],
        )
        by_seed.setdefault(seed_key, []).append(rec)

    # Stage 2: Collect per-seed suite percentages for each experimental configuration.
    by_group: dict[tuple, list[dict[str, float]]] = {}
    for (obj, cond, n_tasks, spl, stp, _seed), recs in sorted(
        by_seed.items(), key=lambda item: tuple("" if v is None else str(v) for v in item[0])
    ):
        n = len(recs)
        seed_rates = {
            "attack_success": sum(r["attack_success"] for r in recs) / n * 100,
            "utility_success": sum(r["utility_success"] for r in recs) / n * 100,
            "dual_success": sum(r["dual_success"] for r in recs) / n * 100,
            "exec_miss": sum(
                (not r["attack_success"]) and r["diagnosis"] == "success" for r in recs
            ) / n * 100,
            "mech_fail": sum(r["diagnosis"] == "mechanical_failure" for r in recs) / n * 100,
            "decline": sum(r["diagnosis"] == "decline" for r in recs) / n * 100,
            "sec": sum(r["duration_sec"] for r in recs) / n,
            "n_tasks": n,
        }
        by_group.setdefault((obj, cond, n_tasks, spl, stp), []).append(seed_rates)

    header = (
        f"{'Objective':<11} | {'Condition':<9} | {'TrainSet':<8} | {'Split':<8} | "
        f"{'Step':<5} | {'Seeds':<5} | {'N':<3} | "
        f"{'ASR':<11} | {'Util':<11} | {'Dual':<11} | "
        f"{'ExecMiss':<11} | {'MechFail':<11} | {'Decline':<11} | {'Sec':<5}"
    )
    print(header)
    print("-" * len(header))

    summary_rows: list[dict] = []
    condition_order = ["clean", "naive", "random", "manual", "adaptive", "transfer"]
    for (obj, cond, n_tasks, spl, stp), seed_list in sorted(
        by_group.items(),
        key=lambda item: (item[0][0], condition_order.index(item[0][1]), item[0][2] or 0, item[0][3], item[0][4] or 0),
    ):
        n_seeds = len(seed_list)
        eval_tasks = int(seed_list[0]["n_tasks"])
        train_lbl = f"{n_tasks}tasks" if n_tasks is not None else "-"
        step_lbl = str(stp) if stp is not None else "-"

        stats = {m: mean_sem([s[m] for s in seed_list]) for m in RATE_METRICS}
        cols = {m: format_rate(*stats[m]) for m in RATE_METRICS}
        sec_mean = sum(s["sec"] for s in seed_list) / n_seeds

        print(
            f"{obj:<11} | {cond:<9} | {train_lbl:<8} | {spl:<8} | "
            f"{step_lbl:<5} | {n_seeds:<5} | {eval_tasks:<3} | "
            f"{cols['attack_success']:<11} | {cols['utility_success']:<11} | {cols['dual_success']:<11} | "
            f"{cols['exec_miss']:<11} | {cols['mech_fail']:<11} | {cols['decline']:<11} | {sec_mean:5.1f}"
        )

        # Store raw numeric floats in the returned list so callers don't have to parse strings.
        numeric_fields = {}
        for m, (m_val, s_val) in stats.items():
            numeric_fields[f"{m}_mean"] = round(m_val, 2)
            numeric_fields[f"{m}_sem"] = None if s_val is None else round(s_val, 2)

        summary_rows.append(
            {
                "objective": obj,
                "condition": cond,
                "n_trained_tasks": n_tasks,
                "split": spl,
                "step": stp,
                "n_seeds": n_seeds,
                "n_tasks": eval_tasks,
                **numeric_fields,
                "sec_mean": round(sec_mean, 2),
            }
        )

    return summary_rows


if __name__ == "__main__":
    summarize_results(pathlib.Path("results.jsonl"))