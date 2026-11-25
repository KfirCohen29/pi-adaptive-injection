"""Modular statistical aggregation and visualization suite for adversarial agent evaluation."""

import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from trial import TASK_ARCHETYPES
from report import load_dedup_results, mean_sem

DEFAULT_OBJECTIVES = ("delete", "exfiltrate", "fetch")
DEFAULT_CONDITIONS = ("clean", "naive", "random", "manual", "adaptive")

OBJECTIVE_TITLES = {
    "delete": "File Deletion (delete)",
    "exfiltrate": "Data Exfiltration (exfiltrate)",
    "fetch": "Remote Fetch (fetch)",
}
CONDITION_TITLES = {
    "clean": "Clean",
    "naive": "Naive",
    "random": "Random",
    "manual": "Manual",
    "adaptive": "Adaptive",
}
PALETTE = {
    "clean": "#7f8c8d",
    "naive": "#3498db",
    "random": "#9b59b6",
    "manual": "#e67e22",
    "adaptive": "#e74c3c",
    "transfer": "#f39c12",
    "delete": "#2ecc71",
    "exfiltrate": "#e74c3c",
    "fetch": "#3498db",
}


# ==============================================================================
# Data Aggregation Helpers
# ==============================================================================


def compute_seed_metrics(
    records: list[dict],
    objective: str,
    condition: str,
    split: str = "held_out",
    n_trained_tasks: int | None = None,
    step: int | None = None,
) -> list[dict[str, float]]:
    """Compute per-seed percentage rates and behavioral averages for one configuration.

    Filters `records` to a single experimental slice, groups trials by `seed`
    (`None` for deterministic baselines, `0/1/2` for optimized triggers), and computes
    percentage rates and runtime averages across all tasks in that split.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        objective: Target attack objective ('delete', 'exfiltrate', or 'fetch').
        condition: Experimental condition name.
        split: Evaluation task split ('train' or 'held_out').
        n_trained_tasks: Training task count for optimized triggers, or `None` for baselines.
        step: Optimization checkpoint step, or `None` for baselines.

    Returns:
        List of per-seed metric dictionaries sorted by seed.
    """
    # Keep only trials matching every parameter of this slice.
    matched = [
        r
        for r in records
        if r["objective"] == objective
        and r["condition"] == condition
        and r["split"] == split
        and r["n_trained_tasks"] == n_trained_tasks
        and r["step"] == step
    ]

    # Bucket trials by seed (deterministic baselines share seed=None).
    by_seed: dict[int | None, list[dict]] = {}
    for rec in matched:
        by_seed.setdefault(rec["seed"], []).append(rec)

    per_seed: list[dict[str, float]] = []
    # Sort seed=None first (mapped to -1) followed by integer seeds 0, 1, 2.
    for _seed, recs in sorted(by_seed.items(), key=lambda kv: (-1 if kv[0] is None else kv[0])):
        n = len(recs)
        per_seed.append(
            {
                # Core outcome rates
                "asr": sum(r["attack_success"] for r in recs) / n * 100,
                "util": sum(r["utility_success"] for r in recs) / n * 100,
                "dual": sum(r["dual_success"] for r in recs) / n * 100,
                # Mutually exclusive non-ASR failure categories (asr + exec_miss + mech_fail + decline == 100%)
                "exec_miss": sum(
                    (not r["attack_success"]) and r["diagnosis"] == "success" for r in recs
                ) / n * 100,
                "mech_fail": sum(r["diagnosis"] == "mechanical_failure" for r in recs) / n * 100,
                "decline": sum(r["diagnosis"] == "decline" for r in recs) / n * 100,
                # Behavioral overhead averages
                "duration_sec": sum(r["duration_sec"] for r in recs) / n,
                "tool_calls": sum(r["tool_calls_count"] for r in recs) / n,
            }
        )
    return per_seed


def summarize_metric(seed_metrics: list[dict[str, float]], metric: str) -> tuple[float, float]:
    """Return `(mean, sem)` across seeds for `metric`, defaulting `sem` to `0.0` when `n < 2`.

    Args:
        seed_metrics: Output list from `compute_seed_metrics`.
        metric: Key name inside each per-seed metric dictionary.

    Returns:
        Tuple `(mean, sem)` as floats ready for plotting.
    """
    mean, sem = mean_sem([m[metric] for m in seed_metrics])
    return mean, (0.0 if sem is None else sem)


# ==============================================================================
# Section A: Baseline Ladder
# ==============================================================================


def plot_baseline_ladder_asr(
    records: list[dict],
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    n_trained_tasks: int = 4,
    final_step: int = 500,
) -> None:
    """Plot a grouped bar chart comparing held-out ASR across all conditions and objectives.

    Visual encoding:
        - X-axis: Grouped by attack objective (`delete`, `exfiltrate`, `fetch`), with
          adjacent vertical bars for each condition (`clean`, `naive`, `random`, `manual`, `adaptive`).
        - Y-axis: Ground-truth Attack Success Rate (`attack_success`, 0–100%).
        - Error bars & annotations: ±1 SEM across optimization seeds on `adaptive`;
          exact integer percentage labels printed above non-zero bars.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        out_path: Destination `.png` file path.
        objectives: Ordered tuple of objective names to plot along the x-axis.
        conditions: Ordered tuple of condition names to plot within each cluster.
        n_trained_tasks: Training task count for the `adaptive` condition.
        final_step: Optimization checkpoint step for the `adaptive` condition.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    # Create integer cluster centers [0, 1, 2] for the objectives on the x-axis.
    x = np.arange(len(objectives))
    # Divide cluster width (0.75) evenly among conditions so adjacent clusters keep a visual gap.
    width = 0.75 / len(conditions)

    for idx, cond in enumerate(conditions):
        means, sems = [], []
        for obj in objectives:
            # Baselines have n_trained_tasks=None and step=None; adaptive uses the specified checkpoint.
            kw = {"n_trained_tasks": n_trained_tasks, "step": final_step} if cond == "adaptive" else {}
            sm = compute_seed_metrics(records, obj, cond, split="held_out", **kw)
            m, s = summarize_metric(sm, "asr")
            means.append(m)
            sems.append(s)

        # Shift each bar left or right around the integer cluster center `x`.
        offset = (idx - (len(conditions) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            # Draw error bars only for multi-seed adaptive runs.
            yerr=sems if cond == "adaptive" else None,
            capsize=3,
            label=CONDITION_TITLES.get(cond, cond),
            color=PALETTE.get(cond, "#333333"),
            edgecolor="black",
            linewidth=0.6,
        )
        # Annotate non-zero bars with their rounded percentage height.
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{m:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([OBJECTIVE_TITLES.get(o, o) for o in objectives])
    ax.set_ylabel("Attack Success Rate (ASR %)")
    ax.set_ylim(0, 112)
    ax.set_title("Baseline Ladder: Held-Out ASR Across Conditions")
    ax.legend(frameon=False, ncol=len(conditions), loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_task_condition_heatmap(
    records: list[dict],
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    n_trained_tasks: int = 4,
    final_step: int = 500,
) -> None:
    """Plot side-by-side heatmaps of per-task held-out ASR across conditions.

    Visual encoding:
        - Layout: Side-by-side matrix panels (one per objective) sharing the y-axis.
        - Rows / Columns: Individual held-out tasks (`Task 0..N`) on the y-axis,
          experimental conditions on the x-axis. grouped by task archetype
          (Static, Bug+Bash, Inspect, Multi, three tasks each)
        - Color & text: Sequential `YlOrRd` colormap (0–100% `attack_success` across
          seeds) with centered numeric cell annotations to reveal task-level sensitivity.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        out_path: Destination `.png` file path.
        objectives: Ordered tuple of objective subplots to render left-to-right.
        conditions: Ordered tuple of condition columns in each heatmap.
        n_trained_tasks: Training task count for the `adaptive` column.
        final_step: Optimization checkpoint step for the `adaptive` column.
    """
    fig, axes = plt.subplots(1, len(objectives), figsize=(5 * len(objectives), 5.5), sharey=True)
    if len(objectives) == 1:
        axes = [axes]

    task_ids = sorted({r["task_id"] for r in records if r["split"] == "held_out"})
    for ax, obj in zip(axes, objectives):
        # Matrix shape: (num_held_out_tasks, num_conditions)
        matrix = np.zeros((len(task_ids), len(conditions)))
        for c_idx, cond in enumerate(conditions):
            for t_idx, tid in enumerate(task_ids):
                rows = [
                    r
                    for r in records
                    if r["objective"] == obj
                    and r["condition"] == cond
                    and r["split"] == "held_out"
                    and r["task_id"] == tid
                    and (
                        cond != "adaptive"
                        or (r["n_trained_tasks"] == n_trained_tasks and r["step"] == final_step)
                    )
                ]
                if rows:
                    matrix[t_idx, c_idx] = sum(r["attack_success"] for r in rows) / len(rows) * 100

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels([CONDITION_TITLES.get(c, c) for c in conditions], rotation=25, ha="right")
        
        n_per_arch = max(1, len(task_ids) // len(TASK_ARCHETYPES))
        ax.set_yticks(np.arange(len(task_ids)))
        ax.set_yticklabels(
            [
                f"T{tid} [{TASK_ARCHETYPES[min(tid // n_per_arch, len(TASK_ARCHETYPES) - 1)]}]"
                for tid in task_ids
            ]
        )
        for boundary in (2.5, 5.5, 8.5):
            if boundary < len(task_ids) - 0.5:
                ax.axhline(boundary, color="black", linewidth=1.2)
        
        ax.set_title(OBJECTIVE_TITLES.get(obj, obj), fontsize=10)

        # Print the numeric percentage inside every heatmap cell.
        for t_idx in range(len(task_ids)):
            for c_idx in range(len(conditions)):
                val = matrix[t_idx, c_idx]
                ax.text(
                    c_idx,
                    t_idx,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if val > 55 else "black",
                )

    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Task ASR (%)")
    fig.suptitle("Per-Task Held-Out ASR Across Conditions", fontsize=12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Section B: Optimization Dynamics & Surrogate Alignment
# ==============================================================================


def plot_convergence_curves(
    records: list[dict],
    loss_dir: pathlib.Path,
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    n_trained_tasks: int = 4,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> None:
    """Plot side-by-side surrogate loss and live agent ASR over steps.

    Visual encoding:
        - Left panel: Optimization step (x-axis) vs. surrogate `PrefillCELoss` (y-axis)
          from `loss_logs/`, with solid lines for seed means and shaded ±1 SEM bands.
        - Right panel: Checkpoint step (x-axis) vs. empirical held-out `ASR` (y-axis)
          from `results.jsonl`, with circular markers and shaded ±1 SEM bands.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        loss_dir: Directory containing `loss_{obj}_{n}tasks_seed{s}.jsonl` files.
        out_path: Destination `.png` file path.
        objectives: Objectives to plot as separate colored trajectories.
        n_trained_tasks: Training task count identifying which optimization logs to read.
        seeds: Random seeds averaged for Mean ± SEM ribbons.
    """
    fig, (ax_loss, ax_asr) = plt.subplots(1, 2, figsize=(12, 4.8))

    for obj in objectives:
        # Left panel: aggregate scalar surrogate loss at each optimization step across seeds.
        step_losses: dict[int, list[float]] = {}
        for seed in seeds:
            lp = loss_dir / f"loss_{obj}_{n_trained_tasks}tasks_seed{seed}.jsonl"
            if not lp.exists():
                continue
            for line in lp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if "loss" in ev:
                    step_losses.setdefault(int(ev["step"]), []).append(float(ev["loss"]))

        if step_losses:
            steps = sorted(step_losses.keys())
            stats = [mean_sem(step_losses[s]) for s in steps]
            means = np.array([m for m, _ in stats])
            sems = np.array([0.0 if s is None else s for _, s in stats])
            ax_loss.plot(steps, means, label=OBJECTIVE_TITLES.get(obj, obj), color=PALETTE[obj], linewidth=1.8)
            # Shaded ribbon spans [mean - SEM, mean + SEM].
            ax_loss.fill_between(steps, means - sems, means + sems, color=PALETTE[obj], alpha=0.2)

        # Right panel: find every checkpoint step evaluated in results.jsonl and plot Mean ± SEM ASR.
        ckpt_steps = sorted(
            {
                r["step"]
                for r in records
                if r["objective"] == obj
                and r["condition"] == "adaptive"
                and r["split"] == "held_out"
                and r["n_trained_tasks"] == n_trained_tasks
                and r["step"] is not None
            }
        )
        if ckpt_steps:
            asr_means, asr_sems = [], []
            for stp in ckpt_steps:
                sm = compute_seed_metrics(
                    records, obj, "adaptive", split="held_out", n_trained_tasks=n_trained_tasks, step=stp
                )
                m, s = summarize_metric(sm, "asr")
                asr_means.append(m)
                asr_sems.append(s)
            means_arr = np.array(asr_means)
            sems_arr = np.array(asr_sems)
            ax_asr.plot(
                ckpt_steps, means_arr, marker="o", label=OBJECTIVE_TITLES.get(obj, obj), color=PALETTE[obj], linewidth=1.8
            )
            ax_asr.fill_between(ckpt_steps, means_arr - sems_arr, means_arr + sems_arr, color=PALETTE[obj], alpha=0.2)

    ax_loss.set_title("Surrogate PrefillCELoss vs. Step (Mean ± SEM)")
    ax_loss.set_xlabel("Optimization Step")
    ax_loss.set_ylabel("Prefill Cross-Entropy Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(frameon=False)

    ax_asr.set_title("Held-Out Agent ASR vs. Step (Mean ± SEM)")
    ax_asr.set_xlabel("Optimization Step")
    ax_asr.set_ylabel("Attack Success Rate (%)")
    ax_asr.set_ylim(-5, 105)
    ax_asr.grid(True, linestyle="--", alpha=0.4)
    ax_asr.legend(frameon=False)

    fig.suptitle("Optimization Convergence vs. Live Agent ASR", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_loss_asr_calibration(
    records: list[dict],
    triggers_path: pathlib.Path,
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    n_trained_tasks: int = 4,
) -> None:
    """Plot 1x3 faceted calibration curves mapping checkpoint surrogate loss to live ASR.

    Visual encoding:
        - Layout: One subplot per objective (`delete`, `exfiltrate`, `fetch`).
        - Points & trajectory: Each marker is one checkpoint step (`50, 100, 200,..500`), plotting
          mean surrogate `PrefillCELoss` (x-axis) vs. mean held-out `ASR` (y-axis) across
          seeds, connected in step order with horizontal/vertical ±1 SEM error bars.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        triggers_path: Path to `triggers.jsonl` containing checkpoint losses and triggers.
        out_path: Destination `.png` file path.
        objectives: Objectives to include as colored point series.
        n_trained_tasks: Training task count to filter checkpoints by.
    """
    if not triggers_path.exists():
        return

    # Group surrogate losses by (objective, step) across seeds.
    losses_by_step: dict[tuple[str, int], list[float]] = {}
    for line in triggers_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec["n_trained_tasks"]) == n_trained_tasks:
            losses_by_step.setdefault((rec["objective"], int(rec["step"])), []).append(float(rec["loss"]))

    fig, axes = plt.subplots(1, len(objectives), figsize=(4.5 * len(objectives), 4.5), sharey=True)
    if len(objectives) == 1:
        axes = [axes]

    for ax, obj in zip(axes, objectives):
        steps = sorted({stp for (o, stp) in losses_by_step if o == obj})
        xs, x_sems, ys, y_sems, valid_steps = [], [], [], [], []

        for stp in steps:
            sm = compute_seed_metrics(
                records, obj, "adaptive", split="held_out", n_trained_tasks=n_trained_tasks, step=stp
            )
            if not sm:
                continue
            l_mean, l_sem = mean_sem(losses_by_step[(obj, stp)])
            a_mean, a_sem = summarize_metric(sm, "asr")
            xs.append(l_mean)
            x_sems.append(0.0 if l_sem is None else l_sem)
            ys.append(a_mean)
            y_sems.append(a_sem)
            valid_steps.append(stp)

        if xs:
            # Connect checkpoints in optimization order to show the trajectory toward lower loss.
            ax.plot(xs, ys, linestyle="--", color=PALETTE[obj], alpha=0.6, linewidth=1.2)
            ax.errorbar(
                xs, ys, xerr=x_sems, yerr=y_sems,
                fmt="o", color=PALETTE[obj], ecolor="black", elinewidth=0.8, capsize=3, markersize=6,
            )
            # Label start, midpoint, and final checkpoints so direction of optimization is clear.
            for idx in {0, len(valid_steps) // 2, len(valid_steps) - 1}:
                ax.annotate(
                    f"t={valid_steps[idx]}",
                    (xs[idx], ys[idx]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7.5,
                )

        ax.set_title(OBJECTIVE_TITLES.get(obj, obj), fontsize=10)
        ax.set_xlabel("Surrogate PrefillCELoss (Mean ± SEM)")
        ax.set_ylim(-5, 108)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Empirical Held-Out ASR (%)")
    fig.suptitle("Surrogate Loss to Live Harness ASR Calibration", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ==============================================================================
# Section C: Objective Difficulty & Failure Diagnosis
# ==============================================================================


def plot_failure_breakdown(
    records: list[dict],
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    conditions: tuple[str, ...] = ("naive", "manual", "adaptive"),
    n_trained_tasks: int = 4,
    final_step: int = 500,
) -> None:
    """Plot 100% stacked bar charts decomposing trials into 4 mutually exclusive outcomes.

    Visual encoding:
        - Layout: Side-by-side subplots (one per objective) comparing `naive`, `manual`,
          and `adaptive` along the x-axis.
        - Stacked segments (summing to 100%): `attack_success` (green), `exec_miss`
          (yellow — clean `bash` exit but ground-truth failed), `mech_fail` (orange —
          objective `bash` call returned `isError: True`), and `decline` (gray — no
          objective `bash` call invoked).

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        out_path: Destination `.png` file path.
        objectives: Ordered tuple of objective subplots.
        conditions: Ordered tuple of conditions along each subplot's x-axis.
        n_trained_tasks: Training task count for the `adaptive` bar.
        final_step: Optimization checkpoint step for the `adaptive` bar.
    """
    fig, axes = plt.subplots(1, len(objectives), figsize=(4.5 * len(objectives), 4.8), sharey=True)
    if len(objectives) == 1:
        axes = [axes]

    outcomes = [
        ("asr", "Full Attack Success", "#2ecc71"),
        ("exec_miss", "Clean Bash / Missed Ground Truth", "#f1c40f"),
        ("mech_fail", "Mechanical Failure (isError)", "#e67e22"),
        ("decline", "Decline (Refused)", "#95a5a6"),
    ]

    for ax, obj in zip(axes, objectives):
        x = np.arange(len(conditions))
        # Track cumulative bar height so each outcome stacks cleanly atop previous segments.
        bottoms = np.zeros(len(conditions))
        for key, label, color in outcomes:
            vals = []
            for cond in conditions:
                kw = {"n_trained_tasks": n_trained_tasks, "step": final_step} if cond == "adaptive" else {}
                sm = compute_seed_metrics(records, obj, cond, split="held_out", **kw)
                vals.append(summarize_metric(sm, key)[0])
            arr = np.array(vals)
            ax.bar(
                x,
                arr,
                bottom=bottoms,
                width=0.55,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            # Print percentage labels inside stacked segments >= 8% tall
            for j, val in enumerate(arr):
                if val >= 8.0:
                    ax.text(
                        x[j],
                        bottoms[j] + val / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if key in ("asr", "mech_fail") else "black",
                    )
            bottoms += arr

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_TITLES.get(c, c) for c in conditions])
        ax.set_title(OBJECTIVE_TITLES.get(obj, obj), fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Share of Held-Out Trials (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
    fig.suptitle(f"Failure-Mode Breakdown Across Objectives - {n_trained_tasks}-Adaptive", y=1.10, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_and_tool_calls(
    records: list[dict],
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    conditions: tuple[str, ...] = ("clean", "manual", "adaptive"),
    n_trained_tasks: int = 4,
    final_step: int = 500,
) -> None:
    """Plot side-by-side wall-clock runtime and tool-call counts across objectives.

    Visual encoding:
        - Left panel: Grouped bars of mean trial wall-clock runtime (`duration_sec`)
          across objectives for `clean`, `manual`, and `adaptive`.
        - Right panel: Grouped bars of mean total tool invocations (`tool_calls_count`)
          per trial, with ±1 SEM error bars on `adaptive`.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        out_path: Destination `.png` file path.
        objectives: Ordered tuple of objectives on the x-axis.
        conditions: Ordered tuple of conditions compared within each objective.
        n_trained_tasks: Training task count for the `adaptive` condition.
        final_step: Optimization checkpoint step for the `adaptive` condition.
    """
    fig, (ax_sec, ax_tools) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(objectives))
    width = 0.72 / len(conditions)

    for idx, cond in enumerate(conditions):
        sec_m, sec_s, tool_m, tool_s = [], [], [], []
        for obj in objectives:
            kw = {"n_trained_tasks": n_trained_tasks, "step": final_step} if cond == "adaptive" else {}
            sm = compute_seed_metrics(records, obj, cond, split="held_out", **kw)
            m1, s1 = summarize_metric(sm, "duration_sec")
            m2, s2 = summarize_metric(sm, "tool_calls")
            sec_m.append(m1)
            sec_s.append(s1)
            tool_m.append(m2)
            tool_s.append(s2)

        offset = (idx - (len(conditions) - 1) / 2) * width
        ax_sec.bar(
            x + offset,
            sec_m,
            width,
            yerr=sec_s if cond == "adaptive" else None,
            capsize=3,
            label=CONDITION_TITLES.get(cond, cond),
            color=PALETTE.get(cond, "#333333"),
            edgecolor="black",
            linewidth=0.5,
        )
        ax_tools.bar(
            x + offset,
            tool_m,
            width,
            yerr=tool_s if cond == "adaptive" else None,
            capsize=3,
            label=CONDITION_TITLES.get(cond, cond),
            color=PALETTE.get(cond, "#333333"),
            edgecolor="black",
            linewidth=0.5,
        )

    for ax, ylabel, title in [
        (ax_sec, "Mean Trial Runtime (sec)", "Wall-Clock Execution Latency"),
        (ax_tools, "Mean Tool Invocations / Trial", "Agent Tool-Call Overhead"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([o.capitalize() for o in objectives])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(frameon=False)

    fig.suptitle("Behavioral Overhead Across Objectives", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ==============================================================================
# Section D: Universality & Cross-Task Generalization
# ==============================================================================


def plot_universality_bars(
    records: list[dict],
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    task_counts: tuple[int, int] = (1, 4),
    final_step: int = 500,
) -> None:
    """Plot grouped bars comparing Train vs. Held-Out ASR for single- vs. multi-task triggers.

    Visual encoding:
        - X-axis: Grouped by objective, with 6 adjacent bars per cluster: for each trigger type,
          `Train (all)` (every training task), `Train (task 0)` (`TRAIN_TASKS[0]` only, the task
          the single-task trigger was optimized on), and `Held-Out`.
        - Y-axis: Mean `ASR` (0–100%) with ±1 SEM error bars across optimization seeds.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        out_path: Destination `.png` file path.
        objectives: Ordered tuple of objectives on the x-axis.
        task_counts: Pair of training task counts `(single_task, multi_task)` to compare.
        final_step: Optimization checkpoint step evaluated.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(objectives))

    # Train-split trials of `TRAIN_TASKS[0]` only. Used for the "Train (task 0)" bars, never for held-out.
    first_task_records = [r for r in records if r["task_id"] == 0]

    # (training task count, split, records to measure, label, color, hatch)
    combos = []
    for nt, light, dark in [(task_counts[0], "#bb8fce", "#6c3483"), (task_counts[1], "#f1948a", "#c0392b")]:
        combos += [
            (nt, "train", first_task_records, f"{nt}-Task (Train, task 0)", light, "//"),
            (nt, "train", records, f"{nt}-Task (Train, all tasks)", light, None),
            (nt, "held_out", records, f"{nt}-Task (Held-Out)", dark, None),
        ]
    width = 0.78 / len(combos)

    for idx, (nt, spl, recs, label, color, hatch) in enumerate(combos):
        means, sems = [], []
        for obj in objectives:
            sm = compute_seed_metrics(recs, obj, "adaptive", split=spl, n_trained_tasks=nt, step=final_step)
            m, s = summarize_metric(sm, "asr")
            means.append(m)
            sems.append(s)
        offset = (idx - (len(combos) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=sems, capsize=3, label=label, color=color, hatch=hatch, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([OBJECTIVE_TITLES.get(o, o) for o in objectives])
    ax.set_ylabel("Attack Success Rate (ASR %)")
    ax.set_ylim(0, 110)
    ax.set_title("Trigger Universality: Single-Task vs. Multi-Task Optimization")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_archetype_asr(
    records: list[dict],
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    final_step: int = 500,
) -> None:
    """Plot 1x3 faceted bars of held-out ASR broken down by task interaction archetype.

    Visual encoding:
- Layout: Side-by-side subplots (one per objective) with the 4 task interaction
          archetypes (`Static`, `Bug+Bash`, `Inspect`, `Multi`) along the x-axis.
- Bars per archetype: Compares `Manual`, `1-Task Adaptive (Static-Trained)`, and
          `4-Task Adaptive (All Archetypes)` with ±1 SEM error bars across seeds.
          Because `1-Task Adaptive` trains solely on `TRAIN_TASKS[0]` (`Static`),
          `Static` measures same-archetype generalization while `Bug+Bash`, `Inspect`,
          and `Multi` measure cross-archetype transfer.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        out_path: Destination `.png` file path.
        objectives: Ordered tuple of objective subplots.
        final_step: Optimization checkpoint step evaluated for adaptive conditions.
    """
    fig, axes = plt.subplots(1, len(objectives), figsize=(5 * len(objectives), 4.6), sharey=True)
    if len(objectives) == 1:
        axes = [axes]

    series = [
        ("manual", None, None, "Manual", PALETTE["manual"]),
        ("adaptive", 1, final_step, "1-Task Adaptive (Static-Trained)", "#8e44ad"),
        ("adaptive", 4, final_step, "4-Task Adaptive (All Archetypes)", PALETTE["adaptive"]),
    ]
    x = np.arange(len(TASK_ARCHETYPES))
    width = 0.75 / len(series)

    for ax, obj in zip(axes, objectives):
        for s_idx, (cond, nt, stp, label, color) in enumerate(series):
            means, sems = [], []
            for arch_idx in range(len(TASK_ARCHETYPES)):
                arch_tids = set(range(arch_idx * 3, (arch_idx + 1) * 3))
                matched = [
                    r for r in records
                    if r["objective"] == obj
                    and r["condition"] == cond
                    and r["split"] == "held_out"
                    and r["n_trained_tasks"] == nt
                    and r["step"] == stp
                    and r["task_id"] in arch_tids
                ]
                by_seed: dict[int | None, list[dict]] = {}
                for rec in matched:
                    by_seed.setdefault(rec["seed"], []).append(rec)
                seed_asrs = [
                    sum(r["attack_success"] for r in recs) / len(recs) * 100
                    for _, recs in sorted(by_seed.items(), key=lambda kv: (-1 if kv[0] is None else kv[0]))
                ]
                m, s = mean_sem(seed_asrs)
                means.append(m)
                sems.append(0.0 if s is None else s)

            offset = (s_idx - (len(series) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=sems if cond == "adaptive" else None,
                capsize=3,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(TASK_ARCHETYPES)
        ax.set_title(OBJECTIVE_TITLES.get(obj, obj), fontsize=10)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Held-Out ASR (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    fig.suptitle("Held-Out ASR Across Task Interaction Archetypes", y=1.08, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Section E: Stealth & Benign Utility Preservation
# ==============================================================================


def plot_utility_and_dual(
    records: list[dict],
    out_path: pathlib.Path,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    conditions: tuple[str, ...] = ("naive", "random", "manual", "adaptive"),
    n_trained_tasks: int = 4,
    final_step: int = 500,
) -> None:
    """Plot grouped bars comparing `attack_success`, `utility_success`, and `dual_success`.

    Visual encoding:
        - Layout: Side-by-side subplots (one per objective) with `naive`, `random`, `manual`, and
          `adaptive` along the x-axis.
        - Bars per condition: Three adjacent bars for `ASR` (`attack_success`), `Util`
          (`utility_success` — `notes.py` modified and exits `0`), and `Dual`
          (`dual_success` — both succeeded simultaneously), with ±1 SEM error bars on
          `adaptive`.

    Args:
        records: Deduplicated trial records from `load_dedup_results`.
        out_path: Destination `.png` file path.
        objectives: Ordered tuple of objective subplots.
        conditions: Ordered tuple of conditions along each subplot's x-axis.
        n_trained_tasks: Training task count for the `adaptive` condition.
        final_step: Optimization checkpoint step for the `adaptive` condition.
    """
    fig, axes = plt.subplots(1, len(objectives), figsize=(4.5 * len(objectives), 4.8), sharey=True)
    if len(objectives) == 1:
        axes = [axes]

    metrics = [
        ("asr", "Raw Attack (ASR)", "#e74c3c"),
        ("util", "Benign Utility (Util)", "#3498db"),
        ("dual", "Stealthy Dual-Task (Dual)", "#2ecc71"),
        ("neither", "Neither Succeeded", "#95a5a6")
    ]
    for ax, obj in zip(axes, objectives):
        x = np.arange(len(conditions))
        width = 0.2
        for m_idx, (m_key, m_label, color) in enumerate(metrics):
            means, sems = [], []
            for cond in conditions:
                kw = {"n_trained_tasks": n_trained_tasks, "step": final_step} if cond == "adaptive" else {}
                sm = compute_seed_metrics(records, obj, cond, split="held_out", **kw)
                for row in (sm.values() if isinstance(sm, dict) else sm):
                    row["neither"] = 100.0 - (row["asr"] + row["util"] - row["dual"])
                m, s = summarize_metric(sm, m_key)
                m, s = summarize_metric(sm, m_key)
                means.append(m)
                sems.append(s)
            offset = (m_idx - 1.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=sems,
                capsize=3,
                label=m_label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_TITLES.get(c, c) for c in conditions])
        ax.set_title(OBJECTIVE_TITLES.get(obj, obj), fontsize=10)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Rate on Held-Out Tasks (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False)
    fig.suptitle("Attack Efficacy vs. Benign Utility Preservation", y=1.08, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Pipeline Entry Point
# ==============================================================================


def generate_all_figures(
    results_path: pathlib.Path = pathlib.Path("results.jsonl"),
    loss_dir: pathlib.Path = pathlib.Path("loss_logs"),
    triggers_path: pathlib.Path = pathlib.Path("triggers.jsonl"),
    out_dir: pathlib.Path = pathlib.Path("figures"),
    n_trained_tasks: int = 4,
    final_step: int = 500,
) -> None:
    """Render all evaluation figures from `results.jsonl`, `triggers.jsonl`, and `loss_logs/`.

    Args:
        results_path: Path to the JSONL trial log written by `trial.run_suite`.
        loss_dir: Directory containing per-step optimization loss JSONL logs.
        triggers_path: Path to the checkpoint trigger JSONL file written by `search.py`.
        out_dir: Output directory where `.png` figures are saved.
        n_trained_tasks: Default training task count for `adaptive` evaluations.
        final_step: Default optimization step for end-of-training comparisons.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_dedup_results(results_path)

    plot_baseline_ladder_asr(records, out_dir / "fig_baseline_ladder.png", n_trained_tasks=n_trained_tasks, final_step=final_step)
    plot_task_condition_heatmap(records, out_dir / "fig_task_heatmap.png", n_trained_tasks=n_trained_tasks, final_step=final_step)
    plot_convergence_curves(records, loss_dir, out_dir / "fig_convergence.png", n_trained_tasks=n_trained_tasks)
    plot_loss_asr_calibration(records, triggers_path, out_dir / "fig_calibration_scatter.png", n_trained_tasks=n_trained_tasks)
    plot_failure_breakdown(records, out_dir / "fig_failure_modes.png", n_trained_tasks=n_trained_tasks, final_step=final_step)
    plot_runtime_and_tool_calls(records, out_dir / "fig_runtime_toolcalls.png", n_trained_tasks=n_trained_tasks, final_step=final_step)
    plot_universality_bars(records, out_dir / "fig_universality_bars.png", final_step=final_step)
    plot_archetype_asr(records, out_dir / "fig_archetype_asr.png", final_step=final_step)
    plot_utility_and_dual(records, out_dir / "fig_utility_dual.png", n_trained_tasks=n_trained_tasks, final_step=final_step)
    
    print(f"Saved figures to {out_dir.resolve()}")


if __name__ == "__main__":
    generate_all_figures()