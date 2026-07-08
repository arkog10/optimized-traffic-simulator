#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from agents.dqn import DQNAgent
from trafficsim.config import SimConfig
from trafficsim.env import TrafficGridEnv

# Dark-ish theme consistent with the sim UI
COLORS = {
    "baseline": "#5b6474",
    "dqn": "#46d282",
    "grid": "#2a3038",
    "text": "#dce2ec",
}


def fixed_timer_action(step: int, period: int) -> int:
    cycle = step // period
    return 0 if cycle % 2 == 0 else 1


def run_controller(
    env: TrafficGridEnv,
    seed: int,
    agent: DQNAgent | None = None,
    period: int = 20,
) -> dict:
    obs, _ = env.reset(seed=seed)
    num_ix = env.config.grid_size**2
    local_dim = int(np.prod(env.observation_space.shape))
    if num_ix > 1:
        local_dim = local_dim // num_ix

    stopped_trace: list[int] = []
    cumulative_wait_trace: list[float] = []
    total_wait = 0
    phase_switches = 0
    steps = 0

    while True:
        if agent is None:
            action = fixed_timer_action(steps, period)
            if num_ix > 1:
                action = np.full(num_ix, action, dtype=np.int32)
        elif num_ix == 1:
            action = agent.act(obs, explore=False)
        else:
            local_obs = obs.reshape(num_ix, local_dim)
            action = np.array(
                [agent.act(local_obs[i], explore=False) for i in range(num_ix)],
                dtype=np.int32,
            )

        obs, _, _, truncated, info = env.step(action)
        stopped = info["stopped_cars"]
        stopped_trace.append(stopped)
        total_wait += stopped
        cumulative_wait_trace.append(float(total_wait))
        phase_switches += info.get("phase_switches", 0)
        steps += 1
        if truncated:
            break

    completed = float(info["completed"])
    avg_stopped = total_wait / max(steps, 1)
    return {
        "avg_stopped": avg_stopped,
        "throughput": completed,
        "total_wait": float(total_wait),
        "max_stopped": float(max(stopped_trace) if stopped_trace else 0),
        "wait_per_car": total_wait / max(completed, 1),
        "phase_switches": float(phase_switches),
        "congestion_steps": float(sum(1 for s in stopped_trace if s > avg_stopped)),
        "stopped_trace": stopped_trace,
        "cumulative_wait_trace": cumulative_wait_trace,
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#11141a")
    ax.figure.set_facecolor("#11141a")
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], alpha=0.35, linewidth=0.6)


def plot_zoomed_bars(results_dir: Path, baseline: list[dict], dqn: list[dict]) -> None:
    metrics = [
        ("avg_stopped", "Avg stopped cars / step", "lower"),
        ("total_wait", "Total wait (car-steps)", "lower"),
        ("wait_per_car", "Wait per completed car", "lower"),
        ("max_stopped", "Peak congestion (max stopped)", "lower"),
        ("phase_switches", "Phase switches / episode", "neutral"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()

    for ax, (key, title, _) in zip(axes_flat[: len(metrics)], metrics, strict=True):
        b_vals = [r[key] for r in baseline]
        d_vals = [r[key] for r in dqn]
        b_mean, d_mean = np.mean(b_vals), np.mean(d_vals)
        positions = [0, 1]
        means = [b_mean, d_mean]
        colors = [COLORS["baseline"], COLORS["dqn"]]

        bars = ax.bar(positions, means, color=colors, width=0.55, edgecolor="white", linewidth=0.6)
        ax.scatter(
            [0] * len(b_vals),
            b_vals,
            color="white",
            alpha=0.55,
            s=18,
            zorder=3,
        )
        ax.scatter(
            [1] * len(d_vals),
            d_vals,
            color="white",
            alpha=0.55,
            s=18,
            zorder=3,
        )

        lo = min(b_vals + d_vals)
        hi = max(b_vals + d_vals)
        pad = (hi - lo) * 0.18 if hi > lo else hi * 0.12
        ax.set_ylim(max(0, lo - pad), hi + pad)

        delta_pct = (b_mean - d_mean) / b_mean * 100 if b_mean else 0
        ax.text(
            0.5,
            hi + pad * 0.15,
            f"{delta_pct:+.1f}%",
            ha="center",
            va="bottom",
            color=COLORS["dqn"] if delta_pct > 0 else COLORS["text"],
            fontsize=10,
            transform=ax.get_xaxis_transform(),
        )

        ax.set_xticks(positions, ["Fixed timer", "DQN"])
        ax.set_title(title, fontsize=11, pad=8)
        _style_axes(ax)
        for bar, val in zip(bars, means, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                color=COLORS["text"],
                fontsize=9,
            )

    axes_flat[len(metrics)].axis("off")
    fig.suptitle("Controller comparison (mean + per-seed dots, zoomed axes)", color=COLORS["text"], y=0.96)
    fig.subplots_adjust(top=0.92, hspace=0.38, wspace=0.28)
    fig.savefig(results_dir / "comparison_bars.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_boxplot(results_dir: Path, baseline: list[dict], dqn: list[dict]) -> None:
    keys = ["avg_stopped", "total_wait", "wait_per_car", "max_stopped"]
    labels = ["Avg stopped\n/ step", "Total wait\n(car-steps)", "Wait / car", "Peak\ncongestion"]

    fig, ax = plt.subplots(figsize=(9, 5))
    data = []
    positions = []
    colors = []
    for i, key in enumerate(keys):
        data.append([r[key] for r in baseline])
        data.append([r[key] for r in dqn])
        positions.extend([i * 3, i * 3 + 1])
        colors.extend([COLORS["baseline"], COLORS["dqn"]])

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.9,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"color": COLORS["grid"]},
        capprops={"color": COLORS["grid"]},
        flierprops={"markerfacecolor": "white", "markersize": 4, "alpha": 0.6},
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    centers = [i * 3 + 0.5 for i in range(len(keys))]
    ax.set_xticks(centers, labels)
    ax.set_title("Metric spread across evaluation seeds", pad=12)
    ax.set_ylabel("Value (lower is better for all shown)")
    _style_axes(ax)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=COLORS["baseline"], label="Fixed timer"),
            Patch(facecolor=COLORS["dqn"], label="DQN"),
        ],
        loc="upper right",
        facecolor="#11141a",
        edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )
    fig.tight_layout()
    fig.savefig(results_dir / "distribution_boxplot.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_stopped_timeseries(results_dir: Path, baseline: list[dict], dqn: list[dict]) -> None:
    max_len = max(len(r["stopped_trace"]) for r in baseline + dqn)
    t = np.arange(max_len)

    def mean_std(traces: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        padded = np.array(
            [trace + [trace[-1]] * (max_len - len(trace)) for trace in traces],
            dtype=float,
        )
        return padded.mean(axis=0), padded.std(axis=0)

    b_mean, b_std = mean_std([r["stopped_trace"] for r in baseline])
    d_mean, d_std = mean_std([r["stopped_trace"] for r in dqn])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(t, b_mean - b_std, b_mean + b_std, color=COLORS["baseline"], alpha=0.25)
    ax.fill_between(t, d_mean - d_std, d_mean + d_std, color=COLORS["dqn"], alpha=0.25)
    ax.plot(t, b_mean, color=COLORS["baseline"], linewidth=2, label="Fixed timer")
    ax.plot(t, d_mean, color=COLORS["dqn"], linewidth=2, label="DQN")

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Stopped cars")
    ax.set_title("Congestion over time (mean ± std across seeds)")
    ax.set_ylim(0, max(b_mean.max(), d_mean.max()) * 1.15)
    ax.legend(facecolor="#11141a", edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(results_dir / "stopped_timeseries.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cumulative_wait(results_dir: Path, baseline: list[dict], dqn: list[dict]) -> None:
    max_len = max(len(r["cumulative_wait_trace"]) for r in baseline + dqn)
    t = np.arange(max_len)

    def mean_std(traces: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        padded = np.array(
            [trace + [trace[-1]] * (max_len - len(trace)) for trace in traces],
            dtype=float,
        )
        return padded.mean(axis=0), padded.std(axis=0)

    b_mean, b_std = mean_std([r["cumulative_wait_trace"] for r in baseline])
    d_mean, d_std = mean_std([r["cumulative_wait_trace"] for r in dqn])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(t, b_mean - b_std, b_mean + b_std, color=COLORS["baseline"], alpha=0.2)
    ax.fill_between(t, d_mean - d_std, d_mean + d_std, color=COLORS["dqn"], alpha=0.2)
    ax.plot(t, b_mean, color=COLORS["baseline"], linewidth=2, label="Fixed timer")
    ax.plot(t, d_mean, color=COLORS["dqn"], linewidth=2, label="DQN")

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Cumulative wait (car-steps)")
    ax.set_title("Total delay accumulated during episode")
    ax.legend(facecolor="#11141a", edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(results_dir / "cumulative_wait.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_improvement_lollipop(results_dir: Path, summary: dict) -> None:
    keys = [
        ("avg_stopped", "Avg stopped / step"),
        ("total_wait", "Total wait"),
        ("wait_per_car", "Wait per car"),
        ("max_stopped", "Peak congestion"),
    ]
    labels = [label for _, label in keys]
    pcts = [summary["improvement"][f"{key}_reduction_pct"] for key, _ in keys]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    ax.hlines(y, 0, pcts, color=COLORS["grid"], linewidth=2)
    ax.scatter(pcts, y, color=COLORS["dqn"], s=120, zorder=3, edgecolors="white", linewidths=0.8)

    for pct, yi in zip(pcts, y, strict=True):
        ax.text(pct + 0.8, yi, f"{pct:.1f}%", va="center", color=COLORS["text"], fontsize=10)

    ax.axvline(0, color=COLORS["text"], linewidth=0.8, alpha=0.4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Reduction vs fixed timer (%)")
    ax.set_title("DQN improvement by metric")
    ax.set_xlim(0, max(pcts) * 1.25 + 2)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(results_dir / "improvement_lollipop.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def summarize(baseline: list[dict], dqn: list[dict]) -> dict:
    keys = [
        "avg_stopped",
        "throughput",
        "total_wait",
        "wait_per_car",
        "max_stopped",
        "phase_switches",
        "congestion_steps",
    ]

    def stats(rows: list[dict], key: str) -> dict:
        vals = [r[key] for r in rows]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    summary: dict = {
        "baseline": {k: stats(baseline, k) for k in keys},
        "dqn": {k: stats(dqn, k) for k in keys},
        "improvement": {},
    }

    for key in keys:
        b = summary["baseline"][key]["mean"]
        d = summary["dqn"][key]["mean"]
        if b:
            summary["improvement"][f"{key}_reduction_pct"] = float((b - d) / b * 100)
        else:
            summary["improvement"][f"{key}_reduction_pct"] = 0.0

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DQN vs fixed timer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dqn_final.pt")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--period", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=1)
    args = parser.parse_args()

    config = SimConfig(grid_size=args.grid_size)
    env = TrafficGridEnv(config=config, max_steps=1000)

    baseline = [run_controller(env, seed=i, period=args.period) for i in range(args.seeds)]

    agent = DQNAgent.load(args.checkpoint)
    agent.epsilon = 0.0
    dqn = [run_controller(env, seed=i, agent=agent) for i in range(args.seeds)]

    summary = summarize(baseline, dqn)
    print("baseline means:", {k: v["mean"] for k, v in summary["baseline"].items()})
    print("dqn means:", {k: v["mean"] for k, v in summary["dqn"].items()})
    print("improvement %:", summary["improvement"])

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    plot_zoomed_bars(results_dir, baseline, dqn)
    plot_boxplot(results_dir, baseline, dqn)
    plot_stopped_timeseries(results_dir, baseline, dqn)
    plot_cumulative_wait(results_dir, baseline, dqn)
    plot_improvement_lollipop(results_dir, summary)

    payload = {
        "generated_at": str(date.today()),
        "grid_size": args.grid_size,
        "seeds": args.seeds,
        "episode_steps": env.max_steps,
        "baseline": {
            "controller": "fixed_timer",
            "period_steps": args.period,
            **{k: summary["baseline"][k] for k in summary["baseline"]},
        },
        "dqn": {
            "controller": "trained_dqn",
            "checkpoint": args.checkpoint,
            **{k: summary["dqn"][k] for k in summary["dqn"]},
        },
        "improvement": summary["improvement"],
        "figures": [
            "comparison_bars.png",
            "distribution_boxplot.png",
            "stopped_timeseries.png",
            "cumulative_wait.png",
            "improvement_lollipop.png",
        ],
    }

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"saved metrics to {metrics_path}")
    for name in payload["figures"]:
        print(f"saved {results_dir / name}")

    env.close()


if __name__ == "__main__":
    main()
