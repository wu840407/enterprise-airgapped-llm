#!/usr/bin/env python3
"""Generate benchmark charts from results_*.json.

Usage:
    python make_charts.py results_vllm.json [results_trtllm.json] --outdir charts/
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"vllm": "#3b7dd8", "trtllm": "#76b900"}   # vLLM blue, NVIDIA green
LABELS = {"vllm": "vLLM", "trtllm": "TensorRT-LLM"}


def load(paths):
    out = []
    for p in paths:
        d = json.load(open(p))
        out.append((d["meta"]["engine"], d))
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)


def chart_throughput_vs_batch(data, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for eng, d in data:
        rows = sorted(d["throughput_sweep"], key=lambda r: r["batch"])
        if not rows:
            continue
        x = [r["batch"] for r in rows]
        y = [r["throughput_tok_s"] for r in rows]
        ax.plot(x, y, "o-", color=COLORS.get(eng, "gray"), label=LABELS.get(eng, eng), lw=2)
        # 理想線性參考線
        if eng == data[0][0]:
            ax.plot(x, [y[0] * b / x[0] for b in x], "--", color="#bbb",
                    lw=1.2, label="ideal linear scaling")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "Throughput vs batch size (input 512, output 256)",
          "concurrent requests", "output tokens / s")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "throughput_vs_batch.png"), dpi=150)
    print("  throughput_vs_batch.png")


def chart_ttft_vs_input(data, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for eng, d in data:
        rows = sorted(d["latency_sweep"], key=lambda r: r["input_len"])
        if not rows:
            continue
        ax.plot([r["input_len"] for r in rows], [r["ttft_ms"]["p50"] for r in rows],
                "o-", color=COLORS.get(eng, "gray"), label=LABELS.get(eng, eng) + " p50", lw=2)
        ax.plot([r["input_len"] for r in rows], [r["ttft_ms"]["p95"] for r in rows],
                "^--", color=COLORS.get(eng, "gray"), alpha=0.45,
                label=LABELS.get(eng, eng) + " p95", lw=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 512, 2048, 4096])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "Time to first token vs prompt length (batch 1)",
          "input tokens", "TTFT (ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ttft_vs_input_len.png"), dpi=150)
    print("  ttft_vs_input_len.png")


def chart_tpot_vs_batch(data, outdir):
    """每 token 延遲隨併發的退化 —— 吞吐的代價。"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for eng, d in data:
        rows = sorted(d["throughput_sweep"], key=lambda r: r["batch"])
        if not rows:
            continue
        ax.plot([r["batch"] for r in rows], [r["tpot_ms"]["p50"] for r in rows],
                "o-", color=COLORS.get(eng, "gray"), label=LABELS.get(eng, eng), lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "Per-token latency vs batch size — the cost of throughput",
          "concurrent requests", "TPOT (ms/token)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tpot_vs_batch.png"), dpi=150)
    print("  tpot_vs_batch.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--outdir", default="charts")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    data = load(a.results)
    print("產生圖表 →", a.outdir)
    chart_throughput_vs_batch(data, a.outdir)
    chart_ttft_vs_input(data, a.outdir)
    chart_tpot_vs_batch(data, a.outdir)


if __name__ == "__main__":
    main()
