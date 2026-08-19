#!/usr/bin/env python3
"""Engine-agnostic LLM inference benchmark client.

Measures TTFT / TPOT / throughput / VRAM against any OpenAI-compatible endpoint,
so vLLM and TensorRT-LLM are exercised by identical client code and identical prompts.

Usage:
    python bench_client.py --engine vllm --base-url http://127.0.0.1:8000/v1 \
        --model qwen3-8b --tokenizer ~/models/qwen3-8b --out results_vllm.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

# ---------------------------------------------------------------- VRAM sampler


class VramSampler:
    """Poll nvidia-smi in the background; report peak MiB over the window."""

    def __init__(self, interval=0.25):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                self.peak = max(self.peak, int(out[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = 0
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)


# ---------------------------------------------------------------- prompt build


def build_prompt(tokenizer, n_tokens: int, unique: bool = True) -> str:
    """Prompt of ~n_tokens.

    `unique=True` prepends random tokens so no two requests share a prefix.
    Without this, vLLM's automatic prefix caching serves prefill from cache and
    TTFT measures a cache hit rather than real prefill work — we hit exactly
    that: measured TTFT at 4k prompt was ~20x below the compute roofline.
    """
    head = "%s. " % uuid.uuid4().hex if unique else ""
    filler = ("The quick brown fox jumps over the lazy dog. "
              "Machine learning systems process data efficiently. ") * 400
    ids = tokenizer.encode(head + filler)[:n_tokens]
    return tokenizer.decode(ids)


# ---------------------------------------------------------------- single request


def one_request(base_url, model, prompt, max_tokens, timeout=600):
    """Stream one completion. Returns (ttft_s, total_s, n_output_tokens)."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,          # force exactly max_tokens so runs are comparable
    }
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            body = raw[6:]
            if body.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tok += 1
    total = time.perf_counter() - t0
    return ttft, total, n_tok


# ---------------------------------------------------------------- one config


def run_config(base_url, model, make_prompt, max_tokens, concurrency, repeats, warmup):
    """Run one (input_len, batch) config. `make_prompt` returns a FRESH prompt
    per call so every request is unique — otherwise prefix caching poisons TTFT."""
    def burst():
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(
                lambda _: one_request(base_url, model, make_prompt(), max_tokens),
                range(concurrency)))
            wall = time.perf_counter() - t0
        return res, wall

    for _ in range(warmup):
        burst()

    ttfts, tpots, throughputs = [], [], []
    with VramSampler() as vram:
        for _ in range(repeats):
            res, wall = burst()
            out_tokens = sum(r[2] for r in res)
            for ttft, total, n in res:
                if ttft is None or n < 2:
                    continue
                ttfts.append(ttft * 1000)
                tpots.append((total - ttft) / (n - 1) * 1000)
            throughputs.append(out_tokens / wall)

    pct = lambda xs, q: (statistics.quantiles(xs, n=100)[q - 1] if len(xs) > 1 else xs[0])
    return {
        "ttft_ms": {"p50": round(statistics.median(ttfts), 2), "p95": round(pct(ttfts, 95), 2)},
        "tpot_ms": {"p50": round(statistics.median(tpots), 2), "p95": round(pct(tpots, 95), 2)},
        "throughput_tok_s": round(statistics.median(throughputs), 2),
        "vram_peak_mib": vram.peak,
        "samples": len(ttfts),
    }


# ---------------------------------------------------------------- env metadata


def collect_env(engine, model_path):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=15).stdout.strip()
        except Exception:
            return "n/a"
    meta = {
        "engine": engine,
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "vram_total_mib": sh("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "gpu_clock_locked": sh("nvidia-smi --query-gpu=clocks.applications.graphics "
                               "--format=csv,noheader"),
        "date": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "model_path": model_path,
    }
    try:
        import torch
        meta["torch"] = torch.__version__
    except Exception:
        pass
    if engine == "vllm":
        try:
            import vllm
            meta["vllm"] = vllm.__version__
        except Exception:
            pass
    return meta


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=["vllm", "trtllm"])
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True, help="served model name (must match server)")
    p.add_argument("--tokenizer", required=True, help="local path to tokenizer")
    p.add_argument("--out", required=True)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    # 一次只變動一個維度 —— 混著變會分不出效應來自哪個
    p.add_argument("--latency-input-lens", default="128,512,2048,4096",
                   help="batch fixed at 1, sweep input length")
    p.add_argument("--throughput-batches", default="1,2,4,8,16,32",
                   help="input fixed at --throughput-input-len, sweep concurrency")
    p.add_argument("--throughput-input-len", type=int, default=512)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.tokenizer))

    result = {"meta": collect_env(a.engine, a.tokenizer),
              "config": {"output_len": a.output_len, "repeats": a.repeats,
                         "warmup": a.warmup},
              "latency_sweep": [], "throughput_sweep": []}

    print("=" * 62)
    print(" %s @ %s" % (a.engine, a.base_url))
    print("=" * 62)

    print("\n--- 延遲掃描 (batch=1, 變動 input length) ---")
    for n in [int(x) for x in a.latency_input_lens.split(",")]:
        m = run_config(a.base_url, a.model, lambda n=n: build_prompt(tok, n),
                       a.output_len, 1, a.repeats, a.warmup)
        m.update(input_len=n, batch=1)
        result["latency_sweep"].append(m)
        print("  in=%-5d TTFT p50 %7.1f ms | TPOT p50 %5.2f ms | %6.1f tok/s | VRAM %d MiB"
              % (n, m["ttft_ms"]["p50"], m["tpot_ms"]["p50"],
                 m["throughput_tok_s"], m["vram_peak_mib"]))
        json.dump(result, open(a.out, "w"), indent=2)

    print("\n--- 吞吐掃描 (input=%d, 變動 batch) ---" % a.throughput_input_len)
    mk = lambda: build_prompt(tok, a.throughput_input_len)
    for b in [int(x) for x in a.throughput_batches.split(",")]:
        try:
            m = run_config(a.base_url, a.model, mk, a.output_len, b, a.repeats, a.warmup)
        except Exception as e:
            print("  batch=%-3d ✗ %s" % (b, str(e)[:60]))
            continue
        m.update(input_len=a.throughput_input_len, batch=b)
        result["throughput_sweep"].append(m)
        print("  batch=%-3d TTFT p50 %7.1f ms | TPOT p50 %5.2f ms | %7.1f tok/s | VRAM %d MiB"
              % (b, m["ttft_ms"]["p50"], m["tpot_ms"]["p50"],
                 m["throughput_tok_s"], m["vram_peak_mib"]))
        json.dump(result, open(a.out, "w"), indent=2)

    print("\n✓ 寫入 %s" % a.out)


if __name__ == "__main__":
    main()
