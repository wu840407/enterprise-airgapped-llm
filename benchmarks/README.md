# TensorRT-LLM vs vLLM on a single RTX 3090

Like-for-like measurements of two LLM serving engines, driven by one client against
identical prompts, with every number cross-checked against a first-principles roofline.

## Result: they perform the same, and that is the interesting part

Across every configuration measured, the two engines land within **±6%** of each other —
inside the run-to-run noise of a consumer GPU.

![Throughput vs batch](charts/throughput_vs_batch.png)

| concurrency | vLLM | TensorRT-LLM | Δ |
|---:|---:|---:|---:|
| 1 | 46.8 tok/s | 46.3 tok/s | −1.0% |
| 4 | 175.8 tok/s | 171.4 tok/s | −2.5% |
| 8 | 317.5 tok/s | 321.6 tok/s | +1.3% |
| 16 | 516.9 tok/s | 529.9 tok/s | +2.5% |
| 32 | **786.6 tok/s** | **802.0 tok/s** | +2.0% |

| prompt tokens | vLLM TTFT | TensorRT-LLM TTFT | Δ |
|---:|---:|---:|---:|
| 128 | 62.5 ms | 65.2 ms | +4.3% |
| 512 | 148.1 ms | 150.8 ms | +1.8% |
| 2048 | 473.2 ms | 500.8 ms | +5.8% |
| 4096 | 962.8 ms | 990.8 ms | +2.9% |

**Neither engine can beat physics.** Both are already running at 84–98% of what the hardware
allows, so there is no headroom left for one to out-optimize the other:

| Phase | Bound by | Roofline | vLLM | TensorRT-LLM |
|---|---|---|---|---|
| Decode (batch 1) | memory bandwidth — 936 GB/s | 17.50 ms/token | 20.84 ms (84%) | 21.04 ms (83%) |
| Prefill (4096 tok) | compute — 71 TFLOPS | 945 ms | 962.8 ms (98%) | 990.8 ms (95%) |

Engine choice cannot move a number the hardware has already pinned. **On this setup the
decision is operational, not performance-driven.**

---

## Where they actually differ

| | vLLM 0.27.1 | TensorRT-LLM 1.2.0rc1 |
|---|---|---|
| Time from launch to serving | **~40 s** | **202 s** (5×) |
| Batch ceiling | runtime flag — change and restart | **fixed at build time** — change means recompiling |
| KV cache allocated | 5.9 GiB / 42,976 tokens | 4.15 GiB / 30,240 tokens |
| Memory knob | `gpu-memory-utilization` (total budget) | `kv_cache_free_gpu_memory_fraction` (share of what's left) |

That 202-second engine preparation is the real cost of ahead-of-time compilation, and it is
paid again on every configuration change. For a team iterating on models daily that dominates;
for a team serving one fixed model for months it is a one-time cost that buys specialized kernels.

**The kernels just have nothing to win here.** TensorRT-LLM's advantages — FP8 on Hopper/Ada,
optimized multi-GPU collectives, INT4 kernels — are all things this configuration does not use.

---

## Setup

| | |
|---|---|
| GPU | RTX 3090, 24 GB GDDR6X — 936 GB/s, 71 TFLOPS FP16 (FP32 accumulate) |
| Driver | 610.62, WSL2 kernel 6.6.87.2 |
| Model | `Qwen/Qwen3-8B` — 36 layers, 32 attention heads, 8 KV heads (GQA), head_dim 128 |
| Precision | BF16, no quantization, identical on both engines |
| vLLM | 0.27.1 · `--max-model-len 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching` |
| TensorRT-LLM | 1.2.0rc1 (NGC) · `--backend pytorch --max_batch_size 32 --max_num_tokens 8192 --kv_cache_free_gpu_memory_fraction 0.85` |

KV cache is the binding constraint: weights take ~15.3 GiB, and this model costs 144 KB/token
of cache — 2.6× a Qwen2.5-7B, because its GQA ratio is 4:1 rather than 7:1. That number sets
how far the batch sweep can go before OOM.

### An honest asymmetry

The two engines expose *different* memory knobs and ended up with different KV budgets
(42,976 vs 30,240 tokens). Each was given its documented default convention rather than being
forced to match, so this is a comparison of engines as they are typically deployed. Both budgets
comfortably cover the sweep (batch 32 × 768 tokens = 24,576), so it does not affect the numbers
here — but it would matter at long context, and pretending the setups were identical would be
dishonest.

## Method

One engine-agnostic client (`bench_client.py`) drives any OpenAI-compatible endpoint over
streaming HTTP, so both engines see identical code, prompts, and measurement logic.

- **3 warmup bursts** discarded, then **5 measured bursts** per configuration
- Reports **p50 and p95**, never bare averages
- `temperature=0`, `ignore_eos=true` — every request emits exactly 256 tokens
- Peak VRAM sampled at 4 Hz from a background thread
- Every prompt carries a UUID prefix so no two requests share a cache-able prefix
- **One dimension varies at a time** — a batch × context cartesian product would OOM at the
  corners and confound the two effects anyway

```bash
# vLLM
vllm serve ~/models/qwen3-8b --served-model-name qwen3-8b --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching

# TensorRT-LLM
docker run -d --gpus all --ipc=host -p 8000:8000 -v ~/models:/models \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc1 \
    trtllm-serve /models/qwen3-8b --host 0.0.0.0 --port 8000 --backend pytorch \
    --max_batch_size 32 --max_num_tokens 8192 --kv_cache_free_gpu_memory_fraction 0.85

# same client against either
python bench_client.py --engine {vllm|trtllm} --base-url http://127.0.0.1:8000/v1 \
    --model qwen3-8b --tokenizer ~/models/qwen3-8b --out results_{engine}.json
python make_charts.py results_vllm.json results_trtllm.json --outdir charts/
```

---

## Reading the curves

### Prefill is compute-bound

![TTFT vs prompt length](charts/ttft_vs_input_len.png)

TTFT scales close to linearly with prompt length on both engines, which is what a compute-bound
phase looks like: prefill costs `2 × P × N` FLOPs, so doubling the prompt doubles the work.

Predicted from the roofline: `2 × 8.19e9 × 4096 / 71e12 = 945 ms`. Measured 962.8 ms (vLLM)
and 990.8 ms (TensorRT-LLM) — **95–98% of what the silicon can do**. The only way to cut TTFT
on this hardware is to do less work: shorter prompts, prefix caching, chunked prefill.

### Decode is memory-bound

Per-token latency sits at ~21 ms on both engines regardless of prompt length, because decode
re-reads the whole weight matrix for every token and that dominates everything else.

Predicted: `16.38 GB / 936 GB/s = 17.50 ms`. Measured 20.84 / 21.04 ms → **83–84% of the
bandwidth roofline**. The remaining gap is kernel launch overhead, sampling, and the attention
work this simple model ignores.

**This is why quantization matters more than raw FLOPs for single-stream generation.** Halving
the weight bytes would nearly halve decode latency; adding compute would do nothing at all.

### Batching trades latency for throughput

![TPOT vs batch](charts/tpot_vs_batch.png)

32× the concurrency buys ~17× the throughput for roughly 50% higher per-token latency — the
weights are read once per step and amortized across every request in flight.

Scaling stays near-linear to batch 4 and then bends. That inflection is where prefill work from
arriving requests starts competing with decode for the same SMs: across the sweep TTFT degrades
~14× while TPOT degrades only ~1.5×.

**The knee is a serving decision, not a hardware limit.** Latency-sensitive workloads should cap
concurrency near 4–8; batch and offline workloads should push past 32 and accept multi-second TTFT.

---

## What went wrong first (and how it was caught)

The first vLLM run reported TTFT of **42 ms at a 4096-token prompt**. The roofline says prefill
alone must cost ~945 ms. A 20× discrepancy is never a pleasant surprise — it means the
measurement is wrong, not that the engine is magic.

Cause: the harness reused one fixed prompt across all requests, so vLLM's automatic prefix
caching served prefill straight from cache. The server log confirmed it — `Prefix cache hit
rate: 75.3%`.

Two fixes, both applied: every request now carries a UUID-prefixed prompt, and the server runs
with `--no-enable-prefix-caching`. Decode numbers were unaffected (generation does real work
either way), but every TTFT figure in that first run was measuring a cache lookup.

**Compute the expected value before trusting a measured one.** The roofline was not decoration —
it is what caught the bug.

Side note: disabling prefix caching *increased* usable KV cache from 4.91 to 5.9 GiB, since the
engine no longer reserves a cache region.

## Running vLLM under WSL2

Out of the box it fails three times. All three are environment issues, not vLLM bugs:

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve ...
```

1. `RuntimeError: UVA is not available` — misleading. UVA *is* available
   (`cudaDevAttrUnifiedAddressing = 1`, pinned allocation succeeds). vLLM disables pinned memory
   on WSL by default and gates it behind `VLLM_WSL2_ENABLE_PIN_MEMORY`.
2. `Could not find nvcc` — `deep_gemm` JIT-compiles kernels; no CUDA toolkit in this WSL image.
3. The same nvcc error again, this time from FlashInfer's sampling kernel. Greedy decoding
   does not need it.

TensorRT-LLM had none of these — the NGC container ships a complete toolchain, which is part of
what the 20 GB image and 202-second startup buy you.

---

## Files

```
bench_client.py       engine-agnostic measurement harness
make_charts.py        chart generation
results_vllm.json     raw measurements, vLLM
results_trtllm.json   raw measurements, TensorRT-LLM
charts/               generated figures
```

## What would change the answer

This configuration puts both engines against the same two hardware walls, so it cannot separate
them. The comparisons where TensorRT-LLM should pull ahead all involve leaving that regime:

- **FP8 on Ada/Hopper** — native tensor-core support the 3090 (Ampere) does not have
- **INT4/AWQ weights** — moves decode off the bandwidth wall, where kernel quality starts to matter
- **Multi-GPU tensor parallel** — optimized collectives versus a generic path
- **Sustained high batch** — where kernel efficiency, not weight traffic, sets the ceiling
