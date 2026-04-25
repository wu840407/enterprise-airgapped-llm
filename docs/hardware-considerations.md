# Hardware Considerations

## The Turing Constraint

The reference deployment uses 2× NVIDIA Quadro RTX 6000 Passive cards. These are 2018-vintage Turing-class GPUs (sm_75) commonly found in pre-Ampere enterprise servers that haven't yet been refreshed.

What Turing **does** support:
- FP32, FP16
- INT8, INT4 (Tensor Cores)
- Standard CUDA 12.x feature set

What Turing **does not** support:
- BF16 (added in Ampere)
- FP8 (added in Hopper)
- Many newer Flash Attention 2/3 kernels (require sm_80+)

These constraints cascade through every layer of the stack. The fact that this deployment achieves production-grade performance on Turing is itself an architectural achievement, not a baseline assumption.

---

## VRAM Budget Analysis

With 48 GB total VRAM (24 GB × 2), the candidate models break down as follows:

| Model | Format | Weights | KV Cache (16K ctx) | Total | Fits? |
|---|---|---|---|---|---|
| Llama 3.3 70B | FP16 | 140 GB | ~5 GB | 145 GB | ❌ |
| Qwen2.5-Coder 32B | FP16 | 64 GB | ~3 GB | 67 GB | ❌ |
| Qwen2.5-Coder 32B | AWQ INT4 | ~19 GB | ~3 GB | ~22 GB | ✅ on 1 GPU |
| **Qwen3-Coder-30B-A3B** | **AWQ INT4** | **~17 GB** | **~3 GB** | **~20 GB** | ✅ + headroom |
| Kimi-Dev 72B | AWQ INT4 | ~38 GB | ~5 GB | ~43 GB | ⚠ tight |

The MoE architecture of Qwen3-Coder-30B-A3B is the deciding factor. Despite having 30 B total parameters, only ~3 B activate per forward pass, so:

- **Memory cost** is dominated by total parameters (~17 GB at INT4)
- **Compute cost** is dominated by activated parameters (~3 B → ~14B-dense throughput)

This decoupling is what enables a 30B-class model to serve at the speed of a much smaller dense model.

---

## Tensor Parallelism Decisions

vLLM's tensor parallelism distributes the model's attention heads across GPUs. Key constraints:

1. `tensor_parallel_size` must evenly divide the number of attention heads
2. PCIe topology matters — NUMA-spanning GPU pairs add latency
3. Odd numbers of GPUs (e.g., 3) almost never divide head counts cleanly

For this deployment, `--tensor-parallel-size=2` distributes the model evenly across both GPUs, with each card holding ~10 GB of weights and serving its share of attention computation.

---

## Storage I/O Profile

Cold-start performance depends heavily on the storage tier:

| Storage | 17 GB Model Load Time |
|---|---|
| Single SATA SSD | ~60 s |
| Single NVMe SSD | ~25 s |
| **6× SAS SSD RAID 10** | **~31 s** |
| 4× NVMe RAID 0 | ~10 s |

The SAS SSD RAID 10 hits a sweet spot of throughput, redundancy, and enterprise serviceability. NVMe would be faster but harder to maintain RAID 10 across hot-swap bays in older 2U chassis.

---

## Power and Thermal

- 2× Quadro RTX 6000 Passive: 250 W × 2 = 500 W under load
- The "Passive" suffix means **no onboard fan** — the chassis must move air across the heatsinks
- Verify chassis fan profile is set to "Performance" or higher in iDRAC; default profiles can throttle GPUs
- The reference deployment uses datacenter-grade chilled water cooling, eliminating thermal as a constraint

---

## Power Supply Redundancy

Dell R740 dual 2400 W PSUs provide ample headroom:

| Load Scenario | Power Draw |
|---|---|
| Idle | ~250 W |
| Inference under load | ~700 W |
| Peak with all CPU + GPU saturated | ~1100 W |
| Headroom for expansion | ~1700 W |

This allows future expansion to 3–4 GPUs without PSU upgrade. The dual PSU configuration also provides A+B power feed redundancy when each PSU is on a separate UPS.
