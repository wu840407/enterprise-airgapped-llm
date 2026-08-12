# Troubleshooting

Field notes from a live air-gapped deployment. Every entry here was hit in production on Turing-class hardware — these are not hypothetical failure modes.

---

## vLLM CLI

### `unrecognized arguments: serve`

**Symptom**
```
vllm: error: unrecognized arguments: serve
```

**Cause** — the official `vllm/vllm-openai` image already declares `ENTRYPOINT ["vllm", "serve"]`. Repeating `serve` in the Compose `command:` produces `vllm serve serve ...`.

**Fix** — the `command:` array starts with the model path as a positional argument:

```yaml
command:
  - "/models/YOUR-MODEL"
  - "--served-model-name"
  - "YOUR-NAME"
```

---

### `unrecognized arguments: --swap-space 16`

**Cause** — `--swap-space` was removed in vLLM 0.19.x. CPU offload of KV cache is handled differently now.

**Fix** — drop the flag. On a correctly sized deployment (model weights + KV cache within VRAM) it was never load-bearing anyway.

---

### `unrecognized arguments: --disable-log-requests`

**Cause** — removed in newer releases.

**Fix** — use `--uvicorn-log-level warning` to reduce log noise instead.

---

### `argument --reasoning-parser: expected one argument`

**Cause** — this error is **misleading**. It usually means an *earlier* flag in the argument list was unrecognized, and argparse's recovery lost track of where values belong. In our case the culprit was:

```yaml
- "--chat-template-kwargs"
- '{"enable_thinking": false}'
```

`--chat-template-kwargs` is **not a `vllm serve` CLI flag**. Template kwargs are per-request, passed in the API body:

```json
{
  "model": "...",
  "messages": [],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

**Fix** — remove it from the CLI. To expose a "fast mode" to end users, inject it at the frontend layer instead (see *Serving a no-thinking variant* below).

**Lesson** — when argparse blames a flag that looks correct, audit *every* flag above it. The reported flag is often the innocent bystander.

---

### Pydantic `ValidationError` on quantization

**Symptom**
```
Value error, Quantization method specified in the model config (compressed-tensors)
does not match the quantization method specified in the `quantization` argument (awq)
```

**Cause** — modern AWQ checkpoints (from `llm-compressor`) declare `compressed-tensors` in `config.json`. Passing `--quantization awq` overrides and conflicts.

**Fix** — omit `--quantization` entirely. vLLM 0.10+ auto-detects from `quant_method`. Manually specifying it is now an anti-pattern that older tutorials still propagate.

---

## Capacity and context length

### `max-model-len` exceeding the actual KV cache pool

**Symptom** — the server starts cleanly, short chats work, but long conversations or reasoning-heavy prompts stall mid-generation and never return an answer. Users retry and it "sometimes works."

**Diagnosis** — compare two numbers in the startup log:

```
GPU KV cache size: 73,696 tokens
Maximum concurrency for 131,072 tokens per request: 2.21x
```

If `max-model-len` (131,072 here) is larger than the entire KV cache pool (73,696), a single long request cannot physically fit. The `2.21x` concurrency figure is an optimistic estimate assuming prefix caching and paging — it is not a guarantee.

**Fix** — set `max-model-len` at or below the measured pool size:

```yaml
- "--max-model-len"
- "65536"
```

**Why this bites reasoning models specifically** — models with a thinking mode emit thousands of tokens inside a `<think>` block before the visible answer. That inflates the KV footprint of a single turn far beyond what the user's prompt suggests. A non-thinking variant of the same model on identical settings will appear to work fine, which misleads diagnosis toward "the model is broken" rather than "the budget is wrong."

**Method** — always read the actual `GPU KV cache size` from the startup log after any change to `--gpu-memory-utilization`, `--max-num-seqs`, or the model itself. Do not assume it scales predictably.

---

### Startup hangs during KV cache allocation

**Symptom**
```
No available shared memory broadcast block found in 60 seconds.
This typically happens when some processes are hanging or doing some
time-consuming work (e.g. compilation, weight/kv cache quantization).
```
repeating every 60 seconds.

**Diagnosis** — check `nvidia-smi` in a second terminal:

| Observation | Meaning |
|---|---|
| VRAM climbing, GPU-Util non-zero | Genuinely working. Wait. |
| VRAM flat at weights-only, GPU-Util 0% | Hung. Requested KV cache is unachievable. |

**Fix** — reduce `--max-model-len` one step (128K to 96K to 64K) and restart. There is always a value that allocates cleanly.

---

## NVIDIA driver lifecycle

### `Failed to initialize NVML: Driver/library version mismatch`

The single most disruptive recurring failure in this deployment. Docker GPU support dies entirely; every GPU container fails with:

```
failed to initialize NVML: Driver/library version mismatch
```

**Root cause** — the loaded kernel module and the userspace libraries are on different versions:

```bash
cat /proc/driver/nvidia/version   # NVRM version: 580.159.03    (kernel module)
nvidia-smi                        # NVML library version: 580.173  (userspace)
```

This drift happens whenever apt updates NVIDIA packages while the system is running. The kernel module only changes on reboot.

**Three compounding traps:**

1. **Multiple driver branches installed simultaneously.** A machine can accumulate 550, 570, and 580 metapackages across upgrade cycles. They share underlying libraries; touching any one can swap what the others depend on.

2. **Holding only the metapackage is insufficient.** `apt-mark hold nvidia-driver-580-server` does *not* prevent `libnvidia-compute-580` from upgrading independently. Hold everything:

   ```bash
   sudo apt-mark hold $(dpkg -l | grep -E "nvidia|libnvidia" | awk '{print $2}')
   apt-mark showhold   # verify — an empty result means it did not take
   ```

3. **`autoremove --purge` after removing old branches can take the surviving one with it.** Removing the 550/570 metapackages orphans shared libraries that 580 still needs; autoremove then sweeps them, leaving the machine with *no* driver at all:

   ```
   Command 'nvidia-smi' not found
   cat: /proc/driver/nvidia/version: No such file or directory
   ```

**Recovery from a fully removed driver** (requires network access):

```bash
sudo apt purge -y '*nvidia*'
sudo apt autoremove -y
sudo apt update
sudo apt install -y nvidia-driver-580-server nvidia-utils-580-server
sudo apt-mark hold $(dpkg -l | grep -E "nvidia|libnvidia" | awk '{print $2}')
sudo systemctl disable --now unattended-upgrades
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer
sudo reboot
```

**Prevention** — on any GPU host that matters, do all three: keep exactly one driver branch, hold every nvidia package (not just the metapackage), and disable both `unattended-upgrades` **and** the `apt-daily-upgrade.timer`. An air-gapped host is immune day to day, but every maintenance window with temporary connectivity re-exposes it.

---

## Frontend and networking

### Blank page with a broken-image icon; JS chunks return 503

**Symptom** — nginx logs fill with:
```
[error] limiting connections by zone "conn"
"GET /_app/immutable/chunks/Xyz.js.map HTTP/2.0" 503
```

**Cause** — connection rate limiting sized for API traffic, not for a single-page application. One page load of a SvelteKit frontend opens 50+ concurrent requests for JS chunks. With `limit_conn conn 20`, the first 20 succeed and the rest are refused — including `/api/config`, so the app never initializes.

**Fix** — size the limits for SPA behavior:

```nginx
limit_req_zone  $binary_remote_addr  zone=api:10m  rate=100r/s;
limit_conn_zone $binary_remote_addr  zone=conn:10m;

limit_req  zone=api burst=200 nodelay;
limit_conn conn 100;
```

Then clear the browser cache — the 503 responses get cached and survive a plain refresh.

---

### Streaming responses arrive in one chunk after a long delay

**Cause** — default nginx proxy buffering defeats server-sent events.

**Fix** — non-negotiable for any LLM reverse proxy:

```nginx
proxy_buffering off;
proxy_cache off;
proxy_request_buffering off;
```

---

### Long responses fail mid-stream; "continue" restarts from scratch

**Cause** — timeouts sized for ordinary web traffic. A reasoning model on a long prompt can generate for several minutes.

**Fix** — raise both layers together; the shorter one wins:

```yaml
# Open WebUI
AIOHTTP_CLIENT_TIMEOUT: "1200"
```

```nginx
proxy_send_timeout 1200s;
proxy_read_timeout 1200s;
```

---

### Open WebUI crashes on startup in an air-gapped network

**Symptom**
```
httpx.ConnectError: [Errno -3] Temporary failure in name resolution
huggingface_hub.errors.LocalEntryNotFoundError
  File "/app/backend/open_webui/retrieval/utils.py", in get_model_path
```

**Cause** — Open WebUI attempts to download a default RAG embedding model at startup. With no DNS, the failure propagates and the web server never binds.

**Fix (RAG not needed)** — disable the retrieval stack:

```yaml
HF_HUB_OFFLINE: "1"
TRANSFORMERS_OFFLINE: "1"
RAG_EMBEDDING_ENGINE: "ollama"
RAG_EMBEDDING_MODEL: ""
ENABLE_RAG_HYBRID_SEARCH: "false"
ENABLE_RAG_WEB_SEARCH: "false"
```

**Fix (RAG wanted)** — pre-download the model and point at an absolute in-container path:

```yaml
environment:
  RAG_EMBEDDING_ENGINE: ""     # empty = built-in sentence-transformers
  RAG_EMBEDDING_MODEL: "/app/backend/data/models/all-MiniLM-L6-v2"
  RAG_EMBEDDING_MODEL_AUTO_UPDATE: "false"
volumes:
  - /opt/ai/models/all-MiniLM-L6-v2:/app/backend/data/models/all-MiniLM-L6-v2:ro
```

Success looks like `BertModel LOAD REPORT from: /app/backend/data/models/all-MiniLM-L6-v2` in the logs. A trailing `embeddings.position_ids | UNEXPECTED` warning is benign.

**What does not work** — mounting into `$SENTENCE_TRANSFORMERS_HOME` under the `sentence-transformers_<name>` convention. Open WebUI calls `huggingface_hub.snapshot_download()` first, which looks for the `models--org--name/snapshots/<hash>/` layout and fails before the local path is ever consulted. An absolute path bypasses that lookup entirely.

---

### Settings changes in `docker-compose.yml` have no effect

**Cause** — Open WebUI's PersistentConfig: many environment variables are read once on first boot and then persisted to the database. `ENABLE_LDAP`, `ENABLE_SIGNUP`, `DEFAULT_USER_ROLE`, and model connection settings all behave this way.

**Fix** — change them in the admin UI, or reset state:

```bash
docker compose down
sudo rm -rf /opt/ai/data/pg/* /opt/ai/data/owui/*
docker compose up -d
```

The second option destroys all accounts and conversation history. The first registered account after reset becomes admin.

**Note** — `docker compose down` must complete before deleting the data directories. Deleting while PostgreSQL is running lets the process immediately recreate them, and the reset silently fails.

---

### Compose refuses to remove a network: "Resource is still in use"

**Cause** — an orphaned container from a previous compose file version is still attached. Removing a service from `docker-compose.yml` does not remove its running container.

**Fix**

```bash
docker compose down --remove-orphans
docker network rm compose_backend
```

---

## Serving a no-thinking variant

Reasoning models default to emitting a thinking block, which costs latency and KV cache. Since `chat_template_kwargs` is per-request rather than a server flag, the split has to happen at the frontend.

In Open WebUI, create a Filter function:

```python
"""
title: Disable Thinking
description: Suppress the reasoning block for faster responses
"""
from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: dict = None) -> dict:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        return body
```

Then register a second model entry with the same base model and this filter attached. Users pick between the reasoning variant and the fast variant from the model dropdown — one served backend, two behavioral profiles, no extra VRAM.

---

## Storage

### Docker daemon root on the wrong volume

**Symptom** — `/` fills after pulling images.

**Cause** — Docker was installed before the data-tier LV was mounted at `/var/lib/docker`.

**Fix**

```bash
sudo systemctl stop docker docker.socket
sudo mv /var/lib/docker /var/lib/docker.bak
sudo mkdir /var/lib/docker
sudo mount /var/lib/docker          # LV must already be in /etc/fstab
sudo rsync -aHAX /var/lib/docker.bak/ /var/lib/docker/
sudo systemctl start docker
sudo rm -rf /var/lib/docker.bak
```

Mount the LV *before* installing Docker and this never happens. `scripts/03-nvidia-docker.sh` asserts the mount point exists before proceeding.

---

### RAID 10 on raw devices vs partitions

Both work. Raw-device RAID (`mdadm --create /dev/md0 ... /dev/sdb /dev/sdc`) is fewer moving parts; partition-based RAID is friendlier if the disks may later host other volumes.

Disk replacement on raw-device RAID skips partition table recreation:

```bash
sudo mdadm /dev/md0 --fail /dev/sdd
sudo mdadm /dev/md0 --remove /dev/sdd
# physical swap
sudo mdadm /dev/md0 --add /dev/sdd
watch cat /proc/mdstat
```

Resync on SAS SSD RAID 10 completes in roughly 60–90 minutes.

---

## Turing-specific behavior (informational, not errors)

These appear on every startup with sm_75 hardware and are correct fallback behavior:

```
Cannot use FA version 2 ... FA2 is only supported on devices with compute capability >= 8
Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']
SymmMemCommunicator: Device capability 7.5 not supported, communicator is not available
```

Flash Attention 2 requires Ampere or newer; vLLM routes to FlashInfer automatically. Symmetric memory communication is an Ampere+ optimization; its absence costs a little tensor-parallel bandwidth and nothing else.

Also expected on hybrid-attention models:

```
Using Triton/FLA GDN prefill kernel
UserWarning: Input tensor shape suggests potential format mismatch: seq_len (16) < num_heads (24)
```

The warning originates from small warmup batches and does not affect inference.

---

## Ecosystem drift

### `huggingface-cli` no longer works

```
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
```

The CLI was renamed in late-2025 `huggingface_hub` releases. Subcommands are unchanged:

```bash
hf download <repo> --local-dir <path> --max-workers 8
```

Install `hf_transfer` and export `HF_HUB_ENABLE_HF_TRANSFER=1` for substantially faster downloads.

---

### netplan apply drops the SSH session

Expected — the IP changed. Use `netplan try` for its 120-second auto-rollback, or have iDRAC/console access ready. For air-gap cutover, always work from the console.
