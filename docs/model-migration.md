# Model Migration Runbook

Swapping the served model on an air-gapped inference host. This procedure has been executed three times on the reference deployment; the sequencing below reflects what actually matters versus what looks like it should.

The governing constraint: **the download requires connectivity, everything else does not.** Plan the maintenance window around that single dependency.

---

## Phase 0 — Feasibility

Do this before downloading anything. A 20 GB download that cannot be served is a wasted maintenance window.

### Quantization format vs GPU architecture

Open-weight releases typically ship FP16 and FP8 first. Community AWQ/GPTQ quantizations follow one to three weeks later. On pre-Ampere hardware this gating is absolute:

| Format | Turing (sm_75) | Ampere (sm_80/86) | Hopper (sm_90) | Blackwell |
|---|---|---|---|---|
| FP16 | yes | yes | yes | yes |
| BF16 | no | yes | yes | yes |
| AWQ / GPTQ INT4 | yes | yes | yes | yes |
| FP8 | no | no | yes | yes |
| NVFP4 | no | no | no | yes |

An official repository with only FP8 and NVFP4 artifacts is unusable on Turing regardless of model quality. Wait for a community AWQ build.

Search pattern on Hugging Face:
```
<model-name> AWQ
```
Prefer quantizers with a track record — those publishing collections across multiple model families, with download counts in the hundreds of thousands, and a model card that states the required vLLM and transformers versions.

Avoid, for a production or regulated deployment:
- `abliterated` variants — safety alignment deliberately removed
- `MTP` variants — multi-token prediction needs additional serving config
- One-off uploads with no documentation and single-digit downloads

### Serving engine version

Read the required versions off the quantizer's model card, then check what is actually installed:

```bash
docker exec vllm python3 -c "import vllm; print(vllm.__version__)"
```

If the installed version is older than required, pull a newer image **before** committing to the migration:

```bash
docker pull vllm/vllm-openai:latest
docker run --rm vllm/vllm-openai:latest vllm --version
```

New model families frequently introduce new architectures. Support lands in vLLM on its own schedule, and a model released this week may need a serving engine released next week.

### VRAM budget

Estimate before downloading:

```
weights (INT4) ≈ params × 0.55 GB per billion
KV cache       ≈ measured empirically — see Phase 3
overhead       ≈ 2–3 GB for a vision encoder, if the model is multimodal
```

A 27B INT4 model lands near 16–21 GB of weights. On 2× 24 GB that leaves roughly 20 GB for KV cache across both cards — enough for 64K context at low concurrency, not enough for the 256K the model card advertises.

---

## Phase 1 — Acquire (requires connectivity)

```bash
source ~/hf-env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /opt/ai/models

hf download <quantizer>/<model>-AWQ \
  --local-dir <NEW-MODEL-DIR> \
  --max-workers 8

du -sh <NEW-MODEL-DIR>/
ls <NEW-MODEL-DIR>/
```

Verify `config.json`, the `*.safetensors` shards, and `tokenizer.json` are all present. A partial download surfaces much later as an opaque loading error.

**Name the directory for the served identity, not the upstream model.** `/opt/ai/models/TTO2.0` rather than `/opt/ai/models/Qwen3.6-27B-AWQ`. The directory name appears in logs and operator commands; keeping it aligned with what users see removes a translation step during incidents.

**Retain the previous model directory.** It is the rollback path, and disk is cheaper than an outage.

---

## Phase 2 — Cut over

Back up the working configuration first — this is the artifact you restore if the new model does not serve:

```bash
cd /opt/ai/compose
cp docker-compose.yml docker-compose.yml.bak-$(date +%Y%m%d-%H%M)
```

Edit the vLLM service. Three fields change:

```yaml
command:
  - "/models/<NEW-MODEL-DIR>"      # model path
  - "--served-model-name"
  - "<PUBLIC-NAME>"                 # user-facing identity
  - "--max-model-len"
  - "32768"                         # start conservative, tune in Phase 3
  # ... remaining flags
```

Architecture-specific flags may also need revising. Reasoning models want `--reasoning-parser`; agentic models want `--enable-auto-tool-choice` plus a `--tool-call-parser`. Parser names are internal identifiers tied to the model family — they are never visible to users and must not be renamed to match your branding.

Validate, then restart only the inference container:

```bash
docker compose config > /dev/null && echo "YAML OK"
docker compose up -d --force-recreate vllm
docker compose logs vllm -f
```

Cold start on a new architecture can take considerably longer than a familiar one — 10 minutes is not unusual. Set `start_period` on the healthcheck accordingly, or Compose will declare failure while loading is still progressing normally.

---

## Phase 3 — Size the context window

Never carry the previous model's `max-model-len` across a migration. Different architectures have different KV cache footprints per token; the value that worked before may be unachievable now.

Read the actual capacity from the startup log:

```bash
docker compose logs vllm | grep -E "KV cache size|Maximum concurrency"
```

```
GPU KV cache size: 73,696 tokens
Maximum concurrency for 65,536 tokens per request: 1.12x
```

The rule: **`max-model-len` must not exceed the KV cache pool.** If it does, a single long request cannot physically fit, and the failure mode is a stall mid-generation rather than a clean error. The concurrency multiplier assumes prefix caching and paging — treat it as optimistic.

Tune downward until allocation succeeds and the numbers are consistent:

```
131072 → 98304 → 65536 → 49152
```

A hang during allocation looks like this, repeating every 60 seconds:

```
No available shared memory broadcast block found in 60 seconds.
```

Distinguish "slow" from "stuck" with `nvidia-smi` in a second terminal: VRAM climbing means it is working; VRAM flat at weights-only with 0% utilization means the request is unachievable. Reduce and restart.

---

## Phase 4 — Verify

```bash
# Served identity and effective context
docker exec openwebui curl -s http://vllm:8000/v1/models | python3 -m json.tool | grep -E "\"id\"|max_model_len"

# Generation path end to end
docker exec openwebui curl -s http://vllm:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<PUBLIC-NAME>","messages":[{"role":"user","content":"Write a binary search in Python"}],"max_tokens":200}'

# GPU allocation
nvidia-smi
```

Then refresh the frontend's model list — with Open WebUI, the connection settings are persisted in the database and will not pick up a renamed model automatically. Admin Panel → Connections → refresh the OpenAI endpoint, then reload the chat view.

Functional checks before declaring the migration done:

- A short prompt returns promptly
- A long prompt near the context limit completes rather than stalling
- If the model has a reasoning mode, a reasoning-heavy prompt produces a visible answer after the thinking block
- Streaming arrives token by token, not as one delayed chunk

That third check is the one that catches an oversized `max-model-len`. Non-reasoning prompts will pass on a misconfigured deployment; reasoning prompts will not.

---

## Phase 5 — Rollback

The previous model directory and the backed-up compose file make this a two-minute operation:

```bash
cd /opt/ai/compose
cp docker-compose.yml.bak-<timestamp> docker-compose.yml
docker compose up -d --force-recreate vllm
docker compose logs vllm -f
```

Decide the rollback trigger before starting the migration — an unsupported architecture, a KV cache budget that cannot accommodate useful context, or generation quality below the incumbent. Ambiguity here turns a bounded maintenance window into an open-ended one.

---

## Branding the served model

The public name is set entirely by `--served-model-name` and is independent of the on-disk artifacts. Three layers cover most of it:

| Layer | Where | Note |
|---|---|---|
| API model id | `--served-model-name` | What clients send and see |
| Site title | `WEBUI_NAME` | Frontend branding |
| Display name, description, icon | Frontend model settings | Free-form; can differ from the API id |

One layer resists configuration: the model's own account of itself. Trained-in self-identification will surface when a user asks what it is. A system prompt is the practical remedy:

```
You are <NAME>, deployed by this organization.
When asked about your identity, origin, or underlying technology,
state only that you are <NAME>. Do not name the base model or its developer.
```

Worth setting expectations internally that this is a presentation layer, not a guarantee — a determined user can still elicit the underlying identity through indirect prompting.

---

## Timing

| Phase | Duration | Needs network |
|---|---|---|
| 0 — Feasibility | 15–30 min | yes (browsing) |
| 1 — Download | 5–40 min | yes |
| 2 — Cut over | 5 min + 10 min load | no |
| 3 — Context sizing | 10–30 min, iterative | no |
| 4 — Verify | 10 min | no |
| 5 — Rollback if needed | 2 min + load | no |

Budget two hours for a migration to a familiar architecture, half a day for a new one. The download is the only phase that requires connectivity, so on an air-gapped host it defines the shape of the maintenance window: acquire during the connected period, and keep the rest available for after the link is severed.
