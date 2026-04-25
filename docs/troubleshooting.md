# Troubleshooting

Real issues encountered during the production deployment, with the resolution that worked. These are the problems you'll actually hit, not the ones in the official tutorials.

---

## vLLM: `unrecognized arguments: serve`

**Symptom:**
```
vllm: error: unrecognized arguments: serve
```

**Cause:** The official `vllm/vllm-openai` Docker image already declares `ENTRYPOINT ["vllm", "serve"]`. Specifying `serve` again in the Compose `command:` produces `vllm serve serve ...`, which the argparse layer rejects.

**Fix:** In `docker-compose.yml`, the `command:` array should start with the model path (positional argument), **not** with `serve`:

```yaml
command:
  - "/models/Qwen3-Coder-30B-A3B-Instruct-AWQ"
  - "--served-model-name"
  - "qwen3-coder"
  - "--tensor-parallel-size"
  - "2"
  # ...
```

---

## vLLM: Pydantic ValidationError on quantization

**Symptom:**
```
ValidationError: 1 validation error for ModelConfig
Value error, Quantization method specified in the model config (compressed-tensors)
does not match the quantization method specified in the `quantization` argument (awq)
```

**Cause:** Modern AWQ checkpoints (notably from `llm-compressor`) declare `compressed-tensors` in `config.json`. Specifying `--quantization awq` on the CLI overrides this and conflicts.

**Fix:** Remove `--quantization` from the CLI entirely. Let vLLM auto-detect from the model config. This is now the recommended pattern (vLLM 0.10+).

---

## vLLM: `--disable-log-requests` no longer recognized

**Symptom:**
```
vllm: error: unrecognized arguments: --disable-log-requests
```

**Cause:** Removed in newer vLLM releases.

**Fix:** Remove the flag. To reduce log noise, use `--uvicorn-log-level warning` instead.

---

## `huggingface-cli` deprecated

**Symptom:**
```
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
```

**Cause:** The `huggingface_hub` package renamed the CLI from `huggingface-cli` to `hf` in late-2025 releases.

**Fix:** Replace all `huggingface-cli` invocations with `hf`. The subcommands (`download`, `upload`, etc.) are unchanged.

```bash
# Old
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ --local-dir ./model

# New
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ --local-dir ./model
```

---

## NVIDIA driver: GPUs not visible after install

**Symptom:** `nvidia-smi` shows fewer GPUs than physically present, or "No devices were found".

**Causes (in priority order):**
1. **Above 4G Decoding** disabled in BIOS — required for any GPU with > 4 GB VRAM
2. **Memory Mapped I/O Base** set too low — Dell defaults to 56 TB; large multi-GPU systems may need 12 TB
3. **Secure Boot enabled** — proprietary NVIDIA modules cannot load
4. `nouveau` kernel module not blacklisted

**Fix:** Validate BIOS first, then:
```bash
echo -e "blacklist nouveau\noptions nouveau modeset=0" | \
  sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot
```

---

## Docker daemon root on wrong volume

**Symptom:** `/` partition fills up after pulling images; `/var/lib/docker` consumes hundreds of GB.

**Cause:** Docker was installed before the data-tier LV was mounted at `/var/lib/docker`.

**Fix:**
```bash
sudo systemctl stop docker docker.socket
sudo mv /var/lib/docker /var/lib/docker.bak
sudo mkdir /var/lib/docker
# ensure lv_docker is in /etc/fstab
sudo mount /var/lib/docker
sudo rsync -aHAX /var/lib/docker.bak/ /var/lib/docker/
sudo systemctl start docker
sudo rm -rf /var/lib/docker.bak
```

The cleaner approach is to mount the LV before Docker installation. The `scripts/03-nvidia-docker.sh` enforces this ordering.

---

## RAID 10 created on raw devices vs partitions

Both approaches work. Raw-device RAID (`mdadm --create /dev/md0 ... /dev/sdb /dev/sdc ...`) is slightly cleaner; partition-based RAID (`/dev/sdb1 /dev/sdc1 ...`) is friendlier when the same disks may host other volumes later.

For the all-data-disk scenario in this stack, raw-device RAID was chosen for simplicity. The replacement procedure differs slightly:

**Replacing a failed disk in raw-device RAID:**
```bash
sudo mdadm /dev/md0 --fail /dev/sdd
sudo mdadm /dev/md0 --remove /dev/sdd
# physical replacement
sudo mdadm /dev/md0 --add /dev/sdd
watch cat /proc/mdstat
```

No partition table re-creation needed. Resync on SAS SSD RAID 10 typically completes within 60–90 minutes.

---

## netplan apply disconnects SSH session

**Symptom:** SSH session freezes when applying network changes.

**Cause:** Expected — the IP changed.

**Mitigation:** Use `netplan try` (auto-rolls back after 120 s if not confirmed), or schedule the change for an in-person console session. For air-gap migrations, always have iDRAC or physical console access ready.

---

## Streaming responses appear "stuck" in browser

**Symptom:** The model response appears in one large chunk after a long delay, instead of streaming token-by-token.

**Cause:** Default Nginx behavior buffers proxied responses, defeating server-sent events.

**Fix:** In the relevant `location` block of `nginx.conf`:
```nginx
proxy_buffering off;
proxy_cache off;
proxy_request_buffering off;
```

This is non-negotiable for any LLM-serving reverse proxy.

---

## Open WebUI shows "Connection failed" despite vLLM healthy

**Symptom:** vLLM `/health` returns 200, but Open WebUI fails to load models.

**Cause:** Open WebUI cannot resolve `vllm:8000` because the services aren't on the same Docker network, or it tried to connect before vLLM finished cold-starting (model load takes ~30s, vLLM healthcheck `start_period` must reflect this).

**Fix:** In `docker-compose.yml`:
- Ensure both services share at least one network (`networks: [backend]`)
- Set vLLM healthcheck `start_period: 600s` (generous margin for cold start)
- Use `depends_on: { vllm: { condition: service_healthy } }` on Open WebUI
