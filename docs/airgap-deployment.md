# Air-Gap Deployment Runbook

The host transitions through two distinct phases during initial deployment. After phase 2, **no outbound network traffic should ever leave the host**. Anything outbound is a configuration error or a security incident.

This runbook is the canonical procedure for moving the host from a temporarily online build environment to its permanent air-gapped production environment.

---

## Phase 1 — Online Staging

The host is connected to a network with internet access for the duration of asset acquisition. Direct internet access is **temporary** and revoked once Phase 2 begins.

### What happens online

1. Ubuntu 24.04.3 LTS install over PXE / USB
2. Apt updates and toolchain install (`scripts/01-system-prep.sh`)
3. Storage provisioning (`scripts/02-raid-setup.sh`)
4. NVIDIA driver, Docker, Container Toolkit (`scripts/03-nvidia-docker.sh`)
5. Docker images pulled from Docker Hub / GHCR
6. Model weights downloaded from Hugging Face (`scripts/04-download-models.sh`)
7. Initial Compose stack brought up; basic smoke tests pass

### Pre-cutover checklist

Before disconnecting from the internet, **verify all of the following**:

- [ ] `nvidia-smi` lists every GPU
- [ ] `cat /proc/mdstat` shows RAID 10 active with all members `U`
- [ ] `df -h` confirms `/opt/ai/models`, `/opt/ai/data`, `/var/lib/docker`, `/opt/ai/backup` are all on LVM volumes
- [ ] `docker compose ps` shows every service `healthy`
- [ ] `curl http://127.0.0.1:8000/v1/models` returns the served model
- [ ] HTTPS reachability tested from a separate machine on the same subnet
- [ ] Login via local admin account works
- [ ] One full chat round-trip completes successfully

### Asset snapshot

Save every external dependency to a known location on the data tier so the host can survive without internet access:

```bash
# Docker images
mkdir -p /opt/ai/backup/images
cd /opt/ai/backup/images
for img in vllm/vllm-openai:latest \
           ghcr.io/open-webui/open-webui:main \
           postgres:16 redis:7-alpine nginx:stable; do
  name=$(echo "$img" | tr '/:' '__')
  docker save "$img" | gzip > "${name}.tar.gz"
done

# Python wheels
mkdir -p /opt/ai/backup/wheels
source ~/hf-env/bin/activate
pip download -d /opt/ai/backup/wheels \
  huggingface_hub hf_transfer requests openai

# Configuration files
sudo tar czf /opt/ai/backup/system-config.tar.gz \
  /etc/fstab \
  /etc/mdadm/mdadm.conf \
  /etc/docker/daemon.json \
  /etc/netplan/ \
  ~/github-repo/enterprise-airgapped-llm/

# Hash everything for integrity
cd /opt/ai/backup
find images wheels -type f -exec sha256sum {} \; > MANIFEST.sha256

# Optional: copy to external media for off-host retention
# rsync -aHAX /opt/ai/backup/ /mnt/external-drive/
```

These artifacts are what enables disaster recovery and re-deployment without re-establishing internet access.

---

## Phase 2 — Cutover

This phase **must** be performed with physical or out-of-band (iDRAC) console access available. SSH sessions will drop when the network changes.

### 1. Migrate network configuration

Edit `/etc/netplan/*.yaml` to reflect the air-gapped VLAN. Example:

```yaml
network:
  version: 2
  ethernets:
    eno1:
      dhcp4: false
      addresses:
        - 192.168.1.186/24
      routes:
        - to: default
          via: 192.168.1.254
      nameservers:
        addresses:
          - 192.168.1.254       # internal DNS
```

```bash
sudo chmod 600 /etc/netplan/*.yaml
sudo netplan try         # 120-second auto-rollback window
# If happy:
sudo netplan apply
```

### 2. Repoint NTP to internal time source

Without correct time, AD/Kerberos authentication fails silently with cryptic errors.

```bash
sudo nano /etc/systemd/timesyncd.conf
# [Time]
# NTP=ntp.corp.local 192.168.1.x

sudo systemctl restart systemd-timesyncd
timedatectl status
# Verify: "System clock synchronized: yes"
```

### 3. Disconnect from the staging network

Either:
- Physically unplug the staging cable and connect the production cable, **or**
- Switch the port's VLAN at the access switch

### 4. Validate isolation

From the host:

```bash
# Should fail — internet is gone
curl -m 5 https://huggingface.co  || echo "OK, blocked as expected"
curl -m 5 https://google.com      || echo "OK, blocked as expected"

# Should succeed — internal services reachable
ping -c2 192.168.1.254
nslookup dc01.corp.local
```

### 5. Restart the stack on the new network

```bash
cd /opt/ai/compose
docker compose down
docker compose up -d
docker compose logs -f vllm
# Wait for "Uvicorn running"
```

### 6. End-to-end smoke test

From an internal user workstation:

- Browse to `https://<host>/`
- Log in
- Send a chat message
- Verify the response streams correctly

If anything fails, the most likely cause is DNS or NTP misconfiguration, not the inference stack.

---

## Phase 3 — Switching to AD/LDAPS Authentication

After local admin login is working in the air-gapped environment, swap authentication to Active Directory.

### Prerequisites

- A service account in AD (e.g. `svc_openwebui`), password set to never expire
- A security group (e.g. `AI-Users`) controlling who can log in
- Internal CA certificate (for LDAPS server validation) imported on the host
- Firewall permits the host to reach the DC on TCP 636

### Configuration

Edit the `openwebui:` environment block in `docker-compose.yml`:

```yaml
ENABLE_LDAP: "true"
ENABLE_SIGNUP: "false"               # disable local registration
LDAP_SERVER_LABEL: "Corporate AD"
LDAP_SERVER_HOST: "dc01.corp.local"
LDAP_SERVER_PORT: "636"
LDAP_USE_TLS: "true"
LDAP_VALIDATE_CERT: "true"
LDAP_APP_DN: "CN=svc_openwebui,OU=ServiceAccounts,DC=corp,DC=local"
LDAP_APP_PASSWORD: "${LDAP_BIND_PASSWORD}"
LDAP_SEARCH_BASE: "DC=corp,DC=local"
LDAP_ATTRIBUTE_FOR_USERNAME: "sAMAccountName"
LDAP_ATTRIBUTE_FOR_MAIL: "mail"
LDAP_SEARCH_FILTER: "(memberOf=CN=AI-Users,OU=Groups,DC=corp,DC=local)"
```

Add `LDAP_BIND_PASSWORD` to your `.env` file. Restart the stack and verify a domain user can log in. Existing local admin accounts continue to work as break-glass credentials.

---

## Long-Term Operations

### Quarterly model / image refresh

Open WebUI and vLLM both ship updates every 1–3 months. Refresh them on the staging host (or any internet-connected machine), `docker save` the new image, transfer to the air-gapped host through approved removable media, then `docker load` and restart Compose.

### Backups

```bash
# /opt/ai/backup.sh
#!/bin/bash
BACKUP=/opt/ai/backup/daily
DATE=$(date +%F)
mkdir -p "$BACKUP"

docker exec postgres pg_dump -U openwebui openwebui | gzip > "$BACKUP/db_$DATE.sql.gz"
tar czf "$BACKUP/owui_$DATE.tar.gz" /opt/ai/data/owui
find "$BACKUP" -name "*.gz" -mtime +30 -delete
```

```bash
# Cron — every night at 2am
echo "0 2 * * * root /opt/ai/backup.sh" | sudo tee /etc/cron.d/ai-backup
```

For true disaster recovery, sync `/opt/ai/backup/` to a second host or NAS on a separate VLAN.

### Health monitoring

```bash
# Weekly RAID scrub (already enabled by default in mdadm)
sudo systemctl status mdcheck_start.timer mdcheck_continue.timer

# Daily SMART check
sudo smartctl -H /dev/sd[b-g]

# GPU health
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv
```

Wire the relevant alerts into the corporate SMTP relay so failures are visible. Air-gapped does not mean blind.
