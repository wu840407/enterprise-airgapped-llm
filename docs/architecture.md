# Architecture

This document describes the full architecture of the air-gapped LLM stack — what each component does, how they communicate, and why the layout is designed the way it is.

---

## High-Level Component View

```mermaid
flowchart TB
    subgraph Client["Internal Network Users"]
        U1[Browser]
        U2[VS Code + Continue]
        U3[Aider CLI]
    end

    subgraph Host["Dell R740 — Ubuntu 24.04.3 LTS, fully air-gapped"]
        subgraph FrontendNet["frontend network (bridge)"]
            NG[Nginx<br/>TLS, rate-limit, WS upgrade]
            OW[Open WebUI<br/>multi-user chat, RBAC, AD]
        end

        subgraph BackendNet["backend network (internal-only, no egress)"]
            VL[vLLM<br/>OpenAI-compatible API<br/>tensor-parallel × 2]
            PG[(PostgreSQL 16<br/>conversations, audit)]
            RD[(Redis 7<br/>sessions)]
        end

        subgraph Hardware["Physical layer"]
            G0[GPU 0: Quadro RTX 6000 24GB]
            G1[GPU 1: Quadro RTX 6000 24GB]
            ST[(RAID 10 / LVM<br/>6× 750GB SAS SSD)]
        end
    end

    subgraph Corporate["Corporate Infrastructure"]
        AD[(Active Directory<br/>LDAPS)]
        DNS[(Internal DNS / NTP)]
    end

    U1 -->|HTTPS 443| NG
    U2 -->|HTTPS 443| NG
    U3 -->|HTTPS 443| NG
    NG -->|HTTP 8080| OW
    OW -->|HTTP 8000| VL
    OW -->|TCP 5432| PG
    OW -->|TCP 6379| RD
    OW -.->|LDAPS 636| AD
    VL --> G0
    VL --> G1
    VL -.->|model weights| ST
    PG -.->|persistent volume| ST
    Host -.->|time sync| DNS
```

**Key properties:**

- Only Nginx is reachable from outside the host (ports 80/443).
- The `backend` Docker network is declared `internal: true`, so the inference and database tier cannot egress even if compromised.
- Every persistent volume lives on the dedicated data-tier RAID 10, never on the OS partition.
- Active Directory is the only external corporate dependency once deployment is complete.

---

## Storage Architecture

Two completely separate storage tiers, each optimized for its role.

```mermaid
flowchart LR
    subgraph BootTier["Boot Tier — Hardware RAID 1"]
        B1[M.2 SATA SSD 240GB]
        B2[M.2 SATA SSD 240GB]
        BOSS[Dell BOSS-S1<br/>hardware RAID controller]
        B1 --> BOSS
        B2 --> BOSS
        BOSS --> OS[/dev/sda<br/>~240GB virtual disk]
        OS --> EFI[1G EFI]
        OS --> BT[2G /boot]
        OS --> SW[32G swap]
        OS --> RT[~205G /]
    end

    subgraph DataTier["Data Tier — Software RAID 10 + LVM"]
        D1[SAS SSD 750G]
        D2[SAS SSD 750G]
        D3[SAS SSD 750G]
        D4[SAS SSD 750G]
        D5[SAS SSD 750G]
        D6[SAS SSD 750G]
        HBA[HBA330<br/>IT mode pass-through]
        D1 --> HBA
        D2 --> HBA
        D3 --> HBA
        D4 --> HBA
        D5 --> HBA
        D6 --> HBA
        HBA --> MD[/dev/md0<br/>RAID 10, 2.25TB]
        MD --> VG[vg_data<br/>LVM volume group]
        VG --> LV1[lv_models 600G]
        VG --> LV2[lv_owui 400G]
        VG --> LV3[lv_docker 300G]
        VG --> LV4[lv_backup 800G]
    end
```

**Design rationale:**

| Decision | Why |
|---|---|
| Hardware RAID 1 on BOSS for OS | iDRAC monitors out-of-band; replacement requires zero OS intervention |
| Software RAID 10 on HBA330 for data | mdadm is auditable, recoverable, and not tied to firmware lifecycle |
| LVM on top of mdadm | Online expansion, snapshot-based pre-upgrade rollback |
| Separate `lv_docker` | Docker pull bloat cannot fill the OS partition |
| `noatime` mount option | Reduces SSD wear; access time is irrelevant for this workload |
| 5% reserved blocks tuned to 0–1% | Reclaims tens of GB on large LVs without sacrificing fsck headroom |

---

## Inference Path

End-to-end flow of a single chat message:

```mermaid
sequenceDiagram
    autonumber
    participant U as User Browser
    participant NG as Nginx
    participant OW as Open WebUI
    participant PG as PostgreSQL
    participant VL as vLLM
    participant G as GPU 0+1

    U->>NG: HTTPS POST /api/chat (prompt)
    NG->>OW: HTTP /api/chat (proxy_buffering off)
    OW->>PG: INSERT conversation row
    OW->>VL: POST /v1/chat/completions (stream=true)
    VL->>G: tensor-parallel forward pass
    G-->>VL: token logits
    VL-->>OW: SSE token stream
    OW-->>NG: SSE token stream
    NG-->>U: SSE token stream (no buffering)
    OW->>PG: UPDATE conversation with assistant message
```

**Latency notes:**

- Token streaming starts in **< 500 ms** for warm prompts; the first token round-trip is dominated by the prefill phase, not network or proxy.
- `proxy_buffering off` in nginx is **non-negotiable** — without it, tokens arrive in chunks and the UX feels broken.
- Open WebUI persists each turn to PostgreSQL synchronously, providing a complete audit trail without measurable user-visible overhead.

---

## Network Boundary

```mermaid
flowchart TB
    Internet[Public Internet]
    Corp[Corporate VLAN]
    Air[Air-Gapped VLAN]

    Internet -.->|"❌ blocked"| Air
    Corp -->|"only LDAPS, NTP, DNS"| Air

    subgraph Air[Air-Gapped Production VLAN]
        Host[Dell R740 ai-r740]
        Users[Internal users]
        Users -->|HTTPS only| Host
    end

    subgraph Online[Online Staging — used during initial install only]
        Stage[Same hardware<br/>temporary internet access]
    end

    Stage -.->|"physical move<br/>after asset bake-in"| Host
```

The host has two distinct lifecycle phases:

1. **Online staging** — temporary internet access for `apt`, `docker pull`, `hf download`. Asset hashes are recorded.
2. **Air-gapped production** — moved to the isolated VLAN. NTP and DNS pivot to internal services. Any outbound packet is a misconfiguration alert.

---

## Service Dependency Graph

`docker compose` enforces the following healthcheck-gated startup order:

```mermaid
flowchart LR
    R[redis] --> O[openwebui]
    P[postgres] --> O
    V[vllm] --> O
    O --> N[nginx]

    classDef ready fill:#dfd,stroke:#6a6
    classDef gated fill:#ffd,stroke:#aa6
    class R,P,V ready
    class O,N gated
```

- `vllm` is the slowest to become healthy (~30 s cold-start) — its `start_period: 600s` healthcheck window absorbs cold-start variance gracefully.
- `openwebui` will not start until both `vllm` and `postgres` report healthy, eliminating the entire class of "Open WebUI shows 'no models'" race conditions.
- `nginx` starts last, so the moment the user can reach the site, the backend is already serving.

---

## Failure Modes and Recovery

| Failure | Detection | Recovery Path |
|---|---|---|
| Single SAS SSD in RAID 10 dies | mdadm email + iDRAC alert | Hot-swap, `mdadm --add`, ~60 min resync |
| BOSS M.2 SSD dies | iDRAC alert | Hot-swap, hardware controller rebuilds RAID 1 automatically |
| GPU dies | `nvidia-smi` reports DEGRADED | Reduce `--tensor-parallel-size` to 1, fall back to single-GPU model |
| vLLM container crashes | Compose `restart: unless-stopped` | Auto-restart; check logs |
| PostgreSQL data corruption | Healthcheck fail | Restore from latest `pg_dump` in `/opt/ai/backup/daily/` |
| Whole host dies | External monitoring alert | Reinstall from BOSS RAID 1 mirror, restore data tier from backup volume / NAS |
| AD outage | Login fails | Pre-staged break-glass local admin account |

Every failure mode has a documented manual recovery path. There is no implicit assumption of cloud-side restoration.
