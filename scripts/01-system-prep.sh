#!/bin/bash
# scripts/01-system-prep.sh
#
# Phase 1 — Base system preparation on a fresh Ubuntu 24.04.3 LTS install.
# Updates the package index, installs the toolchain required by subsequent
# scripts, and applies a few production-grade defaults.
#
# Run on a freshly imaged box, ideally before any other configuration.
# Requires sudo.

set -euo pipefail

echo "==> [1/5] Refreshing package index"
sudo apt update
sudo apt -y upgrade

echo "==> [2/5] Installing core toolchain"
sudo apt install -y \
    build-essential \
    curl wget git vim nano \
    htop tmux \
    pciutils smartmontools lm-sensors \
    mdadm lvm2 parted \
    python3-pip python3-venv \
    ca-certificates gnupg lsb-release \
    openssl \
    jq

echo "==> [3/5] Setting timezone (Asia/Taipei — adjust as needed)"
sudo timedatectl set-timezone Asia/Taipei
timedatectl

echo "==> [4/5] Disabling unattended upgrades on the inference server"
# Production air-gapped boxes should not auto-update; updates must be tested
# in a staging environment and rolled out manually.
sudo systemctl disable --now unattended-upgrades 2>/dev/null || true

echo "==> [5/5] Hardware sanity check"
echo "--- CPU ---"
lscpu | grep -E "Model name|Socket|Core"
echo "--- Memory ---"
free -h
echo "--- NVIDIA devices on PCIe ---"
lspci | grep -i nvidia || echo "No NVIDIA devices detected (will be visible after driver install)"
echo "--- Block devices ---"
lsblk

cat <<'EOM'

==============================================================================
System preparation complete.

Next steps:
  - 02-raid-setup.sh     : provision mdadm RAID 10 + LVM on the data disks
  - 03-nvidia-docker.sh  : install NVIDIA driver, Docker, Container Toolkit
  - 04-download-models.sh: pull model weights from Hugging Face
  - 05-deploy.sh         : start the container stack

WARNING: 02-raid-setup.sh is destructive. Only run it after you have verified
which block devices are the data disks (typically /dev/sd[b-g]).
==============================================================================
EOM
