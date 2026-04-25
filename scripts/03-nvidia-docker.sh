#!/bin/bash
# scripts/03-nvidia-docker.sh
#
# Phase 3 — Install NVIDIA proprietary driver, Docker Engine, and the
# NVIDIA Container Toolkit so containers can access the GPUs.
#
# Driver branch: 550-server (Turing-compatible, long-lived branch).
# CUDA bundled with the driver is sufficient for vLLM containers; no system
# CUDA toolkit needs to be installed on the host.
#
# Run AFTER 02-raid-setup.sh — this script assumes /var/lib/docker is already
# mounted on a dedicated LV.

set -euo pipefail

# -----------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------
echo "==> Verifying /var/lib/docker is mounted on a dedicated volume"
if ! mountpoint -q /var/lib/docker; then
  echo "ERROR: /var/lib/docker is not a mount point. Run 02-raid-setup.sh first."
  exit 1
fi

# -----------------------------------------------------------------
# 1. Disable nouveau and install proprietary NVIDIA driver
# -----------------------------------------------------------------
echo "==> [1/4] Blacklisting nouveau"
sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
sudo update-initramfs -u

echo "==> [2/4] Installing NVIDIA driver 550-server"
# 550-server is the recommended long-lived branch for Turing-class GPUs.
# Newer branches (570+) also work; pick whichever your distribution provides.
sudo apt update
sudo apt install -y nvidia-driver-550-server nvidia-utils-550-server

# -----------------------------------------------------------------
# 2. Docker Engine
# -----------------------------------------------------------------
echo "==> [3/4] Installing Docker Engine via official convenience script"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi

# Allow the invoking user to run docker without sudo on next login
sudo usermod -aG docker "$USER"

# -----------------------------------------------------------------
# 3. NVIDIA Container Toolkit
# -----------------------------------------------------------------
echo "==> [4/4] Installing NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

cat <<'EOM'

==============================================================================
NVIDIA driver + Docker + Container Toolkit installation complete.

A reboot is REQUIRED to load the proprietary NVIDIA kernel modules cleanly.
After reboot, validate with:

  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

Both commands should list every GPU in the system. If the second command
fails but nvidia-smi works, the Container Toolkit configuration didn't take
effect — restart the docker daemon and retry.

Reboot now? (y/N)
==============================================================================
EOM

read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
  sudo reboot
fi
