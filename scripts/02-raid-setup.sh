#!/bin/bash
# scripts/02-raid-setup.sh
#
# Phase 2 — Provision mdadm RAID 10 across 6 SAS SSDs, then carve LVM volumes.
#
# Reference layout:
#   6× 750 GB SAS SSD on Dell HBA330 (IT mode) → /dev/md0 (RAID 10, ~2.25 TB)
#   /dev/md0 → LVM volume group "vg_data"
#       lv_models   600 GB → /opt/ai/models
#       lv_owui     400 GB → /opt/ai/data
#       lv_docker   300 GB → /var/lib/docker  (mounted BEFORE Docker install)
#       lv_backup   800 GB → /opt/ai/backup
#
# The OS lives on a separate Dell BOSS-S1 hardware RAID 1 (not touched here).
#
# WARNING: This script wipes /dev/sd[b-g] without further prompts after
# confirmation. Verify the disk list before typing "yes".

set -euo pipefail

DISKS=(b c d e f g)
RAID_DEV=/dev/md0
VG_NAME=vg_data

echo "==> Block device inventory"
lsblk -o NAME,SIZE,MODEL,TYPE,MOUNTPOINTS

echo
echo "About to:"
echo "  1. Wipe filesystem signatures on: ${DISKS[*]/#/\/dev\/sd}"
echo "  2. Build a RAID 10 array on $RAID_DEV"
echo "  3. Create LVM volume group $VG_NAME and 4 logical volumes"
echo "  4. Format and mount under /opt/ai/* and /var/lib/docker"
echo
read -p "Continue? Type 'yes' to proceed: " confirm
[[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 1; }

echo "==> [1/6] Wiping member disks"
for d in "${DISKS[@]}"; do
  sudo wipefs -a "/dev/sd${d}"
  sudo sgdisk --zap-all "/dev/sd${d}" 2>/dev/null || true
done

echo "==> [2/6] Creating RAID 10 (n2 layout, 256 KiB chunk)"
sudo mdadm --create "$RAID_DEV" \
  --level=10 \
  --raid-devices="${#DISKS[@]}" \
  --layout=n2 \
  --chunk=256 \
  --name=data \
  $(for d in "${DISKS[@]}"; do echo -n "/dev/sd${d} "; done)

echo "==> [3/6] Persisting mdadm config and rebuilding initramfs"
sudo mdadm --detail --scan | sudo tee /etc/mdadm/mdadm.conf
sudo update-initramfs -u

echo "==> [4/6] Creating LVM volume group and logical volumes"
sudo pvcreate "$RAID_DEV"
sudo vgcreate "$VG_NAME" "$RAID_DEV"

sudo lvcreate -L 600G -n lv_models  "$VG_NAME"
sudo lvcreate -L 400G -n lv_owui    "$VG_NAME"
sudo lvcreate -L 300G -n lv_docker  "$VG_NAME"
sudo lvcreate -L 800G -n lv_backup  "$VG_NAME"
# ~150 GB intentionally left unallocated for future expansion / snapshots

echo "==> [5/6] Formatting (ext4 with reduced reserved blocks for SSDs)"
sudo mkfs.ext4 -m 1 -L models  "/dev/${VG_NAME}/lv_models"
sudo mkfs.ext4 -m 1 -L owui    "/dev/${VG_NAME}/lv_owui"
sudo mkfs.ext4 -m 0 -L docker  "/dev/${VG_NAME}/lv_docker"
sudo mkfs.ext4 -m 1 -L backup  "/dev/${VG_NAME}/lv_backup"

echo "==> [6/6] Creating mount points and writing fstab"
sudo mkdir -p /opt/ai/{models,data,backup}
sudo mkdir -p /var/lib/docker

# Remove any stale vg_data entries to keep this script idempotent
sudo sed -i '/vg_data/d' /etc/fstab
sudo tee -a /etc/fstab > /dev/null <<FSTAB

# Air-gapped LLM stack — data tier (mdadm RAID 10 + LVM)
/dev/${VG_NAME}/lv_models   /opt/ai/models    ext4  defaults,noatime  0 2
/dev/${VG_NAME}/lv_owui     /opt/ai/data      ext4  defaults,noatime  0 2
/dev/${VG_NAME}/lv_docker   /var/lib/docker   ext4  defaults,noatime  0 2
/dev/${VG_NAME}/lv_backup   /opt/ai/backup    ext4  defaults,noatime  0 2
FSTAB

sudo mount -a

echo
echo "==> Verification"
cat /proc/mdstat
echo
sudo vgs
sudo lvs
echo
df -h | grep -E "ai|docker"

cat <<'EOM'

==============================================================================
RAID 10 + LVM provisioning complete.

The /var/lib/docker mount must exist BEFORE Docker is installed, otherwise the
daemon will populate the OS partition. Run 03-nvidia-docker.sh next.
==============================================================================
EOM
