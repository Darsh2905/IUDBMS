#!/bin/sh
# Distribution cloud images for the real-kernel testbed (aarch64; ~2 GB in all).
# Downloads into testbed/images/ (git-ignored), resumes partial files, and retries
# a stalled transfer instead of hanging on it.
#
#   sh testbed/fetch_images.sh
set -u
cd "$(dirname "$0")/images" 2>/dev/null || { mkdir -p "$(dirname "$0")/images" && cd "$(dirname "$0")/images"; }
dl() {
  [ -s "$2" ] && echo "have $2" && return
  until curl -sSL --retry 5 --speed-limit 20000 --speed-time 30 -C - -o "$2.part" "$1"; do sleep 3; done
  mv "$2.part" "$2" && echo "done $2"
}
dl https://cloud-images.ubuntu.com/releases/noble/release/ubuntu-24.04-server-cloudimg-arm64.img ubuntu-24.04.qcow2 &
dl https://cloud.debian.org/images/cloud/trixie/latest/debian-13-generic-arm64.qcow2 debian-13.qcow2 &
dl https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Cloud/aarch64/images/Fedora-Cloud-Base-Generic-44-1.7.aarch64.qcow2 fedora-44.qcow2 &
dl https://repo.almalinux.org/almalinux/9/cloud/aarch64/images/AlmaLinux-9-GenericCloud-latest.aarch64.qcow2 alma-9.qcow2 &
wait
