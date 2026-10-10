"""
Boot a Linux distribution headless under QEMU with a chosen set of emulated USB
devices, reachable over SSH and controllable over QMP.

Each guest boots a throw-away qcow2 overlay on top of the distribution's cloud
image, so the image is never modified and every run starts from the same disk.
cloud-init (NoCloud seed) creates a ``tb`` user with the testbed's SSH key and
loads ``usbmon``; nothing else is installed unless a workload asks for it.

The devices are QEMU's emulated USB models on an xHCI controller. The guest's
kernel enumerates them and binds its real drivers (usbhid, usb-storage, uas,
snd-usb-audio, cdc_ether, ftdi_sio, ...), so usbmon records genuine kernel USB
traffic. What is emulated is the device side: the firmware, not the stack.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .qmp import QMP

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "testbed" / "images"
KEY = IMAGES / "id_testbed"

#: distribution -> (image file, what it is). Kernels span 5.14 (RHEL 9 line) to
#: current, two libc-glibc package families (deb, rpm), three init/network stacks.
DISTROS = {
    "ubuntu-24.04": ("ubuntu-24.04.qcow2", "Ubuntu 24.04 LTS cloud image"),
    "debian-13":    ("debian-13.qcow2", "Debian 13 (trixie) generic cloud image"),
    "fedora-44":    ("fedora-44.qcow2", "Fedora 44 Cloud Base"),
    "alma-9":       ("alma-9.qcow2", "AlmaLinux 9 GenericCloud (RHEL 9 kernel line)"),
}

#: HID devices are bound to the display (``display=gpu0``) so QMP can address input to
#: each one by id, and are full-speed (``usb_version=1``) like real keyboards and mice,
#: which are polled on 1 ms frames rather than 125 us high-speed microframes.
#: Device catalogue: name -> (QEMU -device args, extra QEMU args placed before it, what
#: it is). ``{d}`` is replaced by the run directory (backing files). Arguments that
#: must FOLLOW the device (a disk on the UAS device's own SCSI bus) are in _AFTER.
DEVICES = {
    "kbd":     ("usb-kbd,id=kbd0,display=gpu0,usb_version=1", [], "HID boot keyboard"),
    "mouse":   ("usb-mouse,id=mouse0,usb_version=1", [], "HID boot mouse (relative)"),
    "tablet":  ("usb-tablet,id=tablet0,display=gpu0,usb_version=1", [], "HID absolute pointer"),
    "wacom":   ("usb-wacom-tablet,id=wacom0", [], "Wacom PenPartner tablet"),
    "storage": ("usb-storage,drive=stick0,id=msd0,removable=on",
                ["-drive", "if=none,id=stick0,format=raw,file={d}/stick0.img"],
                "mass storage, bulk-only transport"),
    "uas":     ("usb-uas,id=uas0",
                ["-drive", "if=none,id=uasd0,format=raw,file={d}/uas0.img"],
                "mass storage, USB Attached SCSI"),
    "audio":   ("usb-audio,audiodev=snd0,id=aud0", ["-audiodev", "none,id=snd0"],
                "USB audio class (isochronous)"),
    "net":     ("usb-net,netdev=un0,id=net0",
                ["-netdev", "user,id=un0,net=10.0.9.0/24,restrict=on"],
                "CDC Ethernet/RNDIS network adapter"),
    # NOT a null chardev: usb-serial attaches only once its backend reports "opened",
    # and the null backend never does, so the device silently never enumerates.
    "serial":  ("usb-serial,chardev=ser0,id=serial0",
                ["-chardev", "file,id=ser0,path={d}/serial.out"],
                "FTDI USB-serial converter"),
    "ccid":    ("usb-ccid,id=ccid0", [], "smart-card reader (no kernel driver)"),
    "hub":     ("usb-hub,id=hub0", [], "USB hub"),
}


_AFTER = {"uas": ["-device", "scsi-hd,bus=uas0.0,scsi-id=0,lun=0,drive=uasd0"]}


def ensure_key() -> Path:
    if not KEY.exists():
        IMAGES.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "iudbms-testbed",
                        "-f", str(KEY)], check=True)
    return KEY


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _firmware() -> tuple[Path, Path]:
    for share in (Path("/opt/homebrew/share/qemu"), Path("/usr/local/share/qemu"),
                  Path("/usr/share/qemu"), Path("/usr/share/AAVMF")):
        code = share / "edk2-aarch64-code.fd"
        vars_ = share / "edk2-arm-vars.fd"
        if code.exists():
            return code, vars_
    raise FileNotFoundError("edk2-aarch64-code.fd not found: install QEMU (brew install qemu)")


def _seed_iso(dest: Path, user_data: str, hostname: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="seed", dir=dest))
    (d / "user-data").write_text(user_data)
    (d / "meta-data").write_text(f"instance-id: {hostname}\nlocal-hostname: {hostname}\n")
    iso = dest / "seed.iso"
    iso.unlink(missing_ok=True)               # hdiutil will not overwrite
    if platform.system() == "Darwin":
        subprocess.run(["hdiutil", "makehybrid", "-quiet", "-o", str(iso), str(d), "-iso",
                        "-joliet", "-default-volume-name", "cidata"], check=True)
    else:
        tool = shutil.which("genisoimage") or shutil.which("mkisofs") or shutil.which("xorrisofs")
        subprocess.run([tool, "-quiet", "-output", str(iso), "-volid", "cidata", "-joliet",
                        "-rock", str(d)], check=True)
    shutil.rmtree(d)
    return iso


USER_DATA = """#cloud-config
users:
  - name: tb
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: true
    ssh_authorized_keys:
      - {key}
ssh_pwauth: false
disable_root: true
package_update: false
package_upgrade: false
runcmd:
  - [sh, -c, "modprobe usbmon || true"]
  - [sh, -c, "mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug"]
  - [sh, -c, "touch /var/tmp/testbed-ready"]
"""


@dataclass
class Guest:
    distro: str
    devices: tuple[str, ...]
    run_dir: Path
    mem_mb: int = 2048
    cpus: int = 2
    ssh_port: int = field(default_factory=_free_port)
    proc: subprocess.Popen | None = None
    qmp: QMP | None = None

    @property
    def qmp_path(self) -> str:
        # AF_UNIX paths are limited to 104 bytes on macOS
        p = str(self.run_dir / "qmp.sock")
        if len(p) < 100:
            return p
        return str(Path(tempfile.gettempdir()) / f"tb-{self.distro}-{os.getpid()}.sock")

    def _qemu_args(self) -> list[str]:
        code, vars_tpl = _firmware()
        d = self.run_dir
        vars_ = d / "efivars.fd"
        if not vars_.exists():
            if vars_tpl.exists():
                shutil.copy(vars_tpl, vars_)
            else:
                with open(vars_, "wb") as f:
                    f.truncate(64 * 1024 * 1024)
        accel = "hvf" if platform.system() == "Darwin" else "kvm"
        args = ["qemu-system-aarch64", "-machine", "virt,highmem=on", "-accel", accel,
                "-cpu", "host", "-smp", str(self.cpus), "-m", str(self.mem_mb),
                "-drive", f"if=pflash,format=raw,readonly=on,file={code}",
                "-drive", f"if=pflash,format=raw,file={vars_}",
                "-drive", f"if=virtio,format=qcow2,file={d / 'disk.qcow2'}",
                "-drive", f"if=virtio,format=raw,readonly=on,file={d / 'seed.iso'}",
                "-netdev", f"user,id=n0,hostfwd=tcp:127.0.0.1:{self.ssh_port}-:22",
                "-device", "virtio-net-pci,netdev=n0",
                "-device", "qemu-xhci,id=xhci,p2=15,p3=15",
                # QMP input-send-event addresses a DISPLAY's console, and events reach the
                # input device bound to that console whose type matches (key / abs). The
                # resident keyboard and tablet share gpu0; hot-plugged keyboards bind to
                # gpu1, so each keyboard is addressable on its own. -display none keeps
                # the guest headless.
                "-device", "virtio-gpu-pci,id=gpu0", "-device", "virtio-gpu-pci,id=gpu1",
                "-qmp", f"unix:{self.qmp_path},server=on,wait=off",
                "-display", "none", "-serial", f"file:{d / 'console.log'}",
                "-monitor", "none", "-nodefaults"]
        for name in self.devices:
            dev, extra, _ = DEVICES[name]
            args += [a.replace("{d}", str(d)) for a in extra]
            args += ["-device", dev + ",bus=xhci.0"]
            args += _AFTER.get(name, [])
        return args

    def prepare(self) -> None:
        img, _ = DISTROS[self.distro]
        base = IMAGES / img
        if not base.exists():
            raise FileNotFoundError(f"{base} missing: run sh testbed/fetch_images.sh")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        disk = self.run_dir / "disk.qcow2"
        if not disk.exists():
            subprocess.run(["qemu-img", "create", "-q", "-f", "qcow2", "-F", "qcow2",
                            "-b", str(base), str(disk), "12G"], check=True)
        for name, size in (("stick0.img", 256), ("uas0.img", 256)):
            p = self.run_dir / name
            if not p.exists():
                with open(p, "wb") as f:
                    f.truncate(size * 1024 * 1024)
        key = ensure_key().with_suffix(".pub").read_text().strip()
        _seed_iso(self.run_dir, USER_DATA.format(key=key), f"tb-{self.distro}")

    def start(self, boot_timeout: float = 600.0) -> None:
        self.prepare()
        Path(self.qmp_path).unlink(missing_ok=True)
        args = self._qemu_args()
        (self.run_dir / "qemu-args.json").write_text(json.dumps(args, indent=1))
        log = open(self.run_dir / "qemu.log", "wb")
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(100):                  # surface QEMU's own error, not a socket timeout
            if self.proc.poll() is not None:
                raise RuntimeError(f"QEMU exited: {(self.run_dir / 'qemu.log').read_text()[-2000:]}")
            if Path(self.qmp_path).exists():
                break
            time.sleep(0.1)
        self.qmp = QMP(self.qmp_path)
        deadline = time.time() + boot_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"QEMU exited: {(self.run_dir / 'qemu.log').read_text()[-2000:]}")
            r = self.ssh("test -f /var/tmp/testbed-ready && echo ok", check=False, timeout=15)
            if r.returncode == 0 and "ok" in r.stdout:
                return
            time.sleep(5)
        raise TimeoutError(f"{self.distro}: not reachable over SSH in {boot_timeout:.0f}s; "
                           f"see {self.run_dir / 'console.log'}")

    def ssh(self, cmd: str, *, check: bool = True, timeout: float = 120.0,
            sudo: bool = False) -> subprocess.CompletedProcess:
        full = ["ssh", "-i", str(KEY), "-p", str(self.ssh_port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10", "tb@127.0.0.1",
                ("sudo sh -c " + _q(cmd)) if sudo else cmd]
        try:
            r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            r = subprocess.CompletedProcess(full, 124, e.stdout or "", e.stderr or "")
        if check and r.returncode != 0:
            raise RuntimeError(f"{self.distro}: `{cmd}` -> {r.returncode}: {r.stderr.strip()[-800:]}")
        return r

    def scp_from(self, remote: str, local: Path) -> None:
        subprocess.run(["scp", "-q", "-i", str(KEY), "-P", str(self.ssh_port),
                        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR", f"tb@127.0.0.1:{remote}", str(local)], check=True)

    def scp_to(self, local: Path, remote: str) -> None:
        subprocess.run(["scp", "-q", "-i", str(KEY), "-P", str(self.ssh_port),
                        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR", str(local), f"tb@127.0.0.1:{remote}"], check=True)

    def stop(self) -> None:
        if self.qmp is not None:
            try:
                self.qmp.cmd("quit")
            except Exception:
                pass
            self.qmp.close()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"
