"""
Add HID usage pages to descriptor snapshots taken before they were recorded.

    python3 -m testbed.patch_hid_pages RUN_DIR [RUN_DIR ...] --reference REF_RUN_DIR

The first three benign testbed sessions were snapshotted before contract 3.1.0 read
the HID report descriptor. QEMU's emulated HID devices have FIXED report descriptors
for a given QEMU binary, so their usage pages can be copied from a later snapshot of
the same device models taken by the same QEMU. The match is exact on (vid, pid,
product, interface class/subclass/protocol); anything unmatched is left unknown
(None), which the policy treats conservatively. The original snapshot is kept as
descriptors.orig.json and every change is listed in hid_pages_patch.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _key(d: dict, i: dict) -> tuple:
    return (d.get("vid"), d.get("pid"), d.get("product"), i["bInterfaceClass"],
            i.get("bInterfaceSubClass", 0), i.get("bInterfaceProtocol", 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--reference", type=Path, required=True)
    a = ap.parse_args(argv)
    ref = json.loads((a.reference / "descriptors.json").read_text())
    pages = {}
    for d in ref.values():
        for i in d.get("interfaces", []):
            if i.get("hid_usage_pages") is not None:
                pages[_key(d, i)] = i["hid_usage_pages"]
    if not pages:
        raise SystemExit(f"{a.reference} has no HID usage pages to copy")
    for run in a.runs:
        src = run / "descriptors.json"
        orig = run / "descriptors.orig.json"
        if not orig.exists():
            shutil.copy(src, orig)
        desc = json.loads(orig.read_text())
        changes = []
        for key, d in desc.items():
            for i in d.get("interfaces", []):
                if i["bInterfaceClass"] == 3 and i.get("hid_usage_pages") is None:
                    p = pages.get(_key(d, i))
                    if p is not None:
                        i["hid_usage_pages"] = p
                        changes.append({"device": key, "product": d.get("product"), "pages": p})
        src.write_text(json.dumps(desc, indent=1))
        (run / "hid_pages_patch.json").write_text(json.dumps(
            {"reference": str(a.reference), "changes": changes}, indent=1))
        print(f"{run.name}: {len(changes)} HID interfaces given usage pages from {a.reference.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
