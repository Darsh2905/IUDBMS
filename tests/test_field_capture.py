"""Field capture: pseudonymised payloads leave every feature bit-exact."""

import importlib.util
from pathlib import Path

import numpy as np

from iudbms.capture.usbmon import UsbmonSource
from iudbms.features.extractor import WindowFeatureExtractor, window_events
from iudbms.live.cli import _load_descriptor_snapshot
from iudbms.storage.traces import PayloadPseudonymizer

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures" / "usbmon"
spec = importlib.util.spec_from_file_location("field_capture", ROOT / "scripts" / "field_capture.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def _features(path, desc):
    out = {}
    ev = list(UsbmonSource(path=path).events())
    for key, d in desc.items():
        de = [e for e in ev if e.device_key == key]
        if not de:
            continue
        ex = WindowFeatureExtractor()
        out[key] = np.vstack([ex.extract(b, t0, d)
                              for t0, b in window_events(de, duration=max(e.ts for e in ev))])
    return out


def test_pseudonymised_capture_featurises_identically(tmp_path):
    raw = (FX / "real_qemu_vm_capture.txt").read_text().splitlines(keepends=True)
    pz = PayloadPseudonymizer(b"k" * 32)
    out = tmp_path / "p.txt"
    out.write_text("".join(fc.pseudonymise_line(ln, pz) for ln in raw))
    desc = _load_descriptor_snapshot(FX / "real_qemu_vm_descriptors.json")
    a, b = _features(FX / "real_qemu_vm_capture.txt", desc), _features(out, desc)
    assert a.keys() == b.keys() and a
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_payload_bytes_do_not_survive(tmp_path):
    raw = (FX / "real_qemu_vm_capture.txt").read_text().splitlines(keepends=True)
    pz = PayloadPseudonymizer(b"k" * 32)
    orig = [e.data for e in UsbmonSource(path=FX / "real_qemu_vm_capture.txt").events() if e.data]
    out = tmp_path / "p.txt"
    out.write_text("".join(fc.pseudonymise_line(ln, pz) for ln in raw))
    new = [e.data for e in UsbmonSource(path=out).events() if e.data]
    assert len(orig) == len(new) and [len(x) for x in orig] == [len(x) for x in new]
    long = [(o, n) for o, n in zip(orig, new) if len(o) >= 4]
    assert long and sum(o == n for o, n in long) == 0      # no multi-byte payload unchanged


def test_no_payload_mode_drops_bytes_but_keeps_lines():
    line = "ffff8800 4766122 C Ii:3:003:1 0:10 8 = 00000b00 00000000\n"
    assert fc.pseudonymise_line(line, None) == "ffff8800 4766122 C Ii:3:003:1 0:10 8 -\n"
    assert fc.pseudonymise_line("ffff 1 S Ci:1:001:0 s a3 00 0000 0001 0004 4 <\n", None).endswith("<\n")
