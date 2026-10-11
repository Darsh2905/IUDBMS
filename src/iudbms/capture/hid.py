"""
What a HID interface can send, read from its report descriptor.

A HID device's interface class says only "HID"; what the host will accept from it is
declared by its report descriptor (USB HID 1.11 section 6.2.2). Keystrokes reach the
host only through usages on the Keyboard/Keypad usage page (0x07); consumer controls
(page 0x0C) can launch applications. A pointer -- mouse, tablet, pen -- declares
Generic Desktop (0x01), Button (0x09) and Digitizer (0x0D) usages and nothing else.

This matters because the threat the detector models is keystroke injection, and the
kernel exposes the descriptor in sysfs (``.../<hid-id>/report_descriptor``), so
"can this device type?" is a fact about the device, not a guess from its behaviour.
"""

from __future__ import annotations

__all__ = ["usage_pages", "KEYSTROKE_PAGES", "keystroke_pages_in"]

#: Usage pages through which a HID device can make the host act on its behalf:
#: Keyboard/Keypad (0x07) and Consumer (0x0C: application-launch and media keys).
KEYSTROKE_PAGES = frozenset({0x07, 0x0C})


def usage_pages(desc: bytes) -> tuple[int, ...]:
    """Every Usage Page declared in a HID report descriptor, in first-seen order.

    Walks the item stream: short items are a prefix byte (size in bits 1:0, with 3
    meaning 4 bytes; type in bits 3:2; tag in bits 7:4) followed by that many data
    bytes; long items (prefix 0xFE) carry their own size. Usage Page is the GLOBAL
    item with tag 0 (prefix 0x04-0x07). Extended (32-bit) Usage items in LOCAL
    tag 0 also carry a page in their high 16 bits, which is included.
    """
    out: list[int] = []
    i, n = 0, len(desc)
    while i < n:
        b = desc[i]
        if b == 0xFE:                                    # long item
            if i + 2 >= n:
                break
            i += 3 + desc[i + 1]
            continue
        size = (0, 1, 2, 4)[b & 0x03]
        typ, tag = (b >> 2) & 0x03, (b >> 4) & 0x0F
        data = desc[i + 1:i + 1 + size]
        if len(data) < size:
            break                                        # truncated: stop, do not guess
        val = int.from_bytes(data, "little") if size else 0
        if typ == 1 and tag == 0:                        # Global: Usage Page
            page = val & 0xFFFF
        elif typ == 2 and tag == 0 and size == 4:        # Local: extended Usage (page:id)
            page = val >> 16
        else:
            page = None
        if page is not None and page not in out:
            out.append(page)
        i += 1 + size
    return tuple(out)


def keystroke_pages_in(pages) -> bool:
    return bool(set(pages or ()) & KEYSTROKE_PAGES)
