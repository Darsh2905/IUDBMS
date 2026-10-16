"""
House style for the paper's wording: no hyphens in compounds that do not need one, and no
colons joining clauses (each becomes a new sentence).

The authors' own passages (scripts/paper_supplied_text.json) are kept verbatim in that file.
These rules are applied on top of them at build time, so the original wording stays
recoverable, and every change made to it is listed here, not hidden in the document.

    style(text)               hyphen rule, for any text
    supplied(key, text)       colon rewrites for one supplied paragraph, then the hyphen rule

Not touched: the reference list (bibliographic data is quoted as published), identifiers and
numbers (6.12.111+deb13, 1.0e-03), section references (VI-B), and the notation 4-of-6.
"""

from __future__ import annotations

import re

#: hyphenated tokens that stay hyphenated
KEEP = {"4-of-6", "mim-a"}

#: hyphenated tokens that are NOT simply split into two words
SPECIAL = {
    "Kolmogorov-Smirnov": "Kolmogorov–Smirnov",    # a name pair takes an en dash
    "auto-encoder": "autoencoder",
    "Multi-modal": "Multimodal",
    "hand-written": "handwritten",
    "grad-norm": "gradient norm",
    "0-1": "0 to 1",
    "Pre-Registration": "Pre registration",     # heading, sentence-style like the supplied "pre registered"
}

_TOKEN = re.compile(
    r"(?P<sci>\b\d+(?:\.\d+)?e-\d+\b)"                       # 1.0e-03, 6.5e-08
    r"|(?P<sec>\b[IVX]+-[A-Z]\b)"                             # Section VI-B, Table IV-C
    r"|(?P<tok>[A-Za-z0-9µ]+(?:-[A-Za-z0-9µ]+)+)")  # everything else


def _hyphen(m: re.Match) -> str:
    if m.group("sci") or m.group("sec"):
        return m.group(0)
    tok = m.group("tok")
    if tok in KEEP:
        return tok
    if tok in SPECIAL:
        return SPECIAL[tok]
    return tok.replace("-", " ")


def style(text: str) -> str:
    """Remove hyphens from ordinary compounds."""
    return _TOKEN.sub(_hyphen, text)


#: Colon rewrites for the supplied paragraphs: (old, new) pairs, applied in order. Each `old`
#: must be present, so a change to the source text fails the build instead of passing silently.
COLONS: dict[str, list[tuple[str, str]]] = {
    "contrib_1": [("A working pipeline (implemented): capture,", "A working pipeline (implemented). Capture,")],
    "contrib_2": [("Setting the alarm threshold: The usual", "Setting the alarm threshold. The usual")],
    "contrib_4": [("It showed two things: the learned model,", "It showed two things. The learned model,"),
                  ("raising false alarms; and refusing to judge", "raising false alarms. Refusing to judge")],
    "intro_3": [("the scope of its evidence:", "the scope of its evidence.")],
    "ivc_2": [("is reported as unmodelled: it is scored", "is reported as unmodelled. It is scored")],
    "va_intro": [("five profiles: keyboard, mouse,", "five profiles. These are keyboard, mouse,"),
                 ("and their sources, are:", "and their sources, are listed here.")],
    "va_b1": [("Typing speed: each", "Typing speed. Each")],
    "va_b2": [("Key releases: as the", "Key releases. As the")],
    "va_b3": [("Text: a Zipf-weighted", "Text. A Zipf-weighted")],
    "va_b4": [("Idle periods: 40% of benign devices are always active,",
               "Idle periods. Of benign devices, 40% are always active,")],
    "va_b5": [("Session length and attack onset: every", "Session length and attack onset. Every")],
    "va_tableref": [("against real people: the 1,972", "against real people. These are the 1,972")],
    "vb_1": [("the original corpus: 1,000 sessions,", "the original corpus. It has 1,000 sessions,"),
             ("never by window: 30% of each profile is held out for testing,",
              "never by window. Each profile has 30% held out for testing,")],
    "vc_1": [("hardware acceleration: Ubuntu 24.04", "hardware acceleration. These are Ubuntu 24.04"),
             ("11 emulated USB devices: a boot keyboard,", "11 emulated USB devices. These are a boot keyboard,")],
    "vc_2": [("newly attached keyboards: the five", "newly attached keyboards. These are the five")],
    "vd_1": [("Seeds: 7 for Corpora A and B", "The seeds are 7 for Corpora A and B")],
    "vib_2": [("on a fresh seed: keyboard false alarms", "on a fresh seed. Keyboard false alarms")],
    "vic_2": [("families have none: their best", "families have none. Their best")],
    "vic_3": [("to 68.2%: identity switch", "to 68.2%. Identity switch")],
    "vid_2": [("27 benign devices: a network adapter", "27 benign devices. These were a network adapter")],
    "vie_1": [("did not: pooled over", "did not. Pooled over")],
    "vii_2": [("is operational: a monitor that", "is operational. It is a monitor that")],
}


def supplied(key: str, text: str) -> str:
    for old, new in COLONS.get(key, []):
        assert text.count(old) == 1, f"{key}: {old!r} found {text.count(old)} times in the supplied text"
        text = text.replace(old, new)
    return style(text)
