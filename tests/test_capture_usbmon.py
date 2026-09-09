"""Milestone 2 — usbmon reader.

The Sprint-0 parser was written against simulator output and silently dropped
four real usbmon line forms (CLAUDE.md §4 defect 1). Every one of those forms
is asserted here, so a regression fails rather than quietly reducing the event
stream.
"""

import io
from pathlib import Path

import pytest

from iudbms.capture.usbmon import UsbmonParser, UsbmonSource, parse_line
from iudbms.contract import EventSource

FIXTURE = Path(__file__).parent / "fixtures" / "usbmon" / "kernel_doc_examples.txt"


def _one(line: str):
    ev = parse_line(line)
    assert ev is not None, f"failed to parse: {line}"
    return ev


# --------------------------------------------------------------------------
# the four forms Sprint-0 dropped
# --------------------------------------------------------------------------

def test_control_submission_with_setup_packet():
    e = _one("e7217400 3730515507 S Ci:1:001:0 s 80 06 0100 0000 0012 18 <")
    assert (e.evt, e.ttype, e.direction) == ("S", "control", "in")
    assert e.length == 18            # wLength, the requested data length
    assert e.setup == bytes.fromhex("8006") + b"\x00\x01" + b"\x00\x00" + b"\x12\x00"
    assert len(e.setup) == 8
    assert e.data == b""             # "<" tag: nothing captured yet
    assert e.data_truncated


def test_every_submission_line_parses():
    """'<' is a data tag like any other; Sprint-0's regex required '= <hex>'."""
    e = _one("f5ca4980 3854338202 S Ii:1:003:1 -115:8 8 <")
    assert e.evt == "S" and e.status == -115 and e.interval == 8 and e.length == 8
    assert e.data == b"" and e.data_truncated


def test_uncaptured_payload_dash_tag():
    """A 512-byte bulk transfer whose payload the kernel did not record."""
    e = _one("f4849cc0 3854384867 C Bi:1:005:1 0 512 -")
    assert e.length == 512
    assert e.data == b""
    assert e.data_truncated, "must not look like a zero-length transfer"


def test_iso_completion_with_four_part_status_word():
    e = _one("d6c0c000 3854385100 C Zi:1:006:3 0:1:2054:0 3072 = aabbccdd")
    assert e.ttype == "iso" and e.status == 0 and e.interval == 1
    assert e.length == 3072
    assert e.data == bytes.fromhex("aabbccdd")
    assert e.data_truncated, "4 captured bytes of a 3072-byte transfer is truncated"


# --------------------------------------------------------------------------
# the length/interval confusion
# --------------------------------------------------------------------------

def test_interval_is_not_the_length():
    """Regression guard for CLAUDE.md §4 defect 2. In '0:4 8' the 4 is the
    polling interval and the 8 is the data length; Sprint-0 returned 4."""
    e = _one("f5ca4981 3854346400 C Ii:1:003:1 0:4 8 = 00001600 00000000")
    assert e.interval == 4
    assert e.length == 8
    assert len(e.data) == 8


def test_interval_only_for_intr_and_iso():
    """A bulk status word has no interval field, so a colon must not invent one."""
    e = _one("f4849cc0 3854384800 S Bo:1:005:2 -115 512 = deadbeef cafebabe")
    assert e.interval is None
    assert e.length == 512 and len(e.data) == 8 and e.data_truncated


# --------------------------------------------------------------------------
# general grammar
# --------------------------------------------------------------------------

def test_zero_length_completion_is_not_truncated():
    e = _one("f4849cc1 3854384900 C Bi:1:005:1 0 0")
    assert e.length == 0 and e.data == b"" and not e.data_truncated


def test_error_event():
    e = _one("f4849cc0 3854390000 E Bi:1:005:1 -71 0")
    assert e.evt == "E" and e.status == -71


def test_unpadded_bus_number():
    e = _one("e7217400 3854400000 C Ci:12:001:0 0 8 = 12010000 00000008")
    assert (e.bus, e.dev, e.ep) == (12, 1, 0)


def test_endpoint_is_a_number_not_an_address():
    e = _one("f5ca4980 3854346339 C Ii:1:003:1 0:8 8 = 00000000 00000000")
    assert e.ep == 1
    e.validate()          # would raise if ep carried the 0x80 direction bit


@pytest.mark.parametrize("letter,ttype", [("C", "control"), ("Z", "iso"),
                                          ("I", "intr"), ("B", "bulk")])
def test_all_transfer_type_letters(letter, ttype):
    assert _one(f"aaaa 1000 C {letter}i:1:003:1 0 8 = 0000000000000000").ttype == ttype


@pytest.mark.parametrize("line", [
    "",
    "not a usbmon line at all",
    "aaaa 1000 X Ii:1:003:1 0 8",             # bad event type
    "aaaa notanumber C Ii:1:003:1 0 8",       # bad timestamp
    "aaaa 1000 C Qi:1:003:1 0 8",             # unknown transfer type
    "aaaa 1000 C Ix:1:003:1 0 8",             # bad direction
    "aaaa 1000 C Ii:1:003 0 8",               # truncated address word
    "aaaa 1000 C Ii:1:003:1",                 # no length
])
def test_malformed_lines_rejected(line):
    assert parse_line(line) is None


# --------------------------------------------------------------------------
# UsbmonParser: rebasing, wrap, stats
# --------------------------------------------------------------------------

def test_timestamps_rebase_to_first_event():
    p = UsbmonParser()
    evs = list(p.feed([
        "aaaa 5000000 C Ii:1:003:1 0:8 8 = 0000000000000000",
        "aaaa 5250000 C Ii:1:003:1 0:8 8 = 0000000000000000",
    ]))
    assert evs[0].ts == 0.0
    assert evs[1].ts == pytest.approx(0.25)


def test_microsecond_counter_wrap_does_not_go_backwards():
    """usbmon timestamps are unsigned 32-bit microseconds and wrap every ~71 min."""
    wrap = 1 << 32
    p = UsbmonParser()
    ts = [e.ts for e in p.feed([
        "aaaa 1000000 C Ii:1:003:1 0:8 8 = 0000000000000000",
        f"aaaa {wrap - 1000000} C Ii:1:003:1 0:8 8 = 0000000000000000",
        "aaaa 500000 C Ii:1:003:1 0:8 8 = 0000000000000000",
    ])]
    assert ts == [pytest.approx(0.0), pytest.approx(4292.967296), pytest.approx(4294.467296)]
    assert all(b >= a for a, b in zip(ts, ts[1:]))


def test_stats_account_for_every_line():
    p = UsbmonParser()
    lines = [
        "aaaa 1000 C Ii:1:003:1 0:8 8 = 0000000000000000",
        "total garbage here",
        "",                       # blank lines are not records and are not counted
        "# a comment",
        "aaaa 2000 C Ii:1:003:1 0:8 8 = 0000000000000000",
    ]
    evs = list(p.feed(lines))
    assert len(evs) == 2
    assert p.stats.lines_seen == 3
    assert p.stats.events_emitted == 2
    assert p.stats.lines_unparsed == 1
    assert p.stats.loss_rate == pytest.approx(1 / 3)
    assert p.stats.unparsed_samples == ["total garbage here"]


def test_strict_mode_validates_every_event():
    p = UsbmonParser(strict=True)
    assert len(list(p.feed(["aaaa 1000 C Ii:1:003:1 0:8 8 = 0000000000000000"]))) == 1


# --------------------------------------------------------------------------
# UsbmonSource
# --------------------------------------------------------------------------

def test_source_satisfies_the_event_source_protocol():
    assert isinstance(UsbmonSource(stream=io.StringIO("")), EventSource)


def test_every_fixture_line_parses():
    """The whole point of the fixture corpus: zero loss on real kernel output."""
    src = UsbmonSource(path=FIXTURE)
    evs = list(src.events())
    assert src.stats.lines_unparsed == 0, (
        f"dropped real usbmon lines: {src.stats.unparsed_samples}"
    )
    assert src.stats.loss_rate == 0.0
    assert len(evs) == 13
    assert src.stats.events_emitted == 13


def test_fixture_covers_every_transfer_type_and_event_kind():
    evs = list(UsbmonSource(path=FIXTURE).events())
    assert {e.ttype for e in evs} == {"control", "iso", "intr", "bulk"}
    assert {e.evt for e in evs} == {"S", "C", "E"}


def test_all_fixture_events_satisfy_the_contract():
    for e in UsbmonSource(path=FIXTURE).events():
        e.validate()


def test_events_are_non_decreasing_in_time():
    ts = [e.ts for e in UsbmonSource(path=FIXTURE).events()]
    assert all(b >= a for a, b in zip(ts, ts[1:]))


def test_requires_exactly_one_of_path_or_stream():
    with pytest.raises(ValueError):
        UsbmonSource()
    with pytest.raises(ValueError):
        UsbmonSource(path=FIXTURE, stream=io.StringIO(""))


def test_live_raises_a_helpful_error_off_linux(tmp_path):
    with pytest.raises(FileNotFoundError, match="usbmon"):
        UsbmonSource.live(bus=1, usbmon_dir=tmp_path)


@pytest.mark.slow
def test_throughput_meets_plan_requirement():
    """Plan §4.1: >= 20k events/s sustained."""
    import time
    line = "f5ca4980 3854346339 C Ii:1:003:1 0:8 8 = 00000000 00000000\n"
    n = 100_000
    t = time.perf_counter()
    got = sum(1 for _ in UsbmonSource(stream=io.StringIO(line * n)).events())
    rate = got / (time.perf_counter() - t)
    assert got == n
    assert rate >= 20_000, f"only {rate:,.0f} events/s"
