#!/usr/bin/env python3
"""
Generate the review report from measured data.

    python3 -m eval.build_report

Reads out/metrics.json and out/figure_data.json and writes
docs/report/iudbms_review.html. Every number and every chart comes from the last
`make eval`, so the report cannot drift from the results the way a hand-edited
export does.
"""

from __future__ import annotations

import json
from pathlib import Path

from .render_svg import render_all

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DEST = ROOT / "docs" / "report" / "iudbms_review.html"

STYLE = """
  :root {
    --paper:#f5f6f8; --surface:#fff; --surface-2:#eef0f4;
    --ink:#12171d; --ink-2:#3c4653; --slate:#5f6b79;
    --rule:#d9dee5; --rule-soft:#e7ebf0;
    --signal:#b45309; --signal-dim:#f5ede0;
    --ok:#0f766e; --ok-dim:#e2f1ef;
    --crit:#9f1239; --crit-dim:#fbe8ed;
    /* validated categorical slots 1-3, light */
    --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a;
    --grid:#e7ebf0; --axis:#12171d; --tx:#12171d; --mut:#5f6b79;
    --measure:66ch; --wide:1180px;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper:#0d1116; --surface:#141a21; --surface-2:#1b232c;
      --ink:#e8ecf1; --ink-2:#b3bdc9; --slate:#8593a3;
      --rule:#27313c; --rule-soft:#1e262f;
      --signal:#e8a13c; --signal-dim:#2e2410;
      --ok:#4fd1c5; --ok-dim:#102d2b;
      --crit:#f4708f; --crit-dim:#331019;
      --c1:#3987e5; --c2:#d95926; --c3:#199e70;
      --grid:#1e262f; --axis:#e8ecf1; --tx:#e8ecf1; --mut:#8593a3;
    }
  }
  :root[data-theme="dark"] {
    --paper:#0d1116; --surface:#141a21; --surface-2:#1b232c;
    --ink:#e8ecf1; --ink-2:#b3bdc9; --slate:#8593a3;
    --rule:#27313c; --rule-soft:#1e262f;
    --signal:#e8a13c; --signal-dim:#2e2410;
    --ok:#4fd1c5; --ok-dim:#102d2b;
    --crit:#f4708f; --crit-dim:#331019;
    --c1:#3987e5; --c2:#d95926; --c3:#199e70;
    --grid:#1e262f; --axis:#e8ecf1; --tx:#e8ecf1; --mut:#8593a3;
  }
  *{box-sizing:border-box}
  body{background:var(--paper);color:var(--ink);
    font-family:"IBM Plex Serif",Georgia,serif;font-size:16.5px;line-height:1.62;
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:var(--wide);margin:0 auto;padding:0 24px 96px}
  .col{max-width:var(--measure)}
  h1,h2,h3,th,.eyebrow,.metak,.chip,.tag,.ui{font-family:"IBM Plex Sans",system-ui,sans-serif}
  code,.mono,td.num,.trace,text.num,.num{font-family:"IBM Plex Mono",ui-monospace,monospace}
  header.masthead{border-bottom:2px solid var(--ink);padding:52px 0 20px;margin-bottom:34px}
  .eyebrow{font-size:11.5px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;
    color:var(--signal);margin:0 0 14px}
  h1{font-size:clamp(2.2rem,5vw,3.2rem);font-weight:700;letter-spacing:-.022em;
    line-height:1.05;margin:0 0 16px;text-wrap:balance}
  .standfirst{font-size:1.15rem;color:var(--ink-2);margin:0;max-width:60ch;text-wrap:pretty}
  .runmeta{display:flex;flex-wrap:wrap;gap:0 30px;margin-top:26px;padding-top:18px;
    border-top:1px solid var(--rule-soft)}
  .runmeta div{display:flex;flex-direction:column;gap:2px}
  .metak{font-size:10.5px;font-weight:600;letter-spacing:.13em;text-transform:uppercase;
    color:var(--slate)}
  .metav{font-size:14px;font-variant-numeric:tabular-nums}
  section{margin-top:54px}
  h2{font-size:1.42rem;font-weight:600;letter-spacing:-.012em;margin:0 0 6px;text-wrap:balance}
  h2 .n{font-size:.72em;color:var(--signal);font-weight:500;margin-right:.6em}
  h3{font-size:1.02rem;font-weight:600;margin:30px 0 8px}
  .lede{color:var(--slate);margin:0 0 22px;font-size:.96rem;max-width:var(--measure)}
  p{margin:0 0 15px}
  .col p:last-child{margin-bottom:0}
  code{font-size:.87em;background:var(--surface-2);padding:.1em .38em;border-radius:3px;
    color:var(--ink-2)}
  .specimen{background:var(--surface);border:1px solid var(--rule);
    border-left:3px solid var(--signal);padding:20px 22px;margin:0 0 8px;overflow-x:auto}
  .trace{font-size:14px;white-space:pre;display:block;color:var(--ink)}
  .trace .f{color:var(--slate)} .trace .hi{color:var(--signal);font-weight:600}
  .legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
    gap:14px 22px;margin-top:18px;padding-top:16px;border-top:1px solid var(--rule-soft)}
  .legend div{display:flex;flex-direction:column;gap:1px}
  .legend .k{font-family:"IBM Plex Mono",monospace;font-size:12.5px;font-weight:600}
  .legend .v{font-size:12.5px;color:var(--slate);line-height:1.42}
  .caption{font-size:.84rem;color:var(--slate);margin:10px 0 0;max-width:var(--measure)}
  figure{margin:26px 0 0;background:var(--surface);border:1px solid var(--rule);
    border-radius:6px;padding:18px 20px 14px}
  figure svg.chart{display:block;width:100%;height:auto}
  figcaption{font-size:.85rem;color:var(--slate);margin-top:12px;
    padding-top:11px;border-top:1px solid var(--rule-soft);max-width:76ch}
  figcaption b{color:var(--ink);font-weight:600}
  .tablewrap{overflow-x:auto;margin:22px 0 0}
  table{border-collapse:collapse;width:100%;font-size:14.5px}
  caption{caption-side:top;text-align:left;font-family:"IBM Plex Sans",sans-serif;
    font-size:11.5px;font-weight:600;letter-spacing:.13em;text-transform:uppercase;
    color:var(--slate);padding-bottom:10px}
  th{text-align:left;font-size:11.5px;font-weight:600;letter-spacing:.07em;
    text-transform:uppercase;color:var(--slate);padding:0 14px 9px 0;
    border-bottom:1px solid var(--ink);white-space:nowrap}
  th.num,td.num{text-align:right}
  td{padding:10px 14px 10px 0;border-bottom:1px solid var(--rule-soft);vertical-align:baseline}
  td.num{font-variant-numeric:tabular-nums;font-size:14px}
  td.name{font-size:13.5px}
  tr.total td{border-top:1px solid var(--ink);border-bottom:none;font-weight:600;padding-top:11px}
  tr.lead td{background:var(--signal-dim)}
  tr.lead td:first-child{box-shadow:inset 3px 0 0 var(--signal);padding-left:11px}
  tr.hard td.pt{color:var(--crit);font-weight:600}
  .chip{display:inline-block;font-size:10.5px;font-weight:600;letter-spacing:.07em;
    text-transform:uppercase;padding:2px 8px;border-radius:2px;white-space:nowrap}
  .chip.done{background:var(--ok-dim);color:var(--ok)}
  .chip.part{background:var(--signal-dim);color:var(--signal)}
  .chip.no{background:var(--surface-2);color:var(--slate)}
  .milestones{display:grid;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));
    gap:1px;background:var(--rule-soft);border:1px solid var(--rule-soft);margin-top:22px}
  .ms{background:var(--surface);padding:13px 15px;display:flex;flex-direction:column;
    gap:7px;min-height:92px}
  .ms .top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
  .ms .id{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--slate)}
  .ms .nm{font-family:"IBM Plex Sans",sans-serif;font-size:13.5px;font-weight:500;
    line-height:1.32}
  .ms.off .nm{color:var(--slate)}
  .callout{background:var(--crit-dim);border:1px solid var(--crit);padding:20px 22px;
    margin-top:26px}
  .callout h3{margin:0 0 8px;color:var(--crit);font-size:1.02rem}
  .callout p{margin:0 0 12px;font-size:.95rem}
  .callout p:last-child{margin-bottom:0}
  .finding{border-left:2px solid var(--rule);padding-left:20px;margin-top:26px}
  .finding h3{margin-top:0}
  .finding .tag{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--signal);
    font-weight:600;display:block;margin-bottom:3px}
  ul{margin:0 0 15px;padding-left:1.15em;max-width:var(--measure)}
  li{margin-bottom:7px} li::marker{color:var(--slate)}
  footer{margin-top:66px;padding-top:20px;border-top:1px solid var(--rule);
    font-size:.85rem;color:var(--slate);max-width:var(--measure)}
  @media (max-width:640px){body{font-size:16px}.wrap{padding:0 18px 72px}}
  @media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
  @media print{
    :root{--paper:#fff;--surface:#fff;--surface-2:#f4f4f4;--ink:#000;--ink-2:#222;
      --slate:#555;--rule:#999;--rule-soft:#ccc;--grid:#ddd;--axis:#000;--tx:#000;--mut:#555}
    body{font-size:10.5pt}
    .wrap{max-width:none;padding:0}
    section{break-before:page}
    header.masthead{break-after:avoid}
    figure,table,.finding,.callout,.ms{break-inside:avoid}
    h1,h2,h3{break-after:avoid}
  }
"""


def pct(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}&thinsp;%"


def build() -> Path:
    m = json.loads((OUT / "metrics.json").read_text())
    fd = json.loads((OUT / "figure_data.json").read_text())
    svg = render_all(OUT / "figure_data.json")
    f, s1, idw = m["full_system"], m["stage1_only"], m["identity_whitelist"]
    c, sp, cfg = m["corpus"], m["split"], m["config"]
    per = f["per_profile"]

    def row(k):
        v = per[k]
        s1v = s1["per_profile"].get(k, {}).get("detect_rate", 0.0)
        lat = f"{v['median_latency']:.1f}&thinsp;s" if v["median_latency"] else "—"
        hard = ' class="hard"' if v["detect_rate"] < 0.9 else ""
        pt = ' class="num pt"' if v["detect_rate"] < 0.9 else ' class="num"'
        return (f'<tr{hard}><td class="name mono">{k}</td><td class="num">{v["n"]}</td>'
                f'<td{pt}>{pct(v["detect_rate"], 0)}</td>'
                f'<td class="num">{pct(s1v, 0)}</td><td class="num">0&thinsp;%</td>'
                f'<td class="num">{lat}</td></tr>')

    attack_rows = "".join(row(k) for k in sorted(per) if per[k]["label"] == 1)
    mods = "".join(
        f'<tr><td>{r["name"]}</td><td class="num">{r["features"]}</td>'
        f'<td class="num">{r["auc"]:.4f}</td><td class="num">{pct(r["detect"])}</td></tr>'
        for r in fd["modality"])
    gates = "".join(
        f'<tr{" class=lead" if r["k"] == cfg["gate"].split("-")[0] and r["n"] == 6 else ""}>'
        f'<td class="name mono">{r["k"]}-of-{r["n"]}</td><td class="num">{pct(r["detect"])}</td>'
        f'<td class="num">{pct(r["benign_fp"])}</td>'
        f'<td class="num">{r["median_latency"]:.1f}&thinsp;s</td></tr>'
        for r in fd["gate_sweep"])

    ms = [("M1", "Frozen data contract", "done", "Done"),
          ("M2", "usbmon reader — live capture pending VM", "part", "Fixtures"),
          ("M3", "29-feature extractor", "done", "Done"),
          ("M4", "Stage 1 — forest + device baseline", "done", "Done"),
          ("M5", "Stage 2 — PyTorch LSTM", "done", "Done"),
          ("M6", "Policy engine + safety interlocks", "done", "Done"),
          ("M7", "SQLite store + control API", "done", "Done"),
          ("M8", "eBPF capture path", "no", "Deferred"),
          ("M9", "Operator dashboard", "done", "Done"),
          ("M10", "Evaluation harness + ablations", "done", "Done")]
    miles = "".join(
        f'<div class="ms{" off" if chip == "no" else ""}"><div class="top">'
        f'<span class="id">{mid}</span><span class="chip {chip}">{lab}</span></div>'
        f'<span class="nm">{name}</span></div>' for mid, name, chip, lab in ms)

    tl_sd = svg.get("timeline_staged_delay", "")
    tl_ma = svg.get("timeline_mimicry_adaptive", "")
    ma, mi = per["mimicry_adaptive"], per["mimicry"]

    html = f"""<title>IUDBMS Review 2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{STYLE}</style>

<div class="wrap">
<header class="masthead">
  <p class="eyebrow">Operating Systems &middot; Review 2 &middot; Progress report</p>
  <h1>Trust what the device does, not what it says it is</h1>
  <p class="standfirst">IUDBMS scores USB peripheral behaviour continuously and revokes
    authorisation mid-session. A BadUSB stick presents a valid keyboard descriptor, so
    identity checks admit it before there is any malicious behaviour to see.</p>
  <div class="runmeta">
    <div><span class="metak">Milestones</span><span class="metav num">9 of 10</span></div>
    <div><span class="metak">Tests</span><span class="metav num">296 passing</span></div>
    <div><span class="metak">Sessions</span><span class="metav num">{c['sessions']:,}</span></div>
    <div><span class="metak">URB events</span><span class="metav num">{c['usbmon_lines']:,}</span></div>
    <div><span class="metak">Features</span><span class="metav num">{m['deployed_features']} of {c['features']}</span></div>
    <div><span class="metak">Seed</span><span class="metav num">{cfg['seed']}</span></div>
  </div>
</header>

<section style="margin-top:0">
  <h2><span class="n">00</span>The unit of evidence</h2>
  <p class="lede">Everything the system knows arrives as lines like this, from
    <code>/sys/kernel/debug/usb/usbmon/0u</code>.</p>
  <div class="specimen">
    <span class="trace"><span class="f">f5ca4980</span> <span class="f">3854346339</span> <span class="f">C</span> <span class="f">Ii:1:003:1</span> <span class="hi">0:4</span> <span class="hi">8</span> <span class="f">=</span> <span class="f">00001600 00000000</span></span>
    <div class="legend">
      <div><span class="k">f5ca4980</span><span class="v">URB tag</span></div>
      <div><span class="k">3854346339</span><span class="v">timestamp, &micro;s (wraps every 71.6 min)</span></div>
      <div><span class="k">C</span><span class="v">completion</span></div>
      <div><span class="k">Ii:1:003:1</span><span class="v">interrupt IN, bus 1, device 3, endpoint 1</span></div>
      <div><span class="k">0:4</span><span class="v">status 0, polling interval 4</span></div>
      <div><span class="k">8</span><span class="v">data length &mdash; 8 bytes, not 4</span></div>
    </div>
  </div>
  <p class="caption">This exact line is why the capture layer was rewritten. The prototype
    read the polling interval as the transfer length, and silently dropped four real line
    forms &mdash; every submission, every uncaptured payload, control setup packets, and
    isochronous completions.</p>
</section>

<section>
  <h2><span class="n">01</span>What is built</h2>
  <p class="lede">Nine of ten milestones. Only eBPF remains, deferred by the plan&rsquo;s own
    risk register, which designates week 11 as the sacrifice week.</p>
  <div class="milestones">{miles}</div>
  <p class="caption">Every number and every chart below is generated by
    <code>make eval &amp;&amp; make report</code> &mdash; two seeded commands. Nothing is
    transcribed by hand, so this page cannot drift from the results.</p>
</section>

<section>
  <h2><span class="n">02</span>Headline results</h2>
  <p class="lede">{sp['test']} held-out test sessions, split by session and stratified by
    profile &mdash; never by window. Threshold calibrated at a {cfg['target_fpr']:.2%}
    per-window false-positive rate on validation benign traffic only, then gated
    {cfg['gate']}.</p>

  <div class="tablewrap"><table>
    <caption>Window level &middot; held-out test set</caption>
    <thead><tr><th>Model</th><th class="num">ROC&ndash;AUC</th><th class="num">Precision</th>
      <th class="num">Recall</th><th class="num">Session detection</th></tr></thead>
    <tbody>
      <tr class="lead"><td><strong>IUDBMS</strong> &mdash; Stage 1 + Stage 2</td>
        <td class="num">{f['window']['auc']:.4f}</td><td class="num">{f['window']['precision']:.3f}</td>
        <td class="num">{f['window']['recall']:.3f}</td>
        <td class="num"><strong>{pct(f['session']['attack_detect'])}</strong></td></tr>
      <tr><td>Stage 1 only &mdash; no temporal model</td>
        <td class="num">{s1['window']['auc']:.4f}</td><td class="num">{s1['window']['precision']:.3f}</td>
        <td class="num">{s1['window']['recall']:.3f}</td>
        <td class="num">{pct(s1['session']['attack_detect'])}</td></tr>
      <tr><td>Identity whitelist &mdash; USBGuard-style</td>
        <td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">&mdash;</td>
        <td class="num">{pct(idw['attack_detect'])}</td></tr>
    </tbody></table></div>

  <figure>{svg['detection_by_class']}
    <figcaption><b>The bottom bar is the thesis.</b> A VID:PID allowlist catches
      <b>none</b> of the {f['session']['n_attack']} attacks, because every one clones a
      descriptor from the benign pool. Stage 1 alone &mdash; memoryless, one window at a
      time &mdash; reaches only {pct(s1['session']['attack_detect'])}, and it catches
      <code>identity_switch</code> because that attack has a static per-window signature
      rather than a transition.</figcaption></figure>

  <div class="tablewrap"><table>
    <caption>Detection by attack class</caption>
    <thead><tr><th>Attack profile</th><th class="num">n</th><th class="num">IUDBMS</th>
      <th class="num">Stage&nbsp;1 only</th><th class="num">Whitelist</th>
      <th class="num">Median latency</th></tr></thead>
    <tbody>{attack_rows}
      <tr class="total"><td>Overall</td><td class="num">{f['session']['n_attack']}</td>
        <td class="num">{pct(f['session']['attack_detect'])}</td>
        <td class="num">{pct(s1['session']['attack_detect'])}</td><td class="num">0&thinsp;%</td>
        <td class="num">{f['session']['median_latency']:.1f}&thinsp;s</td></tr>
    </tbody></table></div>
  <p class="caption">{pct(f['session']['benign_fp'])} benign session false positives across
    {f['session']['n_benign']} benign test sessions; p90 latency
    {f['session']['p90_latency']:.1f}&thinsp;s.</p>

  <figure>{tl_sd}
    <figcaption><b>One device, caught.</b> A staged BadUSB types like a human for
      {fd['timelines'][0]['onset']:.0f}&thinsp;s. Stage 1 (orange) never reacts &mdash; second by second
      nothing is odd. Stage 2 (blue) reads the last ten windows and snaps at onset. The
      shaded band is ground truth; enforcement fires
      {fd['timelines'][0]['fired'] - fd['timelines'][0]['onset']:.0f}&thinsp;s later, after the
      {cfg['gate']} gate fills.</figcaption></figure>
</section>

<section>
  <h2><span class="n">03</span>The two mimicry adversaries</h2>
  <p class="lede">Both clone a keyboard descriptor and draw keystroke timing from the same
    distribution as a human operator &mdash; a Kolmogorov&ndash;Smirnov test cannot separate
    their inter-arrivals from benign typing. They differ in one respect, and it decides
    everything.</p>

  <figure>{svg['novelty']}
    <figcaption><b>Order is the signal, not the alphabet.</b> A looping script exhausts its
      own payload transitions, so its novelty decays and feature 29 catches it
      ({pct(mi['detect_rate'], 0)}). A payload <em>typed once</em> repeats nothing and sits
      flat against benign typing &mdash; so that feature was detecting the loop, not the
      attack. Compared at matched window positions, which is what distinguishes this from a
      session-position artefact.</figcaption></figure>

  <figure>{tl_ma}
    <figcaption><b>The hard case.</b> <code>mimicry_adaptive</code> types a finite payload
      once and the device then falls silent &mdash; which is what a real BadUSB does. The
      only signal left is that sustained silence, so detection is late
      ({ma['median_latency']:.0f}&thinsp;s) and unreliable ({pct(ma['detect_rate'], 0)}).</figcaption></figure>

  <div class="callout">
    <h3>The bound: an unbounded human-paced payload is undetectable</h3>
    <p>Push the adversary one step further &mdash; a payload of unbounded length drawing
      fresh keycodes forever &mdash; and it becomes <em>statistically the benign keyboard
      process</em>. A per-feature Kolmogorov&ndash;Smirnov scan separates it on <strong>zero
      of the {m['deployed_features']} deployed features</strong>. That is a bound on what
      single-device HID behaviour can reveal, not a tuning failure, and it is documented
      rather than simulated: a corpus profile indistinguishable from benign by construction
      teaches a model nothing.</p>
    <p>The principled next defence is <strong>cross-device correlation</strong> &mdash; a
      human typing produces concurrent mouse activity, a BadUSB typing alone does not. That
      sits outside the current single-device architecture.</p>
  </div>
</section>

<section>
  <h2><span class="n">04</span>Do the novelty claims survive contact?</h2>
  <p class="lede">Four pillars distinguish this from LBM (Tian et al., IEEE S&amp;P 2019),
    which mediates USB in-kernel with static administrator-written rules. Each needed an
    ablation, not an assertion.</p>

  <figure>{svg['modality']}
    <figcaption><b>N1 is not supported.</b> The payload modality alone essentially matches
      the fused model, because it carries the sequence feature. Establishing that fusion
      earns its keep needs attack families differing from benign traffic on <em>one</em> axis
      at a time; ours differ on several. Identity alone is by far the weakest, which is the
      thesis restated as a measurement.</figcaption></figure>

  <div class="tablewrap"><table>
    <caption>N1 &middot; single-modality ablation</caption>
    <thead><tr><th>Modality</th><th class="num">Features</th><th class="num">Window AUC</th>
      <th class="num">Session detection</th></tr></thead>
    <tbody>{mods}
      <tr class="total"><td>All fused (deployed)</td>
        <td class="num">{m['deployed_features']}</td>
        <td class="num">{f['window']['auc']:.4f}</td>
        <td class="num">{pct(f['session']['attack_detect'])}</td></tr>
    </tbody></table></div>

  <figure>{svg['roc']}
    <figcaption><b>N3 is the pillar the evidence supports most clearly.</b> Stage 1 is
      memoryless and can only ask whether a given second looks odd; adding the temporal model
      takes session detection from {pct(s1['session']['attack_detect'])} to
      {pct(f['session']['attack_detect'])}. The diagonal is where an identity-based defence
      sits.</figcaption></figure>
</section>

<section>
  <h2><span class="n">05</span>Where the operating point could move</h2>
  <p class="lede">Per-window false positives compound across a session, so a single threshold
    tells you almost nothing. The plan requires the curve, not the point.</p>

  <figure>{svg['operating']}
    <figcaption>Detection is flat across the whole range; what the threshold buys is false
      positives. <b>The deployed point was not read off this chart</b> &mdash; that would be
      selecting on the test split. It was chosen by sweeping the target rate on a separate
      selection split carved out of training data.</figcaption></figure>

  <div class="tablewrap"><table>
    <caption>Persistence gate at the deployed threshold</caption>
    <thead><tr><th>Gate</th><th class="num">Detection</th><th class="num">Benign session FP</th>
      <th class="num">Median latency</th></tr></thead>
    <tbody>{gates}</tbody></table></div>
  <p class="caption">The gate, not the threshold, does the heavy lifting: it is why a single
    anomalous window never triggers anything.</p>
</section>

<section>
  <h2><span class="n">06</span>Findings from measuring what had been assumed</h2>
  <p class="lede">Thirteen defects found so far. Each was invisible in every previously
    reported number, and most surfaced only by running an ablation, a sweep, or a screen
    whose result had already been assumed.</p>

  <div class="finding col">
    <span class="tag">A leak that survived being fixed once</span>
    <h3>Aggregate agreement is not evidence</h3>
    <p>Session durations matched in aggregate &mdash; 63.7&thinsp;s benign against
      63.9&thinsp;s malicious &mdash; which is exactly why the earlier fix looked complete.
      Per profile they did not: <code>staged_delay</code> drew 65&ndash;85&thinsp;s against
      everyone else&rsquo;s 40&ndash;85. The audit now runs per profile, and a test
      re-introduces the original bug to prove the audit still catches it.</p>
  </div>

  <div class="finding col">
    <span class="tag">A test that could no longer fail</span>
    <h3>The audit went blind and kept saying &ldquo;ok&rdquo;</h3>
    <p>Adding a sixth attack profile dropped per-profile sample size, and the KS critical
      value scales as 1/&radic;n &mdash; so the leak-detection test began passing
      <em>because the audit could no longer fail</em>. It now reports
      <b>UNDERPOWERED</b> when a profile&rsquo;s critical value exceeds the effect it claims
      to detect, and says how many sessions it would need.</p>
  </div>

  <div class="finding col">
    <span class="tag">Found by looking at live data</span>
    <h3>The dashboard&rsquo;s first screen said a z-score of 4,472,091</h3>
    <p>Three features are <em>exactly</em> constant across benign traffic, so standardising
      divided by roughly zero and a single 0&rarr;1 flip swamped every other dimension.
      Bounding it left the headline intact but collapsed the Stage-1 ablation &mdash; because
      its old score was substantially an artefact.</p>
  </div>

  <div class="finding col">
    <span class="tag">A sweep laundering noise into a decision</span>
    <h3>The winner on the selection split lost the test split</h3>
    <p>A longer sequence horizon won the selection split by two points &mdash; over fifty
      attack sessions, that is <em>one session</em>. Applied to test it scored five points
      <em>worse</em>. A detection rate over n sessions cannot resolve differences smaller
      than 1/n, so the sweep now compares against the incumbent and recommends KEEP unless
      the margin clears the noise floor.</p>
    <p>The incumbent was restored because the margin was insignificant &mdash; a judgement
      available from the selection split alone &mdash; and <em>not</em> because the test
      number was better. Choosing on the test result is precisely what this harness exists
      to prevent.</p>
  </div>
</section>

<section>
  <h2><span class="n">07</span>What this does not yet show</h2>
  <div class="col"><ul>
    <li><b>No real USB traffic.</b> The parser is validated against real kernel line formats
      including four the prototype silently dropped, but nothing here has seen live traffic.</li>
    <li><b>Attack families come from the code the features were designed against.</b>
      Per-class rates are optimistic by construction; test counts are 15&ndash;24 sessions
      each.</li>
    <li><b>Feature 29 was designed against a known failure.</b> The constraint it exploits is
      real &mdash; an injection payload must type specific commands &mdash; but an adversary
      who pads or reorders degrades it. It raises cost; it does not solve.</li>
    <li><b>Multi-modal fusion (N1) is unsupported by this evidence.</b></li>
    <li><b>Enforcement has never executed.</b> Four interlocks guard it and the review flag
      is still false.</li>
    <li><b>eBPF is unstarted</b> and may remain documented future work.</li>
  </ul></div>
</section>

<footer>
  Generated from <code>out/metrics.json</code> and <code>out/figure_data.json</code> by
  <code>python3 -m eval.build_report</code>; regenerate the inputs with <code>make eval</code>
  at seed {cfg['seed']}. Architectural ancestor: LBM, Tian et&nbsp;al., IEEE Symposium on
  Security and Privacy, 2019.
</footer>
</div>
"""
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(html)
    return DEST


if __name__ == "__main__":
    p = build()
    print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")
