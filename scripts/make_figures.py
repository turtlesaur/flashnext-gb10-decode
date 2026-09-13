#!/usr/bin/env python3
"""Generate the report figures as standalone SVGs, light and dark variants.

Palette: the validated default data-viz palette. Categorical slots 1 (blue) and 2 (orange) for
two-series comparisons; a single-hue ordinal blue ramp for the ordered-configuration chart.
Both were checked with the palette validator in both modes before use.

Output: docs/figures/<name>-light.svg and -dark.svg, embedded from the README with <picture>
so GitHub picks the right one per theme.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")

THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834",
                  ramp=["#86b6ef", "#3987e5", "#256abf", "#0d366b"], neutral="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926",
                 ramp=["#cde2fb", "#86b6ef", "#3987e5", "#184f95"], neutral="#383835"),
}
FONT = ("ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
        "'Helvetica Neue',Arial,sans-serif")


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def head(w, h, t, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" role="img" aria-label="{esc(title)}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>')


def txt(x, y, s, fill, size=13, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


def titleblock(t, title, sub, w, sub_lines=1):
    """Title plus a subtitle wrapped to the canvas width (12.5px, ~0.58 em advance)."""
    o = txt(24, 34, title, t["ink"], 17, weight="600")
    if not sub:
        return o
    budget = int((w - 48) / (12.5 * 0.58))
    lines, cur = [], ""
    for word in sub.split():
        if cur and len(cur) + 1 + len(word) > budget:
            lines.append(cur); cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines[:sub_lines]):
        o += txt(24, 55 + i * 17, line, t["ink2"], 12.5)
    if len(lines) > sub_lines:
        raise ValueError(f"subtitle needs {len(lines)} lines but only {sub_lines} reserved: {sub!r}")
    return o


def rbar_h(x, y, w, h, fill, r=4):
    """Horizontal bar, rounded at the data end only (anchored to the baseline at x)."""
    if w <= 0.5:
        return ""
    r = min(r, w, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x+w-r:.1f} a{r},{r} 0 0 1 {r},{r} V{y+h-r:.1f} '
            f'a{r},{r} 0 0 1 {-r},{r} H{x:.1f} Z" fill="{fill}"/>')


def rbar_v(x, y, w, h, fill, r=4):
    """Vertical bar, rounded at the top (data end), anchored to the baseline at y+h."""
    if h <= 0.5:
        return ""
    r = min(r, h, w / 2)
    return (f'<path d="M{x:.1f},{y+h:.1f} V{y+r:.1f} a{r},{r} 0 0 1 {r},{-r} H{x+w-r:.1f} '
            f'a{r},{r} 0 0 1 {r},{r} V{y+h:.1f} Z" fill="{fill}"/>')


def legend(x, y, items, t):
    o = ""
    for label, color in items:
        o += f'<rect x="{x:.1f}" y="{y-9:.1f}" width="11" height="11" rx="2.5" fill="{color}"/>'
        o += txt(x + 17, y, label, t["ink2"], 12.5)
        x += 21 + 7.0 * len(label)
    return o


# --------------------------------------------------------------------------- figure 1
def fig_step_anatomy(mode):
    t = THEME[mode]
    rows = [("dense bf16 GEMMs (M>=2)", 53.5, 51.2),
            ("draft-pass GEMV (M=1)", 28.3, 5.8),
            ("NVFP4 grouped MoE", 24.1, 23.4),
            ("GDN / QSA / other", 9.8, 9.1),
            ("bf16 MoE (MTP layer)", 3.4, 3.1)]
    W, H = 820, 400
    L, R, TOP = 210, 70, 96
    plot = W - L - R
    mx = 56.0
    o = head(W, H, t, "Baseline decode step by kernel family")
    o += titleblock(t, "Where a decode step goes",
                    "GPU time per step, paired profiler traces, warm cache. "
                    "One 4-token verify pass plus three 1-token drafts.", W)
    o += legend(L, 76, [("baseline", t["s1"]), ("with FR-Spec draft head", t["s2"])], t)
    bh, gap, pair = 17, 2, 8
    y = TOP + 10
    for name, a, b in rows:
        for g in range(0, 7):
            gx = L + plot * (g * 10) / mx
            if g:
                o += (f'<line x1="{gx:.1f}" y1="{TOP:.1f}" x2="{gx:.1f}" y2="{H-46:.1f}" '
                      f'stroke="{t["grid"]}" stroke-width="1"/>')
        o += txt(L - 14, y + bh + 1, name, t["ink2"], 12.5, "end")
        o += rbar_h(L, y, plot * a / mx, bh, t["s1"])
        o += txt(L + plot * a / mx + 8, y + bh - 4, f"{a:.1f}", t["ink2"], 12)
        y2 = y + bh + gap
        o += rbar_h(L, y2, plot * b / mx, bh, t["s2"])
        o += txt(L + plot * b / mx + 8, y2 + bh - 4, f"{b:.1f}", t["ink2"], 12)
        if name.startswith("draft-pass"):
            o += txt(L + plot * a / mx + 44, y + bh - 4, "-22.4 ms", t["ink"], 12, weight="600")
        y += bh * 2 + gap + pair + 6
    o += (f'<line x1="{L}" y1="{TOP:.1f}" x2="{L}" y2="{H-46:.1f}" stroke="{t["axis"]}" '
          f'stroke-width="1"/>')
    for g in range(0, 7):
        gx = L + plot * (g * 10) / mx
        o += txt(gx, H - 28, str(g * 10), t["muted"], 11.5, "middle")
    o += txt(L + plot / 2, H - 10, "milliseconds per decode step", t["muted"], 12, "middle")
    return o + "</svg>"


# --------------------------------------------------------------------------- figure 2
def fig_ablation(mode):
    t = THEME[mode]
    W, H = 820, 400
    L, R, TOP, BOT = 80, 40, 104, 66
    plot_h = H - TOP - BOT
    mx = 120.0
    steps = [("baseline", 0.0, 106.2, "total"), ("FR-Spec\ndraft head", 106.2, 86.3, "drop"),
             ("+ fp8 side\nlayers", 86.3, 72.5, "drop"), ("measured\ncombined", 0.0, 72.5, "total")]
    o = head(W, H, t, "Ablation of step time")
    o += titleblock(t, "The two optimisations compose additively",
                    "Step time on the code workload. The bars decompose the measured -33.7 ms; "
                    "measured independently the savings are -19.9 and -13.3 ms, summing to -33.2.",
                    W, sub_lines=2)
    n = len(steps)
    slot = (W - L - R) / n
    bw = min(96, slot - 34)

    def ypx(v):
        return TOP + plot_h * (1 - v / mx)

    for g in range(0, 7):
        gv = g * 20
        gy = ypx(gv)
        o += (f'<line x1="{L:.1f}" y1="{gy:.1f}" x2="{W-R:.1f}" y2="{gy:.1f}" '
              f'stroke="{t["grid"]}" stroke-width="1"/>')
        o += txt(L - 12, gy + 4, str(gv), t["muted"], 11.5, "end")
    for i, (lab, top, bot, kind) in enumerate(steps):
        cx = L + slot * i + slot / 2
        x = cx - bw / 2
        if kind == "total":
            o += rbar_v(x, ypx(bot), bw, plot_h * bot / mx, t["ramp"][2 if i == 0 else 3])
            o += txt(cx, ypx(bot) - 10, f"{bot:.1f} ms", t["ink"], 13.5, "middle", "600")
        else:
            o += rbar_v(x, ypx(top), bw, plot_h * (top - bot) / mx, t["s2"])
            o += txt(cx, ypx(top) - 10, f"-{top-bot:.1f}", t["s2"] if mode == "light" else t["s2"],
                     13, "middle", "600")
            o += (f'<line x1="{x-slot*0.16:.1f}" y1="{ypx(top):.1f}" x2="{x:.1f}" '
                  f'y2="{ypx(top):.1f}" stroke="{t["axis"]}" stroke-width="1" '
                  f'stroke-dasharray="3 3"/>')
            o += (f'<line x1="{x+bw:.1f}" y1="{ypx(bot):.1f}" x2="{x+bw+slot*0.16:.1f}" '
                  f'y2="{ypx(bot):.1f}" stroke="{t["axis"]}" stroke-width="1" '
                  f'stroke-dasharray="3 3"/>')
        for j, line in enumerate(lab.split("\n")):
            o += txt(cx, H - 40 + j * 14, line, t["ink2"], 12, "middle")
    o += (f'<line x1="{L:.1f}" y1="{TOP+plot_h:.1f}" x2="{W-R:.1f}" y2="{TOP+plot_h:.1f}" '
          f'stroke="{t["axis"]}" stroke-width="1"/>')
    o += txt(L - 12, TOP - 14, "ms/step", t["muted"], 11.5, "end")
    return o + "</svg>"


# --------------------------------------------------------------------------- figure 3
def fig_throughput(mode):
    t = THEME[mode]
    W, H = 820, 400
    L, R, TOP, BOT = 64, 40, 110, 62
    plot_h = H - TOP - BOT
    mx = 60.0
    cfgs = ["baseline", "FR-Spec", "fp8 side layers", "FR-Spec + fp8"]
    data = {"prose": [20.0, 24.8, 24.1, 30.9], "code": [30.4, 36.2, 36.7, 45.5],
            "verbatim copy": [35.8, 43.6, 41.5, 52.2]}
    o = head(W, H, t, "Single-stream throughput by configuration")
    o += titleblock(t, "Single-stream throughput, tokens per second",
                    "Paired measurements, same session, warm cache, greedy decoding. "
                    "Higher is better.", W)
    o += legend(L, 88, list(zip(cfgs, t["ramp"])), t)

    def ypx(v):
        return TOP + plot_h * (1 - v / mx)

    for g in range(0, 7):
        gv = g * 10
        gy = ypx(gv)
        o += (f'<line x1="{L:.1f}" y1="{gy:.1f}" x2="{W-R:.1f}" y2="{gy:.1f}" '
              f'stroke="{t["grid"]}" stroke-width="1"/>')
        o += txt(L - 12, gy + 4, str(gv), t["muted"], 11.5, "end")
    slot = (W - L - R) / len(data)
    bw = (slot - 70) / 4
    for i, (case, vals) in enumerate(data.items()):
        base = L + slot * i + 35
        for j, v in enumerate(vals):
            x = base + j * (bw + 2)
            o += rbar_v(x, ypx(v), bw, plot_h * v / mx, t["ramp"][j])
            o += txt(x + bw / 2, ypx(v) - 8, f"{v:.1f}", t["ink2"], 11.5, "middle")
        o += txt(L + slot * i + slot / 2, H - 34, case, t["ink"], 13, "middle", "600")
        gain = 100 * (vals[-1] / vals[0] - 1)
        o += txt(L + slot * i + slot / 2, H - 16, f"+{gain:.0f}% overall", t["muted"], 11.5,
                 "middle")
    o += (f'<line x1="{L:.1f}" y1="{TOP+plot_h:.1f}" x2="{W-R:.1f}" y2="{TOP+plot_h:.1f}" '
          f'stroke="{t["axis"]}" stroke-width="1"/>')
    o += txt(L - 12, TOP - 14, "tok/s", t["muted"], 11.5, "end")
    return o + "</svg>"


# --------------------------------------------------------------------------- figure 4
def fig_engram(mode):
    t = THEME[mode]
    W, H = 820, 330
    L, R, TOP, BOT = 190, 90, 106, 56
    plot = W - L - R
    mx = 45.0
    rows = [("cold cache, uniform rows", 31.4, 3.9), ("cold cache, skewed rows", 7.4, 4.5),
            ("fully warm", 3.0, 3.9)]
    o = head(W, H, t, "Engram row gather strategies")
    o += titleblock(t, "Engram gather: page faults vs explicit reads",
                    "Median time to fetch 180 random 160-byte rows from the 51 GB on-disk "
                    "n-gram table. Lower is better.", W)
    o += legend(L, 88, [("np.memmap fancy index (ships today)", t["s1"]),
                        ("os.pread, 8 threads", t["s2"])], t)
    bh, gap = 19, 2
    y = TOP + 6
    for g in range(0, 4):
        gx = L + plot * (g * 15) / mx
        if g:
            o += (f'<line x1="{gx:.1f}" y1="{TOP:.1f}" x2="{gx:.1f}" y2="{H-44:.1f}" '
                  f'stroke="{t["grid"]}" stroke-width="1"/>')
    for name, a, b in rows:
        o += txt(L - 14, y + bh + 1, name, t["ink2"], 12.5, "end")
        o += rbar_h(L, y, plot * a / mx, bh, t["s1"])
        o += txt(L + plot * a / mx + 8, y + bh - 5, f"{a:.1f} ms", t["ink2"], 12)
        o += rbar_h(L, y + bh + gap, plot * b / mx, bh, t["s2"])
        o += txt(L + plot * b / mx + 8, y + bh * 2 + gap - 5, f"{b:.1f} ms", t["ink2"], 12)
        y += bh * 2 + gap + 22
    o += txt(L + plot * 31.4 / mx - 6, TOP - 6, "6.7x", t["ink"], 12.5, "end", "600")
    o += (f'<line x1="{L}" y1="{TOP:.1f}" x2="{L}" y2="{H-44:.1f}" stroke="{t["axis"]}" '
          f'stroke-width="1"/>')
    for g in range(0, 4):
        gx = L + plot * (g * 15) / mx
        o += txt(gx, H - 26, str(g * 15), t["muted"], 11.5, "middle")
    o += txt(L + plot / 2, H - 9, "milliseconds per gather", t["muted"], 12, "middle")
    return o + "</svg>"


# --------------------------------------------------------------------------- figure 5
def fig_bandwidth(mode):
    t = THEME[mode]
    W, H = 820, 300
    L, R, TOP, BOT = 300, 90, 84, 52
    plot = W - L - R
    mx = 280.0
    rows = [("GPU reads device memory", 241, t["ramp"][3]),
            ("GPU reads pinned host memory", 77, t["ramp"][1]),
            ("cudaMemcpy host to device", 59, t["ramp"][0])]
    o = head(W, H, t, "Effective bandwidth by access path")
    o += titleblock(t, "\"Unified\" memory is not uniform",
                    "One physical LPDDR5X pool, three access paths. "
                    "cudaHostGetDevicePointer returns the identical pointer.", W)
    bh = 34
    y = TOP + 8
    for g in range(0, 5):
        gx = L + plot * (g * 70) / mx
        if g:
            o += (f'<line x1="{gx:.1f}" y1="{TOP:.1f}" x2="{gx:.1f}" y2="{H-40:.1f}" '
                  f'stroke="{t["grid"]}" stroke-width="1"/>')
    for name, v, col in rows:
        o += txt(L - 14, y + bh / 2 + 5, name, t["ink2"], 13, "end")
        o += rbar_h(L, y, plot * v / mx, bh, col)
        o += txt(L + plot * v / mx + 9, y + bh / 2 + 5, f"{v} GB/s", t["ink"], 13, "start", "600")
        y += bh + 12
    o += (f'<line x1="{L}" y1="{TOP:.1f}" x2="{L}" y2="{H-40:.1f}" stroke="{t["axis"]}" '
          f'stroke-width="1"/>')
    for g in range(0, 5):
        gx = L + plot * (g * 70) / mx
        o += txt(gx, H - 22, str(g * 70), t["muted"], 11.5, "middle")
    o += txt(L + plot / 2, H - 6, "effective bandwidth, GB/s", t["muted"], 12, "middle")
    return o + "</svg>"


FIGS = {"step-anatomy": fig_step_anatomy, "ablation": fig_ablation,
        "throughput": fig_throughput, "engram-gather": fig_engram, "bandwidth": fig_bandwidth}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in FIGS.items():
        for mode in ("light", "dark"):
            p = os.path.join(OUT, f"{name}-{mode}.svg")
            with open(p, "w", encoding="utf-8") as f:
                f.write(fn(mode))
            print("wrote", os.path.relpath(p, os.path.join(OUT, "..", "..")))
