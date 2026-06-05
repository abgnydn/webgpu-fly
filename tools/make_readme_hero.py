#!/usr/bin/env python3
"""Generate public/readme-hero.svg — the README banner.

A static connectome banner (GitHub strips SVG animation/scripts, so it must
render beautifully without motion): dark abyssal gradient, a neuron network
on the right with blur-glow, amber/violet spikes mid-edge, and the title +
stat chips on the left. Deterministic via a fixed-seed LCG so re-runs are
byte-stable. No third-party deps.

Run:  python3 tools/make_readme_hero.py
"""
import math

W, H = 1200, 380
SEED = 20260605


class R:
    """Tiny deterministic LCG so the banner never changes between runs."""
    def __init__(self, s): self.s = s & 0xFFFFFFFF
    def f(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF
    def rng(self, a, b): return a + (b - a) * self.f()


def build():
    r = R(SEED)
    # cluster centres in the right 2/3 — reads like brain regions
    centres = [(560, 150), (720, 250), (880, 120), (1010, 240), (660, 80)]
    nodes = []
    for cx, cy in centres:
        for _ in range(8):
            ang, rad = r.rng(0, 2 * math.pi), r.rng(8, 95)
            x = min(1185, max(430, cx + math.cos(ang) * rad))
            y = min(348, max(34, cy + math.sin(ang) * rad))
            nodes.append((x, y))
    # a few scattered loners
    for _ in range(6):
        nodes.append((r.rng(440, 1180), r.rng(40, 345)))

    # edges: each node to its 2 nearest neighbours, capped by distance
    edges = []
    for i, (xi, yi) in enumerate(nodes):
        d = sorted(
            ((j, (xi - xj) ** 2 + (yi - yj) ** 2) for j, (xj, yj) in enumerate(nodes) if j != i),
            key=lambda t: t[1],
        )
        for j, d2 in d[:2]:
            if d2 < 150 ** 2 and (min(i, j), max(i, j)) not in edges:
                edges.append((min(i, j), max(i, j)))

    # spikes: amber (excitatory) / violet (inhibitory) dots part-way along edges
    spikes = []
    for k in range(11):
        a, b = edges[int(r.rng(0, len(edges)))]
        t = r.rng(0.3, 0.7)
        x = nodes[a][0] + (nodes[b][0] - nodes[a][0]) * t
        y = nodes[a][1] + (nodes[b][1] - nodes[a][1]) * t
        spikes.append((x, y, "#ffc06b" if k % 4 else "#c08bff"))
    return nodes, edges, spikes


def svg():
    nodes, edges, spikes = build()
    P = []
    P.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="webgpu-fly — a real fly brain, spinal cord and body running in a browser tab">')
    P.append('<defs>')
    P.append('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#05070d"/><stop offset="1" stop-color="#080d16"/></linearGradient>')
    P.append('<radialGradient id="g1" cx="0.18" cy="0.1" r="0.6"><stop offset="0" stop-color="#7fe3ff" stop-opacity="0.16"/><stop offset="1" stop-color="#7fe3ff" stop-opacity="0"/></radialGradient>')
    P.append('<radialGradient id="g2" cx="0.85" cy="0.0" r="0.6"><stop offset="0" stop-color="#34e3c0" stop-opacity="0.16"/><stop offset="1" stop-color="#34e3c0" stop-opacity="0"/></radialGradient>')
    P.append('<filter id="blur" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="3.2"/></filter>')
    P.append('<filter id="soft" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="7"/></filter>')
    P.append('</defs>')
    # background + glows
    P.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    P.append(f'<rect width="{W}" height="{H}" fill="url(#g1)"/>')
    P.append(f'<rect width="{W}" height="{H}" fill="url(#g2)"/>')

    # edges
    P.append('<g stroke="#7fc8ff" stroke-opacity="0.12" stroke-width="1">')
    for a, b in edges:
        P.append(f'<line x1="{nodes[a][0]:.1f}" y1="{nodes[a][1]:.1f}" x2="{nodes[b][0]:.1f}" y2="{nodes[b][1]:.1f}"/>')
    P.append('</g>')

    # node glow halos
    P.append('<g filter="url(#soft)">')
    for x, y in nodes:
        P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#7fe3ff" fill-opacity="0.5"/>')
    P.append('</g>')
    # node cores
    P.append('<g fill="#bfeeff">')
    for x, y in nodes:
        P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.7"/>')
    P.append('</g>')

    # spikes (glow + core)
    for x, y, c in spikes:
        P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{c}" fill-opacity="0.55" filter="url(#blur)"/>')
        P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{c}"/>')

    # left-edge fade so the title sits on calm ground
    P.append('<rect width="520" height="' + str(H) + '" fill="url(#bg)" opacity="0.55"/>')

    # title block
    serif = "Georgia, 'Iowan Old Style', 'Times New Roman', serif"
    sans = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
    mono = "'SF Mono', ui-monospace, 'Cascadia Code', Menlo, monospace"
    P.append(f'<text x="64" y="122" font-family="{mono}" font-size="13" letter-spacing="3" fill="#34e3c0">A REAL FLY BRAIN · IN YOUR BROWSER</text>')
    P.append(f'<text x="60" y="196" font-family="{serif}" font-size="72" font-weight="700" fill="#eaf1f8">webgpu-<tspan font-style="italic" fill="#ffc06b">fly</tspan></text>')
    P.append(f'<text x="64" y="242" font-family="{sans}" font-size="20" fill="#aebccd">A real <tspan fill="#e6eef6">Drosophila</tspan> brain, spinal cord &amp; body —</text>')
    P.append(f'<text x="64" y="270" font-family="{sans}" font-size="20" fill="#aebccd">simulated live in a browser tab. No install.</text>')

    # stat chips
    chips = [("139,255 neurons", 64), ("WebGPU", 244), ("0 install", 360)]
    for label, x in chips:
        w = 16 + len(label) * 7.6
        P.append(f'<rect x="{x}" y="300" width="{w:.0f}" height="30" rx="15" fill="#0c141f" stroke="#7fc8ff" stroke-opacity="0.22"/>')
        P.append(f'<text x="{x + w / 2:.0f}" y="320" font-family="{mono}" font-size="12.5" fill="#9fb0c2" text-anchor="middle">{label}</text>')

    P.append('</svg>')
    return "\n".join(P) + "\n"


if __name__ == "__main__":
    out = "public/readme-hero.svg"
    with open(out, "w") as f:
        f.write(svg())
    print(f"wrote {out}")
