"""Reproducible paper-style schematic for the DLM pruning framework.

This draws a conceptual figure only; it does not run a model or fabricate results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
STEM = "dlm-pruning-framework-paper-2026-09-26"
INK = "#202833"
MUTED = "#607080"
RULE = "#CBD3DA"
BLUE = "#216A87"
BLUE_LIGHT = "#E7F1F5"
ORANGE = "#BD6438"
ORANGE_LIGHT = "#FAEEE8"
PALE = "#F4F6F7"
WHITE = "#FFFFFF"

font_manager.findfont("Noto Sans CJK KR", fallback_to_default=False)
plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK KR",
        "font.size": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

fig = plt.figure(figsize=(15, 8.7), facecolor=WHITE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1500)
ax.set_ylim(0, 870)
ax.axis("off")


def t(x, y, s, size=11, color=INK, weight="regular", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def line(x0, y0, x1, y1, color=RULE, lw=1.0, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="round")


def box(x, y, w, h, fc=WHITE, ec=RULE, lw=1, radius=8):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(patch)
    return patch


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.5, head=11):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=head,
            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
        )
    )


def panel(letter, title, x, y, w):
    t(x, y, f"({letter})", 12, BLUE, "bold")
    t(x + 38, y, title, 15, INK, "bold")
    line(x, y - 21, x + w, y - 21, RULE, 1.0)


def token(x, y, label, kind):
    colors = {
        "visible": (BLUE_LIGHT, BLUE),
        "masked": (PALE, RULE),
        "query": (ORANGE_LIGHT, ORANGE),
    }
    fc, ec = colors[kind]
    box(x, y, 42, 42, fc, ec, 1.3, 4)
    t(x + 21, y + 21, label, 11, ec if kind != "masked" else MUTED, "bold", "center")


# Figure header
t(50, 830, "FIGURE 1  |  METHOD OVERVIEW", 10, BLUE, "bold")
t(50, 791, "DLM 문맥 반응을 이용한 정적 가중치 pruning", 24, INK, "bold")
t(50, 756, "같은 query의 문맥 변화를 보존하도록 블록별 제거량을 배분한다", 12, MUTED)
line(50, 737, 1450, 737, INK, 1.3)

# (a) Calibration states
panel("a", "부분 공개 문맥 구성", 50, 704, 415)
t(61, 634, "초기 상태", 11, MUTED, "bold")
t(61, 542, "공개 후", 11, MUTED, "bold")
xs = [151 + i * 49 for i in range(6)]
initial = [("T", "visible"), ("M", "masked"), ("M", "masked"), ("Q", "query"), ("M", "masked"), ("T", "visible")]
later = [("T", "visible"), ("T", "visible"), ("T", "visible"), ("Q", "query"), ("M", "masked"), ("T", "visible")]
for x, (label, kind) in zip(xs, initial):
    token(x, 611, label, kind)
for x, (label, kind) in zip(xs, later):
    token(x, 519, label, kind)
arrow(299, 598, 299, 568, BLUE, 1.3, 10)
t(320, 583, "정답 문맥 공개", 10, BLUE)
t(151, 490, "Q: 끝까지 masked인 동일 질문", 10, ORANGE)
t(151, 472, "T: 보이는 토큰  ·  M: masked 토큰", 9.5, MUTED)

# (b) Dense vs sparse response on the same states
panel("b", "두 모델의 응답 비교", 505, 704, 430)
t(515, 665, "동일 문맥을 Dense와 Sparse에 입력", 10.5, MUTED)
x0, x1 = 570, 890
y_d0, y_d1 = 515, 625
y_s0, y_s1 = 548, 584
line(x0, 494, x1, 494, INK, 1.0)
line(x0, 494, x0, 635, INK, 1.0)
line(x0, y_d0, x1, y_d1, BLUE, 3.0)
line(x0, y_s0, x1, y_s1, ORANGE, 3.0)
for x, y in [(x0, y_d0), (x1, y_d1)]:
    ax.plot(x, y, "o", ms=7, color=BLUE, zorder=4)
for x, y in [(x0, y_s0), (x1, y_s1)]:
    ax.plot(x, y, "o", ms=7, color=ORANGE, zorder=4)
line(x0, y_d0, x0, y_s0, ORANGE, 1.3, "--")
line(x1, y_s1, x1, y_d1, ORANGE, 1.3, "--")
t(x0 + 14, 533, "$e_0$", 10, ORANGE)
t(x1 - 17, 607, "$e_1$", 10, ORANGE, ha="right")
t(x0, 476, "$x_0$", 10, INK, ha="center")
t(x1, 476, "$x_1$", 10, INK, ha="center")
line(597, 651, 625, 651, BLUE, 2.5)
t(633, 651, "Dense", 10, BLUE)
line(717, 651, 745, 651, ORANGE, 2.5)
t(753, 651, "Sparse", 10, ORANGE)
t(723, 460, "개념 예시 · 측정값 아님", 9, MUTED)

# (c) Objective
panel("c", "상태와 반응의 오차", 975, 704, 475)
t(991, 659, "$e_i = f_{S}(x_i)-f_{D}(x_i)$", 15, INK)
line(991, 638, 1438, 638, RULE, 0.8)
t(991, 602, "A", 19, BLUE, "bold")
t(1024, 602, r"$=\;\mathrm{mean}_{i}\;e_i^2$", 17, INK)
t(991, 577, "각 문맥에서 원본 응답을 보존", 10, MUTED)
t(991, 535, "C", 19, ORANGE, "bold")
t(1024, 535, r"$=\;\mathrm{mean}_{(i,j)\in E}\;(e_j-e_i)^2$", 16, INK)
t(991, 510, "문맥 변화에 따른 응답 갱신을 보존", 10, MUTED)
t(991, 472, r"$L = A + \lambda C$", 15, INK, "bold")
t(1160, 472, "$E$: 비교할 상태 쌍", 9.5, MUTED)

# Bottom: allocation pipeline
line(50, 438, 1450, 438, INK, 1.1)
panel("d", "목표 오차를 이용한 정확한 예산의 정적 마스크 선택", 50, 411, 1400)

# Step 1: fixed within-row Wanda ranking
t(70, 355, "1", 11, BLUE, "bold")
t(90, 355, "Wanda 순위 고정", 13, INK, "bold")
t(70, 332, "각 행에서 제거할 순서는 유지", 10, MUTED)
for r in range(3):
    for c in range(7):
        col = INK if c < 3 + (r == 0) else WHITE
        ax.add_patch(Rectangle((91 + c * 24, 242 + r * 23), 18, 16, facecolor=col, edgecolor=RULE, lw=0.8))
t(88, 221, "■ 제거  □ 유지", 9.5, MUTED)

arrow(291, 279, 367, 279, BLUE, 1.6, 13)

# Step 2: block probes
t(383, 355, "2", 11, BLUE, "bold")
t(403, 355, "블록별 변화 측정", 13, INK, "bold")
t(383, 332, "50% 기준에서 한 블록만 48/52%", 10, MUTED)
box(393, 235, 254, 71, PALE, RULE, 0.9, 5)
for y, label, frac, color in [(280, "48%", .48, BLUE), (251, "52%", .52, ORANGE)]:
    t(408, y, label, 9.5, MUTED)
    ax.add_patch(Rectangle((455, y - 7), 166, 13, facecolor=WHITE, edgecolor=RULE, lw=.8))
    ax.add_patch(Rectangle((455, y - 7), 166 * frac, 13, facecolor=color, edgecolor="none"))
t(390, 219, "동일 calibration bank에서 $A$, $C$ 측정", 9.5, MUTED)

arrow(665, 279, 739, 279, BLUE, 1.6, 13)

# Step 3: allocate exact quota
t(756, 355, "3", 11, BLUE, "bold")
t(776, 355, "블록별 제거량 배분", 13, INK, "bold")
t(756, 332, "오차 기반 순위 → 층별 sparsity", 10, MUTED)
bar_x = 780
heights = [56, 75, 62, 91, 53, 69, 82, 58]
for i, h in enumerate(heights):
    ax.add_patch(Rectangle((bar_x + 25 * i, 232), 16, h, facecolor=BLUE if i != 3 else ORANGE, edgecolor="none"))
line(776, 230, 1001, 230, INK, .8)
t(776, 211, "합계는 정확히 전체 50% 예산", 9.5, MUTED)

arrow(1026, 279, 1096, 279, BLUE, 1.6, 13)

# Step 4: final model and evaluation
t(1110, 355, "4", 11, BLUE, "bold")
t(1130, 355, "하나의 정적 마스크", 13, INK, "bold")
t(1110, 332, "모든 denoising 상태에 공통 적용", 10, MUTED)
box(1120, 258, 310, 56, BLUE_LIGHT, BLUE, 1.1, 5)
t(1275, 286, "완성된 Sparse DLM", 12, BLUE, "bold", ha="center")
arrow(1275, 255, 1275, 229, BLUE, 1.2, 10)
box(1120, 174, 310, 52, WHITE, RULE, 1.0, 5)
t(1275, 200, "별도 문항에서 생성 성능 평가", 10.5, INK, ha="center")

line(50, 141, 1450, 141, RULE, 1.0)
t(50, 115, "방법별 변형", 10, BLUE, "bold")
t(153, 115, "A-only: C 제외    ·    Short / Path / All / Multi: 상태 쌍 $E$ 변경    ·    Vector: 응답 $f$ 변경    ·    Exchange: 배분 탐색 변경", 10, INK)
t(50, 77, "실선 흐름은 기존 scalar A+C의 공통 골격. 변형은 각 한 요소를 바꾸는 연구 후보이며, 이 그림은 성능 우열을 나타내지 않는다.", 9.5, MUTED)
t(50, 47, "D: dense model   S: sparse model   Q: fixed masked query   48/52%: 기존 50% block probe 예시", 9, MUTED)

fig.savefig(OUT / f"{STEM}.svg", facecolor=WHITE, bbox_inches=None)
fig.savefig(OUT / f"{STEM}.png", dpi=200, facecolor=WHITE, bbox_inches=None)

svg = (OUT / f"{STEM}.svg").read_text(encoding="utf-8")
svg = svg[svg.index("<svg ") :]
svg = svg.replace("<svg ", '<svg role="img" aria-labelledby="paper-figure-title paper-figure-desc" ', 1)
svg = svg.replace(
    ">\n <metadata>",
    '><title id="paper-figure-title">DLM 문맥 반응 기반 정적 가중치 pruning</title>'
    '<desc id="paper-figure-desc">부분 공개 문맥을 동일한 질문에 대해 만들고, 원본과 sparse 모델의 응답 오차 A와 변화 오차 C를 측정하여 블록별 제거량을 배분하고 하나의 정적 마스크를 평가하는 논문 형식의 방법론 도식.</desc>\n <metadata>',
    1,
)
html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DLM pruning framework — paper figure</title>
<style>body{{margin:0;background:#fff;color:#202833;font-family:'Noto Sans CJK KR',sans-serif}}
main{{max-width:1600px;margin:0 auto;padding:24px}}svg{{display:block;width:100%;height:auto}}
p{{font-size:13px;line-height:1.6;color:#607080;margin:12px 2px}}</style></head>
<body><main>{svg}<p>Figure 1. 같은 masked query에서 문맥을 공개하며 얻은 응답을 Dense와 Sparse 모델에 공통으로 측정한다. A는 각 상태의 오차, C는 상태 사이의 응답 변화 오차다. 기존 scalar A+C 구현은 고정된 Wanda 순위와 block probe를 이용해 정확한 전체 예산의 정적 마스크를 만든다.</p></main></body></html>"""
(OUT / f"{STEM}.html").write_text(html, encoding="utf-8")
