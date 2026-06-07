import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

C = {
    "bg":       "#FFFFFF",
    "init_fc":  "#E8F4F8",
    "init_ec":  "#1A7A8A",
    "loop_fc":  "#EDE7F6",
    "loop_ec":  "#5E35B1",
    "cond_fc":  "#E8F5E9",
    "cond_ec":  "#2E7D32",
    "res_fc":   "#FFF8E1",
    "res_ec":   "#F57F17",
    "start_fc": "#E53935",
    "arrow":    "#607D8B",
    "accent":   "#1565C0",
    "txt":      "#1A1A2A",
    "dim":      "#546E7A",
    "loop_lbl": "#5E35B1",
}

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

def box(ax, cx, cy, w, h, fc, ec, line1, line2=None, line3=None, fs1=8.5, fs2=7, fs3=8):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=fc, edgecolor=ec,
        linewidth=1.5, zorder=3))
    
    txt_color = "#FFFFFF" if fc == C["start_fc"] else C["txt"]
    
    if line3:
        ax.text(cx, cy + h*0.28, line1, ha="center", va="center",
                color=txt_color, fontsize=fs1, fontfamily=SANS, fontweight="bold", zorder=4)
        ax.text(cx, cy + h*0.02, line2, ha="center", va="center",
                color=C["dim"], fontsize=fs2, fontfamily=SANS, style="italic", zorder=4)
        ax.text(cx, cy - h*0.25, line3, ha="center", va="center",
                color=txt_color, fontsize=fs3, zorder=4)
    elif line2:
        ax.text(cx, cy + h*0.18, line1, ha="center", va="center",
                color=txt_color, fontsize=fs1, fontfamily=SANS, fontweight="bold", zorder=4)
        ax.text(cx, cy - h*0.18, line2, ha="center", va="center",
                color=C["dim"], fontsize=fs2, fontfamily=SANS, style="italic", zorder=4)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                color=txt_color, fontsize=fs1+1, fontfamily=SANS, fontweight="bold", zorder=4)

def diamond(ax, cx, cy, w, h, fc, ec, line1, line2=None):
    pts = np.array([[cx, cy+h/2],[cx+w/2, cy],[cx, cy-h/2],[cx-w/2, cy]])
    ax.add_patch(plt.Polygon(pts, closed=True,
                              facecolor=fc, edgecolor=ec,
                              linewidth=1.5, zorder=3))
    if line2:
        ax.text(cx, cy + h*0.2,  line1, ha="center", va="center",
                color=C["txt"], fontsize=8.5, fontfamily=SANS, fontweight="bold", zorder=4)
        ax.text(cx, cy - h*0.22, line2, ha="center", va="center",
                color=C["dim"], fontsize=7, fontfamily=SANS, style="italic", zorder=4)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                color=C["txt"], fontsize=8.5, fontfamily=SANS, fontweight="bold", zorder=4)

def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=1.3, mutation_scale=11), zorder=2)

def polyline(ax, xs, ys):
    ax.plot(xs, ys, color=C["arrow"], lw=1.3, zorder=2)

def lbl(ax, x, y, text, ha="left"):
    ax.text(x, y, text, ha=ha, va="center",
            color=C["accent"], fontsize=8, fontfamily=SANS, fontweight="bold", zorder=5)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

BW = 3.4
BH = 0.95
DW = 3.1
DH = 0.75

C1, C2, C3 = 1.8, 5.5, 9.2

ax.text(5.5, 8.72, "Algorytm ADAM  —  jak to działa?",
        ha="center", va="center",
        color=C["txt"], fontsize=14, fontfamily=MONO, fontweight="bold")
ax.plot([0.4, 10.6], [8.45, 8.45], color="#CFD8DC", lw=1)

R1 = 7.9
box(ax, C1, R1, 1.6, BH, C["start_fc"], C["start_fc"], "START")
arr(ax, C1+1.6/2, R1, C2-BW/2, R1)

box(ax, C2, R1, BW, BH, C["init_fc"], C["init_ec"],
    "Konfiguracja", "Ustaw α, β₁, β₂, ε i punkt startowy")
arr(ax, C2+BW/2, R1, C3-BW/2, R1)

box(ax, C3, R1, BW, BH, C["init_fc"], C["init_ec"],
    "Wyzeruj stan wewnętrzny", "Licznik kroków = 0, pamięć gradientów = 0")

R2 = 6.5
arr(ax, C3, R1-BH/2, C3, R2+BH/2)

box(ax, C3, R2, BW, BH, C["loop_fc"], C["loop_ec"],
    "Kolejny krok", "Zwiększ licznik iteracji o 1", r"$t \leftarrow t + 1$")
arr(ax, C3-BW/2, R2, C2+BW/2, R2)

box(ax, C2, R2, BW, BH, C["loop_fc"], C["loop_ec"],
    "Oblicz gradient", "Wyznacz gradient funkcji w bieżącym punkcie", r"$g_t = \nabla f(\theta_{t-1})$")
arr(ax, C2-BW/2, R2, C1+BW/2, R2)

box(ax, C1, R2, BW, BH, C["loop_fc"], C["loop_ec"],
    "Zaktualizuj pamięć", "Uśrednij bieżące i przeszłe gradienty",
    r"$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$" + "\n" + r"$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$", fs3=7.5)

R3 = 5.1
arr(ax, C1, R2-BH/2, C1, R3+BH/2)

box(ax, C1, R3, BW, BH, C["loop_fc"], C["loop_ec"],
    "Skoryguj niedoszacowanie", "Zniweluj błąd systematyczny (bias)",
    r"$\hat{m}_t = m_t / (1-\beta_1^t) \quad \hat{v}_t = v_t / (1-\beta_2^t)$")
arr(ax, C1+BW/2, R3, C2-BW/2, R3)

box(ax, C2, R3, BW, BH, C["loop_fc"], C["loop_ec"],
    "Przesuń punkt w przestrzeni", "Krok zgodnie z poprawionym kierunkiem",
    r"$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$")
arr(ax, C2+BW/2, R3, C3-BW/2, R3)

box(ax, C3, R3, BW, BH, C["loop_fc"], C["loop_ec"],
    "Zapisz stan do historii", "Zapamiętaj pozycję i wartość funkcji", r"$f(\theta_t)$")

R4 = 3.7
arr(ax, C3, R3-BH/2, C3, R4+DH/2)

diamond(ax, C3, R4, DW, DH, C["cond_fc"], C["cond_ec"], "Warunek stopu?")

R5 = 2.3
arr(ax, C3, R4-DH/2, C3, R5+BH/2)
lbl(ax, C3 + 0.15, R4 - DH/2 - 0.2, "TAK", ha="left")

nie_x = C3 - DW/2
gap_x = 7.35 

polyline(ax,
         [nie_x, gap_x, gap_x, C3],
         [R4,    R4,    R2 + BH/2 + 0.35, R2 + BH/2 + 0.35])
arr(ax, C3, R2 + BH/2 + 0.35, C3, R2 + BH/2)
lbl(ax, (nie_x + gap_x)/2, R4 + 0.12, "NIE", ha="center")

box(ax, C3, R5, BW, BH, C["res_fc"], C["res_ec"],
    "Zwróć najlepszy wynik", "Optymalny punkt i wartość funkcji celu")

loop_rect = FancyBboxPatch(
    (C1-BW/2-0.2, R3-BH/2-0.15),
    (C3+BW/2+0.2)-(C1-BW/2-0.2),
    (R2+BH/2+0.15)-(R3-BH/2-0.15),
    boxstyle="round,pad=0.04",
    facecolor="none", edgecolor=C["loop_lbl"],
    linewidth=0.8, linestyle="--", zorder=1, alpha=0.4)
ax.add_patch(loop_rect)
ax.text(C1-BW/2-0.15, R2+BH/2+0.08, "PĘTLA",
        ha="left", va="bottom",
        color=C["loop_lbl"], fontsize=6.5,
        fontfamily=MONO, fontweight="bold", alpha=0.7)


plt.tight_layout(pad=0.15)
plt.savefig("adam_flowchart_final.png", dpi=192, bbox_inches="tight", facecolor="white")
print("Zapisano: adam_flowchart_final.png")