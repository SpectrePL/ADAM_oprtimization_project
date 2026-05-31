import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C = {
    "bg":       "#FFFFFF",
    "arrow":    "#546E7A",
    "txt":      "#1A1A2A",
    "lbl_txt":  "#37474F",
    
    # Kolory dla bloku Danych (XML, Parser)
    "data_fc":  "#E3F2FD",
    "data_ec":  "#1565C0",
    
    # Kolory dla logiki Głównej (Pętla, Optymalizator, Klasy)
    "core_fc":  "#F3E5F5",
    "core_ec":  "#6A1B9A",
    
    # Kolory dla Wizualizacji
    "vis_fc":   "#FFF3E0",
    "vis_ec":   "#E65100",
}

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

def box(ax, cx, cy, w, h, text, fc, ec):
    # Rysuje blok z podanymi kolorami tła (fc) i ramki (ec)
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=ec,
        linewidth=1.5, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center",
            color=C["txt"], fontsize=10, fontfamily=SANS, fontweight="bold", zorder=4)

def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=1.5, mutation_scale=12), zorder=2)

def text_label(ax, x, y, text, ha="center", va="center"):
    # Etykiety z białym tłem, by linia strzałki ich nie przekreślała
    ax.text(x, y, text, ha=ha, va=va,
            color=C["lbl_txt"], fontsize=9, fontfamily=SANS, fontweight="bold",
            bbox=dict(facecolor=C["bg"], edgecolor='none', pad=2), zorder=5)

# ── canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# Wymiary bloków
BW = 2.4
BH = 1.4

# Współrzędne X dla osi poziomej
X_XML = 1.8
X_PAR = 5.8
X_GLO = 10.2
X_WIZ = 14.2

# Współrzędne Y dla trzech rzędów
Y_MID = 4.5
Y_TOP = 7.5
Y_BOT = 1.5

# ── rysowanie bloków z podziałem na kolory ────────────────────────────────────

# Grupa 1: Dane (Niebieski)
box(ax, X_XML, Y_MID, BW, BH, "XML", C["data_fc"], C["data_ec"])
box(ax, X_PAR, Y_MID, BW, BH, "Parser", C["data_fc"], C["data_ec"])

# Grupa 2: Silnik optymalizacji (Fioletowy)
box(ax, X_GLO, Y_MID, BW, BH, "Główna\npętla", C["core_fc"], C["core_ec"])
box(ax, X_GLO, Y_TOP, BW, BH, "Optymalizator ADAM", C["core_fc"], C["core_ec"])
box(ax, X_GLO, Y_BOT, BW, BH, "Klasy funkcji\ni ograniczeń", C["core_fc"], C["core_ec"])

# Grupa 3: Wyjście / Wizualizacja (Pomarańczowy)
box(ax, X_WIZ, Y_MID, BW, BH, "Wizualizacja", C["vis_fc"], C["vis_ec"])


# ── rysowanie strzałek i etykiet ──────────────────────────────────────────────

# 1. XML -> Parser
arr(ax, X_XML + BW/2, Y_MID, X_PAR - BW/2, Y_MID)
text_label(ax, (X_XML + X_PAR)/2, Y_MID + 0.25, "dane", va="bottom")

# 2. Parser -> Główna pętla
arr(ax, X_PAR + BW/2, Y_MID, X_GLO - BW/2, Y_MID)
text_label(ax, (X_PAR + X_GLO)/2, Y_MID + 0.25, "parametry optymalizacji", va="bottom")

# 3. Główna pętla -> Wizualizacja
arr(ax, X_GLO + BW/2, Y_MID, X_WIZ - BW/2, Y_MID)
text_label(ax, (X_GLO + X_WIZ)/2, Y_MID + 0.25, "trajektorie", va="bottom")

# 4. Główna pętla -> Optymalizator
arr(ax, X_GLO - 0.3, Y_MID + BH/2, X_GLO - 0.3, Y_TOP - BH/2)
text_label(ax, X_GLO - 0.45, (Y_MID + Y_TOP)/2, "obiekt funkcji\ni parametry", ha="right")

# 5. Optymalizator -> Główna pętla
arr(ax, X_GLO + 0.3, Y_TOP - BH/2, X_GLO + 0.3, Y_MID + BH/2)
text_label(ax, X_GLO + 0.45, (Y_MID + Y_TOP)/2, "wyniki i\ntrajektorie", ha="left")

# 6. Klasy funkcji -> Główna pętla
arr(ax, X_GLO, Y_BOT + BH/2, X_GLO, Y_MID - BH/2)
text_label(ax, X_GLO + 0.2, (Y_BOT + Y_MID)/2, "Wybrany obiekt", ha="left")

# ── dekoracje ─────────────────────────────────────────────────────────────────
ax.text(8.0, 8.6, "Architektura kodu",
        ha="center", va="center",
        color=C["txt"], fontsize=14, fontfamily=MONO, fontweight="bold")
ax.plot([0.4, 15.6], [8.3, 8.3], color="#CFD8DC", lw=1)

plt.tight_layout(pad=0.15)
plt.savefig("architektura_flowchart_kolory.png", dpi=192, bbox_inches="tight", facecolor=C["bg"])
print("Zapisano: architektura_flowchart_kolory.png")
