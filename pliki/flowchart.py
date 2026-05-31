"""
Skrypt generujący flowchart pipeline'u algorytmu ADAM
na podstawie implementacji z plików ADAM.py, funkcje.py i main.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─── Konfiguracja kolorystyki ────────────────────────────────────────────────
KOLORY = {
    "tlo":           "#0D1117",
    "siatka":        "#161B22",
    "start_stop":    "#58A6FF",
    "init":          "#3FB950",
    "obliczenia":    "#F0883E",
    "warunek":       "#BC8CFF",
    "zapis":         "#56D364",
    "wynik":         "#E3B341",
    "tekst_ciemny":  "#0D1117",
    "tekst_jasny":   "#E6EDF3",
    "strzalka":      "#8B949E",
    "akcent":        "#1F6FEB",
    "ramka":         "#30363D",
}

FONT_MONO  = "DejaVu Sans Mono"
FONT_SANS  = "DejaVu Sans"

# ─── Pomocnicze funkcje rysowania ────────────────────────────────────────────

def ramka(ax, x, y, w, h, kolor_wypelnienia, tekst,
          styl="round,pad=0.08", tekst_kolor=None,
          rozmiar_czcionki=8.5, bold=False):
    """Rysuje zaokrągloną ramkę z tekstem."""
    if tekst_kolor is None:
        tekst_kolor = KOLORY["tekst_ciemny"]

    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=styl,
        facecolor=kolor_wypelnienia,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, tekst,
        ha="center", va="center",
        color=tekst_kolor,
        fontsize=rozmiar_czcionki,
        fontfamily=FONT_SANS,
        fontweight="bold" if bold else "normal",
        zorder=4,
        wrap=True,
    )
    return box


def romb(ax, x, y, w, h, kolor_wypelnienia, tekst, rozmiar_czcionki=8):
    """Rysuje romb (węzeł decyzyjny)."""
    pts = np.array([
        [x,         y + h / 2],
        [x + w / 2, y],
        [x,         y - h / 2],
        [x - w / 2, y],
    ])
    poly = plt.Polygon(pts, closed=True,
                       facecolor=kolor_wypelnienia,
                       edgecolor="white", linewidth=0.8, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, tekst, ha="center", va="center",
            color=KOLORY["tekst_ciemny"],
            fontsize=rozmiar_czcionki,
            fontfamily=FONT_SANS, fontweight="bold", zorder=4)


def strzalka(ax, x1, y1, x2, y2, etykieta=None, kolor=None):
    """Rysuje strzałkę między dwoma punktami."""
    if kolor is None:
        kolor = KOLORY["strzalka"]
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=kolor,
            lw=1.4,
            mutation_scale=12,
        ),
        zorder=2,
    )
    if etykieta:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, etykieta,
                ha="left", va="center",
                color=KOLORY["tekst_jasny"],
                fontsize=7.5,
                fontfamily=FONT_SANS,
                zorder=5)


def linia(ax, xs, ys, kolor=None):
    """Rysuje łamaną linię (bez grotu)."""
    if kolor is None:
        kolor = KOLORY["strzalka"]
    ax.plot(xs, ys, color=kolor, lw=1.4, zorder=2)


# ─── Główna funkcja rysująca ─────────────────────────────────────────────────

def rysuj_flowchart():
    fig, ax = plt.subplots(figsize=(13, 22))
    fig.patch.set_facecolor(KOLORY["tlo"])
    ax.set_facecolor(KOLORY["tlo"])
    ax.set_xlim(-1.5, 8.5)
    ax.set_ylim(-1.5, 26.5)
    ax.axis("off")

    # Subtelna siatka
    for gy in np.arange(0, 27, 1):
        ax.axhline(gy, color=KOLORY["siatka"], lw=0.4, zorder=0)
    for gx in np.arange(-1, 9, 1):
        ax.axvline(gx, color=KOLORY["siatka"], lw=0.4, zorder=0)

    cx = 3.5   # oś główna X
    W  = 3.8   # szerokość bloków prostokątnych
    H  = 0.65  # wysokość bloków

    # ── Tytuł ────────────────────────────────────────────────────────────────
    ax.text(cx, 26.0, "ADAM OPTIMIZER",
            ha="center", va="center",
            color=KOLORY["start_stop"],
            fontsize=20, fontfamily=FONT_MONO,
            fontweight="bold", zorder=5)
    ax.text(cx, 25.4, "Pipeline — diagram przepływu algorytmu",
            ha="center", va="center",
            color=KOLORY["strzalka"],
            fontsize=9, fontfamily=FONT_SANS, zorder=5)
    ax.plot([0.2, 6.8], [25.1, 25.1],
            color=KOLORY["ramka"], lw=1.0, zorder=2)

    # ── Blok 1 — START ───────────────────────────────────────────────────────
    y1 = 24.2
    ramka(ax, cx, y1, W, H, KOLORY["start_stop"],
          "START", styl="round,pad=0.12", bold=True, rozmiar_czcionki=10)
    strzalka(ax, cx, y1 - H/2, cx, y1 - H/2 - 0.35)

    # ── Blok 2 — Inicjalizacja funkcji celu ──────────────────────────────────
    y2 = 23.1
    ramka(ax, cx, y2, W, H, KOLORY["init"],
          "Inicjalizacja funkcji celu\n(function_template / Funkcja_Z_Kara)",
          rozmiar_czcionki=8)
    strzalka(ax, cx, y2 - H/2, cx, y2 - H/2 - 0.35)

    # ── Blok 3 — Inicjalizacja ADAM ──────────────────────────────────────────
    y3 = 22.0
    ramka(ax, cx, y3, W, H * 1.55, KOLORY["init"],
          "Inicjalizacja ADAM\nα, β₁, β₂, ε, max_iter\n"
          "x ← start_point,  m=0,  v=0,  t=0",
          rozmiar_czcionki=8)
    strzalka(ax, cx, y3 - H * 1.55/2, cx, y3 - H * 1.55/2 - 0.35)

    # ── Blok 4 — Pętla: t ← t + 1 ───────────────────────────────────────────
    y4 = 20.5
    ramka(ax, cx, y4, W, H, KOLORY["obliczenia"],
          "t  ←  t + 1", rozmiar_czcionki=9, bold=True)
    strzalka(ax, cx, y4 - H/2, cx, y4 - H/2 - 0.35)

    # ── Blok 5 — Gradient ────────────────────────────────────────────────────
    y5 = 19.65
    ramka(ax, cx, y5, W, H, KOLORY["obliczenia"],
          "g  ←  ∇f(x)   [grad_val(x)]",
          rozmiar_czcionki=8.5)
    strzalka(ax, cx, y5 - H/2, cx, y5 - H/2 - 0.35)

    # ── Blok 6 — Moment rzędu 1 ──────────────────────────────────────────────
    y6 = 18.8
    ramka(ax, cx, y6, W, H, KOLORY["obliczenia"],
          "m  ←  β₁·m + (1−β₁)·g",
          rozmiar_czcionki=8.5)
    strzalka(ax, cx, y6 - H/2, cx, y6 - H/2 - 0.35)

    # ── Blok 7 — Moment rzędu 2 ──────────────────────────────────────────────
    y7 = 17.95
    ramka(ax, cx, y7, W, H, KOLORY["obliczenia"],
          "v  ←  β₂·v + (1−β₂)·g²",
          rozmiar_czcionki=8.5)
    strzalka(ax, cx, y7 - H/2, cx, y7 - H/2 - 0.35)

    # ── Blok 8 — Korekcja biasu ──────────────────────────────────────────────
    y8 = 17.1
    ramka(ax, cx, y8, W, H, KOLORY["obliczenia"],
          "m̂  ←  m / (1−β₁ᵗ)     v̂  ←  v / (1−β₂ᵗ)",
          rozmiar_czcionki=8)
    strzalka(ax, cx, y8 - H/2, cx, y8 - H/2 - 0.35)

    # ── Blok 9 — Aktualizacja parametrów ─────────────────────────────────────
    y9 = 16.2
    ramka(ax, cx, y9, W, H, KOLORY["obliczenia"],
          "x  ←  x − α · m̂ / (√v̂ + ε)",
          rozmiar_czcionki=8.5)
    strzalka(ax, cx, y9 - H/2, cx, y9 - H/2 - 0.35)

    # ── Blok 10 — Zapis historii ─────────────────────────────────────────────
    y10 = 15.3
    ramka(ax, cx, y10, W, H, KOLORY["zapis"],
          "Zapis: historia_pozycji[t],  historia_wartosci[t]",
          rozmiar_czcionki=7.8)
    strzalka(ax, cx, y10 - H/2, cx, y10 - H/2 - 0.35)

    # ── Warunek 1 — t > brak_zmian_maxiter ───────────────────────────────────
    y11 = 14.35
    RD_W, RD_H = 4.0, 0.72
    romb(ax, cx, y11, RD_W, RD_H,
         KOLORY["warunek"], "t  >  brak_zmian_maxiter ?",
         rozmiar_czcionki=8)
    strzalka(ax, cx, y11 - RD_H/2, cx, y11 - RD_H/2 - 0.35,
             etykieta="TAK")
    # NIE → strzałka w prawo i w górę (pętla)
    ax.text(cx + RD_W/2 + 0.08, y11, "NIE",
            ha="left", va="center",
            color=KOLORY["tekst_jasny"], fontsize=7.5, fontfamily=FONT_SANS)
    linia(ax,
          [cx + RD_W/2, cx + RD_W/2 + 0.55, cx + RD_W/2 + 0.55],
          [y11,          y11,                  y4 + H/2])
    strzalka(ax, cx + RD_W/2 + 0.55, y4 + H/2, cx + W/2, y4 + H/2)

    # ── Warunek 2 — brak zmian w ostatnich N iteracjach ──────────────────────
    y12 = 13.3
    romb(ax, cx, y12, RD_W, RD_H,
         KOLORY["warunek"],
         "Δf < ε_stop przez N iter?",
         rozmiar_czcionki=8)
    strzalka(ax, cx, y12 - RD_H/2, cx, y12 - RD_H/2 - 0.35,
             etykieta="TAK")
    # NIE → pętla powrót do t←t+1
    ax.text(cx - RD_W/2 - 0.08, y12, "NIE",
            ha="right", va="center",
            color=KOLORY["tekst_jasny"], fontsize=7.5, fontfamily=FONT_SANS)
    linia(ax,
          [cx - RD_W/2, cx - RD_W/2 - 0.55, cx - RD_W/2 - 0.55],
          [y12,           y12,                  y4 + H/2])
    strzalka(ax, cx - RD_W/2 - 0.55, y4 + H/2, cx - W/2, y4 + H/2)

    # ── Warunek 3 — t >= max_iter ─────────────────────────────────────────────
    y13 = 12.25
    romb(ax, cx, y13, RD_W, RD_H,
         KOLORY["warunek"], "t  ≥  max_iter ?",
         rozmiar_czcionki=8)
    strzalka(ax, cx, y13 - RD_H/2, cx, y13 - RD_H/2 - 0.35,
             etykieta="TAK")
    # NIE → powrót do pętli
    ax.text(cx + RD_W/2 + 0.08, y13, "NIE",
            ha="left", va="center",
            color=KOLORY["tekst_jasny"], fontsize=7.5, fontfamily=FONT_SANS)
    linia(ax,
          [cx + RD_W/2, cx + RD_W/2 + 0.85, cx + RD_W/2 + 0.85],
          [y13,           y13,                   y4 + H/2])
    strzalka(ax, cx + RD_W/2 + 0.85, y4 + H/2, cx + W/2, y4 + H/2)

    # ── Blok wynikowy — zwrot x*, f(x*) ─────────────────────────────────────
    y14 = 11.0
    ramka(ax, cx, y14, W, H, KOLORY["wynik"],
          "Zwróć:  x*,  f(x*)   [optymalizuj() → return]",
          rozmiar_czcionki=8, bold=True)
    strzalka(ax, cx, y14 - H/2, cx, y14 - H/2 - 0.35)

    # ── Sekcja wizualizacji ───────────────────────────────────────────────────
    y15 = 10.1
    ramka(ax, cx, y15, W, H * 0.8, KOLORY["ramka"],
          "Opcjonalna wizualizacja wyników",
          tekst_kolor=KOLORY["tekst_jasny"],
          rozmiar_czcionki=8, bold=True)
    strzalka(ax, cx, y15 - H * 0.8/2, cx, y15 - H * 0.8/2 - 0.35)

    # Rozgałęzienie na dwa wykresy
    y_fork = y15 - H * 0.8/2 - 0.35
    CX_L = cx - 1.6
    CX_R = cx + 1.6
    W2 = 2.8

    # linia pozioma
    linia(ax, [CX_L, CX_R], [y_fork, y_fork])

    y16 = 8.7
    strzalka(ax, CX_L, y_fork, CX_L, y16 + H/2)
    strzalka(ax, CX_R, y_fork, CX_R, y16 + H/2)

    ramka(ax, CX_L, y16, W2, H, KOLORY["akcent"],
          "wykres_fval()\nf(x) vs iteracja [plt]",
          tekst_kolor="#FFFFFF", rozmiar_czcionki=7.5)
    ramka(ax, CX_R, y16, W2, H, KOLORY["akcent"],
          "wykres_sciezka_3d()\nścieżka na pow. f [plotly]",
          tekst_kolor="#FFFFFF", rozmiar_czcionki=7.5)

    # Zbieżność strzałek
    strzalka(ax, CX_L, y16 - H/2, CX_L, y16 - H/2 - 0.35)
    strzalka(ax, CX_R, y16 - H/2, CX_R, y16 - H/2 - 0.35)
    y_join = y16 - H/2 - 0.35
    linia(ax, [CX_L, CX_R], [y_join, y_join])
    strzalka(ax, cx, y_join, cx, y_join - 0.35)

    # ── STOP ────────────────────────────────────────────────────────────────
    y17 = y_join - 0.35 - H/2
    ramka(ax, cx, y17, W * 0.6, H, KOLORY["start_stop"],
          "STOP", styl="round,pad=0.12", bold=True, rozmiar_czcionki=10)

    # ── Legenda ──────────────────────────────────────────────────────────────
    legenda_x = 6.5
    legenda_y_start = 23.5
    pozycje = [
        (KOLORY["start_stop"], "Punkt graniczny (START / STOP)"),
        (KOLORY["init"],       "Inicjalizacja"),
        (KOLORY["obliczenia"], "Obliczenia ADAM"),
        (KOLORY["warunek"],    "Warunek stopu"),
        (KOLORY["zapis"],      "Zapis stanu"),
        (KOLORY["wynik"],      "Wynik / zwrot"),
        (KOLORY["akcent"],     "Wizualizacja"),
    ]
    ax.text(legenda_x, legenda_y_start + 0.45, "LEGENDA",
            ha="center", va="center",
            color=KOLORY["tekst_jasny"],
            fontsize=8, fontfamily=FONT_MONO, fontweight="bold")
    ax.plot([5.55, 7.45], [legenda_y_start + 0.2, legenda_y_start + 0.2],
            color=KOLORY["ramka"], lw=0.8)
    for i, (kol, opis) in enumerate(pozycje):
        yy = legenda_y_start - i * 0.55
        rect = FancyBboxPatch((5.6, yy - 0.16), 0.35, 0.32,
                              boxstyle="round,pad=0.04",
                              facecolor=kol,
                              edgecolor="white", linewidth=0.5, zorder=4)
        ax.add_patch(rect)
        ax.text(6.05, yy, opis, ha="left", va="center",
                color=KOLORY["tekst_jasny"],
                fontsize=6.8, fontfamily=FONT_SANS)

    # ── Stopka ───────────────────────────────────────────────────────────────
    ax.plot([0.2, 6.8], [0.1, 0.1], color=KOLORY["ramka"], lw=0.8)
    ax.text(cx, -0.1,
            "Implementacja: ADAM.py  |  funkcje.py  |  main.py",
            ha="center", va="center",
            color=KOLORY["strzalka"], fontsize=7, fontfamily=FONT_MONO)

    plt.tight_layout(pad=0.3)
    nazwa_pliku = "adam_flowchart.png"
    plt.savefig(nazwa_pliku, dpi=180, bbox_inches="tight",
                facecolor=KOLORY["tlo"])
    print(f"Flowchart zapisany jako '{nazwa_pliku}'.")
    plt.show()


if __name__ == "__main__":
    rysuj_flowchart()