import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

KOLORY = {
    "tlo":           "#FFFFFF",
    "siatka":        "#F0F0F0",
    "start_stop":    "#1A6FBF",
    "init":          "#1F8C3B",  
    "obliczenia":    "#C96010",   
    "warunek":       "#6A3DAA", 
    "zapis":         "#1A8C3B",  
    "wynik":         "#A07800",  
    "tekst_jasny":   "#FFFFFF",  
    "tekst_ciemny":  "#FFFFFF", 
    "strzalka":      "#444444",  
    "akcent":        "#1050AA",  
    "ramka":         "#555555",   
    "ramka_border":  "#333333",   
    "legenda_bg":    "#F5F5F5",  
    "legenda_border":"#CCCCCC",
    "tytul":         "#1A3A6F",   
}

FONT_MONO  = "DejaVu Sans Mono"
FONT_SANS  = "DejaVu Sans"

OPISY = {
    "gradient":   "pochodna f po x",
    "moment1":    "wygładzona średnia gradientu",
    "moment2":    "wygładzona wariancja gradientu",
    "bias":       "korekcja wychylenia momentów",
    "update":     "krok w kierunku min. f(x)",
}


def ramka(ax, x, y, w, h, kolor_wypelnienia, tekst,
          styl="round,pad=0.08", tekst_kolor=None,
          rozmiar_czcionki=8.5, bold=False, opis=None):
    if tekst_kolor is None:
        tekst_kolor = KOLORY["tekst_jasny"]

    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=styl,
        facecolor=kolor_wypelnienia,
        edgecolor=KOLORY["ramka_border"],
        linewidth=0.9,
        zorder=3,
    )
    ax.add_patch(box)

    if opis:
        ax.text(
            x, y + h * 0.15, tekst,
            ha="center", va="center",
            color=tekst_kolor,
            fontsize=rozmiar_czcionki,
            fontfamily=FONT_SANS,
            fontweight="bold" if bold else "normal",
            zorder=4,
        )
        ax.text(
            x, y - h * 0.30, f"↳ {opis}",
            ha="center", va="center",
            color="#FFE0B0" if kolor_wypelnienia == KOLORY["obliczenia"] else "#E8FFE8",
            fontsize=rozmiar_czcionki - 2.5,
            fontfamily=FONT_SANS,
            fontstyle="italic",
            zorder=4,
        )
    else:
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
    pts = np.array([
        [x,         y + h / 2],
        [x + w / 2, y],
        [x,         y - h / 2],
        [x - w / 2, y],
    ])
    poly = plt.Polygon(pts, closed=True,
                       facecolor=kolor_wypelnienia,
                       edgecolor=KOLORY["ramka_border"], linewidth=0.9, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, tekst, ha="center", va="center",
            color=KOLORY["tekst_jasny"],
            fontsize=rozmiar_czcionki,
            fontfamily=FONT_SANS, fontweight="bold", zorder=4)


def strzalka(ax, x1, y1, x2, y2, etykieta=None, kolor=None):
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
                color="#222222",
                fontsize=9,
                fontfamily=FONT_SANS,
                fontweight="bold",
                zorder=5)


def linia(ax, xs, ys, kolor=None):
    if kolor is None:
        kolor = KOLORY["strzalka"]
    ax.plot(xs, ys, color=kolor, lw=1.4, zorder=2)

def rysuj_flowchart():
    fig, ax = plt.subplots(figsize=(13, 22))
    fig.patch.set_facecolor(KOLORY["tlo"])
    ax.set_facecolor(KOLORY["tlo"])
    ax.set_xlim(-1.5, 8.5)
    ax.set_ylim(-1.5, 26.5)
    ax.axis("off")



    cx = 3.5   # oś główna X
    W  = 3.8   # szerokość bloków prostokątnych
    H  = 0.72  # wysokość bloków (zwiększona dla opisu)

    ax.text(cx, 26.0, "ADAM OPTIMIZER",
            ha="center", va="center",
            color=KOLORY["tytul"],
            fontsize=24, fontfamily=FONT_MONO,
            fontweight="bold", zorder=5)
    ax.text(cx, 25.4, "Pipeline — diagram przepływu algorytmu",
            ha="center", va="center",
            color=KOLORY["strzalka"],
            fontsize=10.8, fontfamily=FONT_SANS, zorder=5)
    ax.plot([0.2, 6.8], [25.1, 25.1],
            color="#BBBBBB", lw=1.0, zorder=2)

    y1 = 24.2
    ramka(ax, cx, y1, W, H, KOLORY["start_stop"],
          "START", styl="round,pad=0.12", bold=True, rozmiar_czcionki=12)
    strzalka(ax, cx, y1 - H/2, cx, y1 - H/2 - 0.35)

    y2 = 23.1
    ramka(ax, cx, y2, W, H, KOLORY["init"],
          "Inicjalizacja funkcji celu\n(function_template / Funkcja_Z_Kara)",
          rozmiar_czcionki=9.6)
    strzalka(ax, cx, y2 - H/2, cx, y2 - H/2 - 0.35)

    y3 = 22.0
    ramka(ax, cx, y3, W, H * 1.55, KOLORY["init"],
          "Inicjalizacja ADAM\nα, β₁, β₂, ε, max_iter\n"
          "x ← start_point,  m=0,  v=0,  t=0",
          rozmiar_czcionki=9.6)
    strzalka(ax, cx, y3 - H * 1.55/2, cx, y3 - H * 1.55/2 - 0.35)

    y4 = 20.5
    ramka(ax, cx, y4, W, H, KOLORY["obliczenia"],
          "t  ←  t + 1", rozmiar_czcionki=10.8, bold=True)
    strzalka(ax, cx, y4 - H/2, cx, y4 - H/2 - 0.35)

    y5 = 19.55
    ramka(ax, cx, y5, W, H, KOLORY["obliczenia"],
          "g  ←  ∇f(x)   [grad_val(x)]",
          rozmiar_czcionki=10.2,
          opis=OPISY["gradient"])
    strzalka(ax, cx, y5 - H/2, cx, y5 - H/2 - 0.35)

    y6 = 18.6
    ramka(ax, cx, y6, W, H, KOLORY["obliczenia"],
          "m  ←  β₁·m + (1−β₁)·g",
          rozmiar_czcionki=10.2,
          opis=OPISY["moment1"])
    strzalka(ax, cx, y6 - H/2, cx, y6 - H/2 - 0.35)

    y7 = 17.65
    ramka(ax, cx, y7, W, H, KOLORY["obliczenia"],
          "v  ←  β₂·v + (1−β₂)·g²",
          rozmiar_czcionki=10.2,
          opis=OPISY["moment2"])
    strzalka(ax, cx, y7 - H/2, cx, y7 - H/2 - 0.35)

    y8 = 16.7
    ramka(ax, cx, y8, W, H, KOLORY["obliczenia"],
          "m̂ ← m/(1−β₁ᵗ)    v̂ ← v/(1−β₂ᵗ)",
          rozmiar_czcionki=9.6,
          opis=OPISY["bias"])
    strzalka(ax, cx, y8 - H/2, cx, y8 - H/2 - 0.35)

    y9 = 15.75
    ramka(ax, cx, y9, W, H, KOLORY["obliczenia"],
          "x  ←  x − α · m̂ / (√v̂ + ε)",
          rozmiar_czcionki=10.2,
          opis=OPISY["update"])
    strzalka(ax, cx, y9 - H/2, cx, y9 - H/2 - 0.35)

    y10 = 14.8
    ramka(ax, cx, y10, W, H, KOLORY["zapis"],
          "Zapis: historia_pozycji[t],  historia_wartosci[t]",
          rozmiar_czcionki=9.36)
    strzalka(ax, cx, y10 - H/2, cx, y10 - H/2 - 0.35)

    y11 = 13.85
    RD_W, RD_H = 4.0, 0.72
    romb(ax, cx, y11, RD_W, RD_H,
         KOLORY["warunek"], "t  >  brak_zmian_maxiter ?",
         rozmiar_czcionki=9.6)
    strzalka(ax, cx, y11 - RD_H/2, cx, y11 - RD_H/2 - 0.35,
             etykieta="TAK")

    ax.text(cx + RD_W/2 + 0.08, y11, "NIE",
            ha="left", va="center",
            color="#222222", fontsize=9, fontfamily=FONT_SANS,
            fontweight="bold")
    linia(ax,
          [cx + RD_W/2, cx + RD_W/2 + 0.55, cx + RD_W/2 + 0.55],
          [y11,          y11,                  y4 + H/2])
    strzalka(ax, cx + RD_W/2 + 0.55, y4 + H/2, cx + W/2, y4 + H/2)

    y12 = 12.8
    romb(ax, cx, y12, RD_W, RD_H,
         KOLORY["warunek"],
         "Δf < ε_stop przez N iter?",
         rozmiar_czcionki=9.6)
    strzalka(ax, cx, y12 - RD_H/2, cx, y12 - RD_H/2 - 0.35,
             etykieta="TAK")

    ax.text(cx - RD_W/2 - 0.08, y12, "NIE",
            ha="right", va="center",
            color="#222222", fontsize=9, fontfamily=FONT_SANS,
            fontweight="bold")
    linia(ax,
          [cx - RD_W/2, cx - RD_W/2 - 0.55, cx - RD_W/2 - 0.55],
          [y12,           y12,                  y4 + H/2])
    strzalka(ax, cx - RD_W/2 - 0.55, y4 + H/2, cx - W/2, y4 + H/2)

    y13 = 11.75
    romb(ax, cx, y13, RD_W, RD_H,
         KOLORY["warunek"], "t  ≥  max_iter ?",
         rozmiar_czcionki=9.6)
    strzalka(ax, cx, y13 - RD_H/2, cx, y13 - RD_H/2 - 0.35,
             etykieta="TAK")

    ax.text(cx + RD_W/2 + 0.08, y13, "NIE",
            ha="left", va="center",
            color="#222222", fontsize=9, fontfamily=FONT_SANS,
            fontweight="bold")
    linia(ax,
          [cx + RD_W/2, cx + RD_W/2 + 0.85, cx + RD_W/2 + 0.85],
          [y13,           y13,                   y4 + H/2])
    strzalka(ax, cx + RD_W/2 + 0.85, y4 + H/2, cx + W/2, y4 + H/2)

    y14 = 10.5
    ramka(ax, cx, y14, W, H, KOLORY["wynik"],
          "Zwróć:  x*,  f(x*)   [optymalizuj() → return]",
          rozmiar_czcionki=9.6, bold=True)
    strzalka(ax, cx, y14 - H/2, cx, y14 - H/2 - 0.35)

    y15 = 9.6
    ramka(ax, cx, y15, W, H * 0.8, KOLORY["ramka"],
          "Opcjonalna wizualizacja wyników",
          rozmiar_czcionki=9.6, bold=True)
    strzalka(ax, cx, y15 - H * 0.8/2, cx, y15 - H * 0.8/2 - 0.35)

    y_fork = y15 - H * 0.8/2 - 0.35
    CX_L = cx - 1.6
    CX_R = cx + 1.6
    W2 = 2.8

    linia(ax, [CX_L, CX_R], [y_fork, y_fork])

    y16 = 8.2
    strzalka(ax, CX_L, y_fork, CX_L, y16 + H/2)
    strzalka(ax, CX_R, y_fork, CX_R, y16 + H/2)

    ramka(ax, CX_L, y16, W2, H, KOLORY["akcent"],
          "wykres_fval()\nf(x) vs iteracja [plt]",
          rozmiar_czcionki=9)
    ramka(ax, CX_R, y16, W2, H, KOLORY["akcent"],
          "wykres_sciezka_3d()\nścieżka na pow. f [plotly]",
          rozmiar_czcionki=9)

    strzalka(ax, CX_L, y16 - H/2, CX_L, y16 - H/2 - 0.35)
    strzalka(ax, CX_R, y16 - H/2, CX_R, y16 - H/2 - 0.35)
    y_join = y16 - H/2 - 0.35
    linia(ax, [CX_L, CX_R], [y_join, y_join])
    strzalka(ax, cx, y_join, cx, y_join - 0.35)

    y17 = y_join - 0.35 - H/2
    ramka(ax, cx, y17, W * 0.6, H, KOLORY["start_stop"],
          "STOP", styl="round,pad=0.12", bold=True, rozmiar_czcionki=12)

    legenda_x = 6.13  
    legenda_y_start = 23.5

    leg_bg = FancyBboxPatch((5.35, legenda_y_start - 3.6), 1.575, 4.35,
                            boxstyle="round,pad=0.1",
                            facecolor=KOLORY["legenda_bg"],
                            edgecolor=KOLORY["legenda_border"],
                            linewidth=0.8, zorder=2)
    ax.add_patch(leg_bg)

    pozycje = [
        (KOLORY["start_stop"], "START / STOP"),
        (KOLORY["init"],       "Inicjalizacja"),
        (KOLORY["obliczenia"], "Obliczenia ADAM"),
        (KOLORY["warunek"],    "Warunek stopu"),
        (KOLORY["zapis"],      "Zapis stanu"),
        (KOLORY["wynik"],      "Wynik / zwrot"),
        (KOLORY["akcent"],     "Wizualizacja"),
    ]
    ax.text(legenda_x, legenda_y_start + 0.45, "LEGENDA",
            ha="center", va="center",
            color=KOLORY["tytul"],
            fontsize=7.2, fontfamily=FONT_MONO, fontweight="bold")
    ax.plot([5.42, 6.85], [legenda_y_start + 0.2, legenda_y_start + 0.2],
            color="#CCCCCC", lw=0.8)
    for i, (kol, opis) in enumerate(pozycje):
        yy = legenda_y_start - i * 0.55
        rect = FancyBboxPatch((5.42, yy - 0.13), 0.26, 0.26,
                              boxstyle="round,pad=0.03",
                              facecolor=kol,
                              edgecolor="#333333", linewidth=0.5, zorder=4)
        ax.add_patch(rect)
        ax.text(5.74, yy, opis, ha="left", va="center",
                color="#222222",
                fontsize=6.0, fontfamily=FONT_SANS)

   

    plt.tight_layout(pad=0.3)
    nazwa_pliku = "/mnt/user-data/outputs/adam_flowchart.png"
    plt.savefig(nazwa_pliku, dpi=180, bbox_inches="tight",
                facecolor=KOLORY["tlo"])
    print(f"Flowchart zapisany jako '{nazwa_pliku}'.")
    plt.close()


if __name__ == "__main__":
    rysuj_flowchart()