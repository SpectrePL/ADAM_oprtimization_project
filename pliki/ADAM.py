import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib
import os
matplotlib.use('TkAgg')

class ADAM:
    def __init__(self, funkcja_celu, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=100000):
        #ukradnięcie funkcji
        self.funkcja = funkcja_celu
        #hiperparametry adama
        self.alpha=alpha
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.max_iter=max_iter
        #inicjalizacja stanu algorytmu
        self.x=np.copy(self.funkcja.start_point)
        self.m=np.zeros_like(self.x)
        self.v=np.zeros_like(self.x)
        self.t=0
        #dane do wizualizacji
        self.historia_pozycji=[np.copy(self.x)] 
        self.historia_wartosci=[self.funkcja.fval(self.x)]
        self.historia_m = [np.copy(self.m)]
        self.historia_v = [np.copy(self.v)]

        self.brak_zmian = 0
        self.brak_zmian_maxiter = 10
        self.warunek_braku_zmian = 1e-10

    def optymalizuj(self):
        for _ in range(self.max_iter): #tymczasowo tylko ten warunek stopu
            self.t+=1
            gradient=self.funkcja.grad_val(self.x)
            #tu matematyka adama PROSZE DZIAŁAJ
            self.m=self.m*self.beta1+(1-self.beta1)*gradient
            self.v=self.v*self.beta2+(1-self.beta2)*(gradient**2)
            dm=self.m/(1-self.beta1**self.t)
            dv=self.v/(1-self.beta2**self.t)
            self.x=self.x-(self.alpha*dm/np.sqrt(dv+self.eps))
            #tu zapis do historii, by robić wizkę
            self.historia_pozycji.append(np.copy(self.x))
            self.historia_wartosci.append(self.funkcja.fval(self.x))
            self.historia_m.append(np.copy(self.m))
            self.historia_v.append(np.copy(self.v))
            #tu się dorobi ekstra warunek stopu
            if self.t>self.brak_zmian_maxiter:
                for i in range(self.brak_zmian_maxiter):
                    if abs(self.historia_wartosci[self.t -i ] - self.historia_wartosci[self.t - i - 1]) < self.warunek_braku_zmian: 
                        self.brak_zmian += 1
                if self.brak_zmian == self.brak_zmian_maxiter:
                    print(f"Przerwano przez brak zmian; iteracja: {self.t}")
                    break
            self.brak_zmian = 0
        print(f"Iteracja: {self.t}")
        return self.x, self.historia_wartosci[-1]
    
    def wykres_fval(self):
        plt.figure(figsize=(8,8))
        plt.plot(self.historia_wartosci, label='FVal', color='blue')
        plt.yscale('log')
        plt.title('Wartość funkcji celu')
        plt.xlabel('Iteracja')
        plt.ylabel('Wartość f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def wykres_sciezka_3d(self, exp_id="EXP", exp_name="Brak_nazwy", margines=0.2, gestosc=100):
        print(f"Generowanie wykresu 3D dla {exp_name}...")
        
        historia_x = [pozycja[0] for pozycja in self.historia_pozycji]
        historia_y = [pozycja[1] for pozycja in self.historia_pozycji]
        historia_z = self.historia_wartosci
        
        def oblicz_zakres(dane, m):
            d_min, d_max = min(dane), max(dane)
            rozpietosc = d_max - d_min
            if rozpietosc == 0: rozpietosc = 1.0
            return d_min - m * rozpietosc, d_max + m * rozpietosc

        x_zakres = oblicz_zakres(historia_x, margines)
        y_zakres = oblicz_zakres(historia_y, margines)

        x_wartosci = np.linspace(x_zakres[0], x_zakres[1], gestosc)
        y_wartosci = np.linspace(y_zakres[0], y_zakres[1], gestosc)
        X, Y = np.meshgrid(x_wartosci, y_wartosci)
        Z = np.zeros_like(X)
        
        for i in range(gestosc):
            for j in range(gestosc):
                Z[i, j] = self.funkcja.fval([X[i, j], Y[i, j]])
                
        fig = go.Figure()
        
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y, 
            colorscale='Viridis', 
            opacity=0.6, 
            name='Powierzchnia f(x)',
            showscale=False
        ))
        
        fig.add_trace(go.Scatter3d(
            x=historia_x, y=historia_y, z=historia_z,
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(color='red', width=3),
            name='Ścieżka ADAM'
        ))
        
        fig.update_layout(
            title=f'[{exp_id}] {exp_name} - 3D',
            scene=dict(
                xaxis=dict(title='Oś X1', range=[x_zakres[0], x_zakres[1]]),
                yaxis=dict(title='Oś X2', range=[y_zakres[0], y_zakres[1]]),
                zaxis=dict(title='Wartość f(x)')
            ),
            width=1000, height=800,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        os.makedirs('wyniki', exist_ok=True)
        czysta_nazwa = exp_name.replace(' ', '_').replace('/', '_')
        nazwa_pliku = os.path.join('wyniki', f"{exp_id}_{czysta_nazwa}_3D.html")
        fig.write_html(nazwa_pliku)
        print(f"Gotowe! Zapisano jako: {nazwa_pliku}")

    def wykres_poziomicowy(self, exp_id="EXP", exp_name="Brak_nazwy", margines=0.2, gestosc=200):
        print(f"Generowanie wykresu poziomicowego dla {exp_name}...")
        
        historia_x = [pozycja[0] for pozycja in self.historia_pozycji]
        historia_y = [pozycja[1] for pozycja in self.historia_pozycji]
        
        def oblicz_zakres(dane, m):
            d_min, d_max = min(dane), max(dane)
            rozpietosc = max(d_max - d_min, 1.0)
            return d_min - m * rozpietosc, d_max + m * rozpietosc

        x_zakres = oblicz_zakres(historia_x, margines)
        y_zakres = oblicz_zakres(historia_y, margines)

        x_wartosci = np.linspace(x_zakres[0], x_zakres[1], gestosc)
        y_wartosci = np.linspace(y_zakres[0], y_zakres[1], gestosc)
        X, Y = np.meshgrid(x_wartosci, y_wartosci)
        Z = np.zeros_like(X)
        
        ma_kary = hasattr(self.funkcja, 'target')
        funkcja_bazowa = self.funkcja.target if ma_kary else self.funkcja
        
        for i in range(gestosc):
            for j in range(gestosc):
                Z[i, j] = funkcja_bazowa.fval([X[i, j], Y[i, j]])
                
        plt.figure(figsize=(10, 8))
        
        cp = plt.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        plt.colorbar(cp, label='Wartość bazowej f(x) (bez kar)')
        
        if ma_kary:
            if hasattr(self.funkcja, 'g_constrains') and self.funkcja.g_constrains:
                for idx, g in enumerate(self.funkcja.g_constrains):
                    Z_g = np.zeros_like(X)
                    for i in range(gestosc):
                        for j in range(gestosc):
                            Z_g[i, j] = g.fval([X[i, j], Y[i, j]])
                    
                    plt.contour(X, Y, Z_g, levels=[0], colors='red', linewidths=2, linestyles='dashed')
                    plt.contourf(X, Y, Z_g, levels=[0, np.inf], colors='red', alpha=0.15)
                
                plt.plot([], [], color='red', linestyle='dashed', linewidth=2, label='Obszar g(x) > 0 (niedopuszczalny)')
                    
            if hasattr(self.funkcja, 'h_constrains') and self.funkcja.h_constrains:
                for idx, h in enumerate(self.funkcja.h_constrains):
                    Z_h = np.zeros_like(X)
                    for i in range(gestosc):
                        for j in range(gestosc):
                            Z_h[i, j] = h.fval([X[i, j], Y[i, j]])
                    
                    plt.contour(X, Y, Z_h, levels=[0], colors='blue', linewidths=2)
                
                plt.plot([], [], color='blue', linestyle='solid', linewidth=2, label='Linia h(x) = 0')

        plt.plot(historia_x, historia_y, color='black', marker='.', markersize=5, linestyle='-', linewidth=1.5, label='Ścieżka ADAM')
        plt.plot(historia_x[0], historia_y[0], marker='s', color='white', markeredgecolor='black', markersize=8, label='Start')
        plt.plot(historia_x[-1], historia_y[-1], marker='*', color='gold', markeredgecolor='black', markersize=14, label='Koniec')
        
        plt.title(f'[{exp_id}] {exp_name} - Poziomice i Ograniczenia')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0)) 
        plt.grid(True, linestyle='--', alpha=0.4)
        os.makedirs('wyniki', exist_ok=True)
        czysta_nazwa = exp_name.replace(' ', '_').replace('/', '_')
        nazwa_pliku = os.path.join('wyniki', f"{exp_id}_{czysta_nazwa}_Poziomice.png")
        plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gotowe! Zapisano jako: {nazwa_pliku}")

def wykres_grupowy_3d(lista_adamow, exp_id, exp_name, margines=0.2, gestosc=100):
    if not lista_adamow: return
    print(f"Generowanie zbiorczego wykresu 3D dla grupy {exp_id}...")

    wszystkie_x, wszystkie_y = [], []
    for instancja in lista_adamow:
        wszystkie_x.extend([p[0] for p in instancja.historia_pozycji])
        wszystkie_y.extend([p[1] for p in instancja.historia_pozycji])

    def oblicz_zakres(dane, m):
        d_min, d_max = min(dane), max(dane)
        rozpietosc = max(d_max - d_min, 1.0)
        return d_min - m * rozpietosc, d_max + m * rozpietosc

    x_zakres = oblicz_zakres(wszystkie_x, margines)
    y_zakres = oblicz_zakres(wszystkie_y, margines)

    X, Y = np.meshgrid(np.linspace(x_zakres[0], x_zakres[1], gestosc), np.linspace(y_zakres[0], y_zakres[1], gestosc))
    Z = np.zeros_like(X)
    
    funkcja_wzorcowa = lista_adamow[0].funkcja
    funkcja_bazowa = funkcja_wzorcowa.target if hasattr(funkcja_wzorcowa, 'target') else funkcja_wzorcowa
    
    for i in range(gestosc):
        for j in range(gestosc):
            Z[i, j] = funkcja_bazowa.fval([X[i, j], Y[i, j]])
        
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.5, showscale=False, name='Powierzchnia f(x)'))
    
    paleta = ['red', 'blue', 'orange', 'green', 'magenta', 'cyan', 'black']

    for idx, instancja in enumerate(lista_adamow):
        hx = [p[0] for p in instancja.historia_pozycji]
        hy = [p[1] for p in instancja.historia_pozycji]
        hz = instancja.historia_wartosci
        kolor = paleta[idx % len(paleta)]
        
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode='lines+markers',
            marker=dict(size=3, color=kolor), line=dict(color=kolor, width=3),
            name=f"Wariant {idx+1} (Start: [{hx[0]:.2f}, {hy[0]:.2f}])"
        ))
    
    fig.update_layout(
        title=f'[{exp_id}] {exp_name} - Zbiorczy Wykres 3D',
        scene=dict(xaxis=dict(title='X1', range=[x_zakres[0], x_zakres[1]]),
                yaxis=dict(title='X2', range=[y_zakres[0], y_zakres[1]]),
                zaxis=dict(title='f(x)')),
        width=1000, height=800, margin=dict(l=0, r=0, b=0, t=50)
    )
    
    os.makedirs('wyniki', exist_ok=True)
    czysta_nazwa = exp_name.replace(' ', '_').replace('/', '_')
    fig.write_html(os.path.join('wyniki', f"{exp_id}_{czysta_nazwa}_Wspolny_3D.html"))

def wykres_grupowy_poziomicowy(lista_adamow, exp_id, exp_name, margines=0.2, gestosc=200):
    if not lista_adamow: return
    print(f"Generowanie zbiorczego wykresu poziomicowego dla grupy {exp_id}...")
    
    wszystkie_x, wszystkie_y = [], []
    for instancja in lista_adamow:
        wszystkie_x.extend([p[0] for p in instancja.historia_pozycji])
        wszystkie_y.extend([p[1] for p in instancja.historia_pozycji])

    def oblicz_zakres(dane, m):
        d_min, d_max = min(dane), max(dane)
        rozpietosc = max(d_max - d_min, 1.0)
        return d_min - m * rozpietosc, d_max + m * rozpietosc

    x_zakres = oblicz_zakres(wszystkie_x, margines)
    y_zakres = oblicz_zakres(wszystkie_y, margines)

    X, Y = np.meshgrid(np.linspace(x_zakres[0], x_zakres[1], gestosc), np.linspace(y_zakres[0], y_zakres[1], gestosc))
    Z = np.zeros_like(X)
    
    funkcja_wzorcowa = lista_adamow[0].funkcja
    ma_kary = hasattr(funkcja_wzorcowa, 'target')
    funkcja_bazowa = funkcja_wzorcowa.target if ma_kary else funkcja_wzorcowa
    
    for i in range(gestosc):
        for j in range(gestosc):
            Z[i, j] = funkcja_bazowa.fval([X[i, j], Y[i, j]])
            
    plt.figure(figsize=(11, 8))
    cp = plt.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.5)
    plt.colorbar(cp, label='Wartość f(x) (bez kar)')
    
    if ma_kary:
        if hasattr(funkcja_wzorcowa, 'g_constrains') and funkcja_wzorcowa.g_constrains:
            for g in funkcja_wzorcowa.g_constrains:
                Z_g = np.zeros_like(X)
                for i in range(gestosc):
                    for j in range(gestosc): Z_g[i, j] = g.fval([X[i, j], Y[i, j]])
                plt.contour(X, Y, Z_g, levels=[0], colors='red', linewidths=2, linestyles='dashed')
                plt.contourf(X, Y, Z_g, levels=[0, np.inf], colors='red', alpha=0.08)
            plt.plot([], [], color='red', linestyle='dashed', linewidth=2, label='Obszar niedopuszczalny g(x) > 0')

        if hasattr(funkcja_wzorcowa, 'h_constrains') and funkcja_wzorcowa.h_constrains:
            for h in funkcja_wzorcowa.h_constrains:
                Z_h = np.zeros_like(X)
                for i in range(gestosc):
                    for j in range(gestosc): Z_h[i, j] = h.fval([X[i, j], Y[i, j]])
                plt.contour(X, Y, Z_h, levels=[0], colors='blue', linewidths=2)
            plt.plot([], [], color='blue', linestyle='solid', linewidth=2, label='Linia równościowa h(x) = 0')

    paleta = ['black', 'darkgreen', 'darkred', 'purple', 'darkorange', 'saddlebrown']
    for idx, instancja in enumerate(lista_adamow):
        hx = [p[0] for p in instancja.historia_pozycji]
        hy = [p[1] for p in instancja.historia_pozycji]
        kolor = paleta[idx % len(paleta)]
        
        plt.plot(hx, hy, color=kolor, marker='.', markersize=3, linestyle='-', linewidth=1.5,
                label=f"Wariant {idx+1} (Start: [{hx[0]:.1f}, {hy[0]:.1f}])")
        plt.plot(hx[0], hy[0], marker='s', color='white', markeredgecolor=kolor, markersize=6)
        plt.plot(hx[-1], hy[-1], marker='*', color='gold', markeredgecolor=kolor, markersize=10)
        
    plt.title(f'[{exp_id}] {exp_name} - Zbiorcze Poziomice')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.0))
    plt.grid(True, linestyle='--', alpha=0.4)
    
    os.makedirs('wyniki', exist_ok=True)
    czysta_nazwa = exp_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join('wyniki', f"{exp_id}_{czysta_nazwa}_Wspolne_Poziomice.png"), dpi=300, bbox_inches='tight')
    plt.close()

def wykres_grupowy_momentow(lista_adamow, exp_id, exp_name):
    if not lista_adamow: return
    print(f"Generowanie zbiorczego wykresu momentów dla grupy {exp_id}...")

    # Zwiększamy nieco szerokość, żeby zmieścić wspólną legendę z boku
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    paleta = ['black', 'darkgreen', 'darkred', 'purple', 'darkorange', 'saddlebrown']

    for idx, instancja in enumerate(lista_adamow):
        hist_m = np.array(instancja.historia_m)
        hist_v = np.array(instancja.historia_v)
        iteracje = np.arange(len(hist_m))
        kolor = paleta[idx % len(paleta)]
        
        etykieta = f"Wariant {idx+1}"

        # Rysujemy pierwszy moment (m)
        ax1.plot(iteracje, hist_m[:, 0], color=kolor, linestyle='-', alpha=0.8, linewidth=1.5, label=f'{etykieta} (Oś X1)')
        ax1.plot(iteracje, hist_m[:, 1], color=kolor, linestyle=':', alpha=0.8, linewidth=2.0, label=f'{etykieta} (Oś X2)')

        # Rysujemy drugi moment (v)
        ax2.plot(iteracje, hist_v[:, 0], color=kolor, linestyle='-', alpha=0.8, linewidth=1.5, label=f'{etykieta} (Oś X1)')
        ax2.plot(iteracje, hist_v[:, 1], color=kolor, linestyle=':', alpha=0.8, linewidth=2.0, label=f'{etykieta} (Oś X2)')

    # Konfiguracja wykresu m
    ax1.set_title('Pierwszy moment (m) - Pęd')
    ax1.set_ylabel('Wartość m')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Konfiguracja wykresu v
    ax2.set_title('Drugi moment (v) - Wariancja (Skala logarytmiczna)')
    ax2.set_xlabel('Iteracja')
    ax2.set_ylabel('Wartość v')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Wyciągamy legendę wspólną dla obu wykresów i wyrzucamy ją na zewnątrz
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.25, 0.5))

    plt.suptitle(f'[{exp_id}] {exp_name} - Zbiorcze zmiany momentów')
    
    # Korygujemy marginesy, żeby legenda nie ucięła się przy zapisie
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    # Zapis
    os.makedirs('wyniki', exist_ok=True)
    czysta_nazwa = exp_name.replace(' ', '_').replace('/', '_')
    nazwa_pliku = os.path.join('wyniki', f"{exp_id}_{czysta_nazwa}_Wspolne_Momenty.png")
    plt.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gotowe! Zapisano jako: {nazwa_pliku}")
