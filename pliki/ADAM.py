import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib
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


    def wykres_sciezka_3d(self, margines=0.2, gestosc=100):
        print(f"Generowanie wykresu 3D dla {self.funkcja.nazwa_funkcji} (to może chwilę potrwać)...")
        
        # 1. Pobieranie danych z historii algorytmu
        historia_x = [pozycja[0] for pozycja in self.historia_pozycji]
        historia_y = [pozycja[1] for pozycja in self.historia_pozycji]
        historia_z = self.historia_wartosci
        
        # 2. Automatyczne skalowanie z marginesem
        def oblicz_zakres(dane, m):
            d_min, d_max = min(dane), max(dane)
            rozpietosc = d_max - d_min
            # Jeśli algorytm stał w miejscu, tworzymy sztuczną rozpiętość
            if rozpietosc == 0: rozpietosc = 1.0
            return d_min - m * rozpietosc, d_max + m * rozpietosc

        x_zakres = oblicz_zakres(historia_x, margines)
        y_zakres = oblicz_zakres(historia_y, margines)

        # 3. Tworzenie siatki dla powierzchni funkcji
        x_wartosci = np.linspace(x_zakres[0], x_zakres[1], gestosc)
        y_wartosci = np.linspace(y_zakres[0], y_zakres[1], gestosc)
        X, Y = np.meshgrid(x_wartosci, y_wartosci)
        Z = np.zeros_like(X)
        
        # Obliczanie wysokości (Z) dla każdego punktu na siatce
        for i in range(gestosc):
            for j in range(gestosc):
                Z[i, j] = self.funkcja.fval([X[i, j], Y[i, j]])
                
        # 4. Rysowanie w Plotly
        fig = go.Figure()
        
        # Dodanie powierzchni funkcji
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y, 
            colorscale='Viridis', 
            opacity=0.6, 
            name='Powierzchnia f(x)',
            showscale=False
        ))
        
        # Dodanie ścieżki algorytmu ADAM
        fig.add_trace(go.Scatter3d(
            x=historia_x, y=historia_y, z=historia_z,
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(color='red', width=3),
            name='Ścieżka ADAM'
        ))
        
        # Ustawienia wyglądu i dynamiczne zakresy osi
        fig.update_layout(
            title=f'Optymalizacja ADAM: {self.funkcja.nazwa_funkcji}',
            scene=dict(
                xaxis=dict(title='Oś X1', range=[x_zakres[0], x_zakres[1]]),
                yaxis=dict(title='Oś X2', range=[y_zakres[0], y_zakres[1]]),
                zaxis=dict(title='Wartość f(x)')
            ),
            width=1000, height=800,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        # 5. Zapis do pliku HTML
        nazwa_pliku = f"sciezka_adam_3d_{self.funkcja.nazwa_funkcji}.html"
        fig.write_html(nazwa_pliku)
        print(f"Gotowe! Wykres zapisano jako '{nazwa_pliku}'.") 