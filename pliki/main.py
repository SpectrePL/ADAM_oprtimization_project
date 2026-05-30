import os
import numpy as np

from funkcje import Sphere, Rosenbrock, Three_Hump, Himmelblau, Wartosc_Kary, Funkcja_Z_Kara, Constraint_Circle
from ADAM import ADAM, wykres_grupowy_3d, wykres_grupowy_poziomicowy
from parser_xml import wczytaj_konfiguracje_xml

DOSTEPNE_KLASY = {
    "Rosenbrock": Rosenbrock,
    "Three_Hump": Three_Hump,
    "Himmelblau": Himmelblau,
    "Wartosc_Kary": Wartosc_Kary,
    "Constraint_Circle": Constraint_Circle
}

def main():
    konfiguracja = wczytaj_konfiguracje_xml('eksperymenty.xml')
    adam_params = konfiguracja['adam_defaults']

    for grupa in konfiguracja['experiment_groups']:
        print(f"GRUPA EKSPERYMENTÓW: [{grupa['id']}] {grupa['name']}")
        print(f"Liczba wariantów do złączenia na wykresie: {len(grupa['runs'])}")

        lista_uruchomien_adam = []
        for run_idx, run in enumerate(grupa['runs']):
            print(f"\n  Wariant {run_idx + 1}: {run['name']} | Start: {run['start_point']}")
            
            klasa_funkcji = DOSTEPNE_KLASY[run['function_name']]
            funkcja_celu = klasa_funkcji(start_point=run['start_point'])

            if run['is_constrained']:
                g_constrains, h_constrains = [], []
                for const_def in run['constraints']:
                    klasa_ogr = DOSTEPNE_KLASY[const_def['class_name']]
                    instancja_ogr = klasa_ogr(**const_def['parameters'])
                    if const_def['type'] == 'equality':
                        h_constrains.append(instancja_ogr)
                    elif const_def['type'] == 'inequality':
                        g_constrains.append(instancja_ogr)
                
                funkcja_celu = Funkcja_Z_Kara(funkcja_celu, g_constrains, h_constrains, r=run['r'])

            aktualne_parametry_adam = adam_params.copy()
            if run['adam_overrides']:
                aktualne_parametry_adam.update(run['adam_overrides'])

            adam = ADAM(funkcja_celu, **aktualne_parametry_adam)
            best_point, fval_min = adam.optymalizuj()
            
            lista_uruchomien_adam.append(adam)

            os.makedirs('wyniki', exist_ok=True)
            czysta_nazwa_runu = run['name'].replace(' ', '_').replace('/', '_')
            nazwa_pliku_txt = os.path.join('wyniki', f"{grupa['id']}_wariant_{run_idx+1}_{czysta_nazwa_runu}_wyniki.txt")
            
            with open(nazwa_pliku_txt, 'w', encoding='utf-8') as f:
                f.write(f"RAPORT Z PRZEBIEGU: {run['name']}\n")
                f.write(f"ID grupy wykresów:  {grupa['id']}\n")
                f.write(f"Punkt startowy:     {run['start_point']}\n")
                f.write(f"PARAMETRY ALGORYTMU ADAM:\n")
                f.write(f"Alpha:    {aktualne_parametry_adam['alpha']}\n")
                f.write(f"Beta1:    {aktualne_parametry_adam['beta1']}\n")
                f.write(f"Beta2:    {aktualne_parametry_adam['beta2']}\n")
                f.write(f"Eps:      {aktualne_parametry_adam['eps']}\n")
                f.write(f"Max iter: {aktualne_parametry_adam['max_iter']}\n")
                f.write(f"Wyniki działania algorytmu:\n")
                f.write(f"Wykonanych iteracji: {adam.t}\n")
                f.write(f"Znalezione minimum:  {best_point}\n")
                f.write(f"Wartość funkcji:     {fval_min:.10f}\n")

        print(f"\n[WIZUALIZACJA] Rysowanie trajektorii dla grupy {grupa['id']}...")
        wykres_grupowy_3d(lista_uruchomien_adam, exp_id=grupa['id'], exp_name=grupa['name'])
        wykres_grupowy_poziomicowy(lista_uruchomien_adam, exp_id=grupa['id'], exp_name=grupa['name'])
        print(f"Ukończono generowanie wykresów dla grupy {grupa['id']}.\n")

if __name__ == "__main__":
    main()
