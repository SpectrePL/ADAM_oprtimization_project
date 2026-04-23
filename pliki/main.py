from funkcje import Sphere as Sph
from funkcje import Rosenbrock as Ros
from funkcje import Three_Hump as Th
from funkcje import Himmelblau as Him
from funkcje import Wartosc_Kary as Kara
from funkcje import Funkcja_Z_Kara as Fun_K
from ADAM import ADAM as ADAM

def main():

    kara1=Kara(-0.5,-1,1.5,"kara")
    Rosen=Ros()
    test_function=Fun_K(Rosen,[kara1],None,10000)
    print(f"Punkt startowy: {test_function.start_point}")
    adam = ADAM(test_function)

    print("Optymalizacja rozpoczęta")
    best_point, fval_min = adam.optymalizuj()

    print(f"Najlepszy punkt: {best_point}")
    print(f"Osiągnięta wartość funkcji: {fval_min}")

    #adam.wykres_fval()
    adam.wykres_sciezka_3d()

if __name__ == "__main__":
    main()