from funkcje import Sphere
from funkcje import Rosenbrock
from funkcje import Three_Hump
from funkcje import Himmelblau
from ADAM import ADAM as ADAM

def main():

    x0 = [-5.0, 5.0]
    test_function=Himmelblau(x0)

    adam = ADAM(test_function)

    print("Optymalizacja rozpoczęta")
    best_point, fval_min = adam.optymalizuj()

    print(f"Najlepszy punkt: {best_point}")
    print(f"Osiągnięta wartość funkcji: {fval_min}")

    #adam.wykres_fval()
    adam.wykres_sciezka_3d()

if __name__ == "__main__":
    main()