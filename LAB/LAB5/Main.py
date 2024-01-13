"""
:author: Tin Jukić
"""
from Matrica import Matrica
from PostupciIFunkcije import Loader, Drawer
from PostupciIFunkcije import Funkcije
from PostupciIFunkcije import Euler


def zad1() -> None:
    print(f"ZAD1", end="\n\n")

    # loading all required matrices
    A: Matrica
    x: Matrica

    # B matrix is not needed here
    A, _, x = Loader.load_from(file=f"Matrice/zad1_matrice.txt", without_B=True)

    print(f"Euler:")
    Loader.save_to(
        file="Matrice/zad1_euler.txt",
        data=Euler.calculate(A=A, B=None, x0=x, f_real=Funkcije.f1, T=0.01, t_max=10)
    )
    Drawer.draw_from(file=f"Matrice/zad1_euler.txt", title=f"Euler - calculated solution", t_max=10, T=0.01)


def zad2() -> None:
    print(f"ZAD2", end="\n\n")

    # loading all required matrices
    A: Matrica
    x: Matrica

    # B matrix is not needed here
    A, _, x = Loader.load_from(file=f"Matrice/zad2_matrice.txt", without_B=True)

    print(f"Euler:")
    Loader.save_to(
        file="Matrice/zad2_euler.txt",
        data=Euler.calculate(A=A, B=None, x0=x, f_real=None, T=0.1, t_max=1)
    )
    Drawer.draw_from(file=f"Matrice/zad2_euler.txt", title=f"Euler - calculated solution", t_max=1, T=0.1)


def zad3() -> None:
    print(f"ZAD3", end="\n\n")


def zad4() -> None:
    print(f"ZAD4", end="\n\n")


def main() -> None:
    zad1()
    # zad2()
    # zad3()
    # zad4()


if __name__ == "__main__":
    main()
