"""
:author: Tin Jukić
"""
from PostupciIFunkcije import Funkcije, ZlatniRez
import Matrica


def zad1() -> None:
    zlatniRez: ZlatniRez = ZlatniRez(x0=10, f=Funkcije.f1)
    interval: Matrica = zlatniRez.golden_section(f=Funkcije.f1)
    interval.print_matrix()
    print(f"min = {(interval.get_element_at(position=(0, 0)) + interval.get_element_at(position=(0, 1))) / 2}")


def main() -> None:
    zad1()


if __name__ == "__main__":
    main()
