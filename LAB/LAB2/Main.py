"""
:author: Tin Jukić
"""
from PostupciIFunkcije import Funkcije, ZlatniRez, PretrazivanjePoKoordinatnimOsima
from Matrica import Matrica


def zad1() -> None:
    print(f"Zlatni rez:")
    zlatniRez: ZlatniRez = ZlatniRez(x0=10, f=Funkcije.f1)
    interval: Matrica = zlatniRez.golden_section(f=Funkcije.f1, print_progress=True)
    print(f"Interval:", end="")
    interval.print_matrix()
    print(f"min = {(interval.get_element_at(position=(0, 0)) + interval.get_element_at(position=(0, 1))) / 2}")

    # print(f"\n\nPretraživanje po koordinatnim osima:")
    # pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(x0=Matrica(elements=[[10]]), n=1)
    # pretrPoKoord.coordinate_search(f=Funkcije.f1, print_progress=True).print_matrix()


def main() -> None:
    zad1()


if __name__ == "__main__":
    main()
