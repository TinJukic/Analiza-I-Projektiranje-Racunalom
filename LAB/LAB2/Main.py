"""
:author: Tin Jukić
"""
from PostupciIFunkcije import Funkcije, ZlatniRez, PretrazivanjePoKoordinatnimOsima, NelderMeaduSimplex
from Matrica import Matrica
import sys


def zad1() -> None:
    print(f"Proba uni-modal:")
    ZlatniRez(x0=100, f=Funkcije.uni_modal_test) \
        .find_uni_modal_interval(x0=100, h=1, f=Funkcije.uni_modal_test).print_matrix()  # works :)

    print(f"Proba zlatni rez:")  # works :)
    ZlatniRez(x0=0,f=Funkcije.golden_section_test, e=1).golden_section(f=Funkcije.golden_section_test).print_matrix()

    print(f"Zlatni rez:")
    zlatniRez: ZlatniRez = ZlatniRez(x0=10, f=Funkcije.f1)
    interval: Matrica = zlatniRez.golden_section(f=Funkcije.f1, print_progress=True)
    print(f"Interval:", end="")
    interval.print_matrix()
    print(f"min = {(interval.get_element_at(position=(0, 0)) + interval.get_element_at(position=(0, 1))) / 2}")

    print(f"\n\nPretraživanje po koordinatnim osima:")
    pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(x0=Matrica(elements=[[10]]), n=1)
    pretrPoKoord.coordinate_search(f=Funkcije.f1, print_progress=True).print_matrix()

    try:
        print(f"\n\nPretraživanje po koordinatnim osima:")
        pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(x0=Matrica(elements=[[0.1, 0.3]]), n=2)
        pretrPoKoord.coordinate_search(f=Funkcije.f2, print_progress=True).print_matrix()
    except OverflowError:
        sys.stderr.write(f"Cannot solve this problem using coordinate axis search.\n")

    # print(f"\n\nNelder-Meadu simplex:")
    # nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[0.1, 0.3]]))
    # nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()


def main() -> None:
    zad1()


if __name__ == "__main__":
    main()
