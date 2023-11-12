"""
:author: Tin Jukić
"""
from PostupciIFunkcije import Funkcije, ZlatniRez, PretrazivanjePoKoordinatnimOsima, NelderMeaduSimplex, HookeJeeves
from Matrica import Matrica


def zad1() -> None:
    print(f"Proba uni-modal:")
    ZlatniRez(x0=100, f=Funkcije.uni_modal_test) \
        .find_uni_modal_interval(x0=100, h=1, f=Funkcije.uni_modal_test).print_matrix()  # works :)

    print(f"Proba zlatni rez:")  # works :)
    ZlatniRez(x0=0,f=Funkcije.golden_section_test, e=1).golden_section(f=Funkcije.golden_section_test).print_matrix()

    print(f"Zlatni rez:")
    zlatniRez: ZlatniRez = ZlatniRez(x0=10, f=Funkcije.f)
    interval: Matrica = zlatniRez.golden_section(f=Funkcije.f, print_progress=True)
    print(f"Interval:", end="")
    interval.print_matrix()
    print(f"min = {(interval.get_element_at(position=(0, 0)) + interval.get_element_at(position=(0, 1))) / 2}")

    print(f"\n\nPretraživanje po koordinatnim osima:")
    pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(x0=Matrica(elements=[[10]]), n=1)
    pretrPoKoord.coordinate_search(f=Funkcije.f, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex:")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[10]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f, print_progress=True).print_matrix()

    print(f"\n\nHooke-Jeeves simplex:")
    hookeJeeves: HookeJeeves = HookeJeeves(x0=Matrica(elements=[[10]]))
    hookeJeeves.calculate_hooke_jeeves(f=Funkcije.f, print_progress=True).print_matrix()


def zad2() -> None:
    print(f"Funkcija f1:")

    print(f"\n\nPretraživanje po koordinatnim osima:")
    pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(x0=Matrica(elements=[[-1.9, 2]]), n=2)
    pretrPoKoord.coordinate_search(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex:")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[-1.9, 2]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nHooke-Jeeves simplex:")
    hookeJeeves: HookeJeeves = HookeJeeves(x0=Matrica(elements=[[-1.9, 2]]))
    hookeJeeves.calculate_hooke_jeeves(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"Funkcija f2")

    print(f"\n\nPretraživanje po koordinatnim osima:")
    pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(x0=Matrica(elements=[[0.1, 0.3]]), n=2)
    pretrPoKoord.coordinate_search(f=Funkcije.f2, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex:")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[0.1, 0.3]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f2, print_progress=True).print_matrix()

    print(f"\n\nHooke-Jeeves simplex:")
    hookeJeeves: HookeJeeves = HookeJeeves(x0=Matrica(elements=[[0.1, 0.3]]))
    hookeJeeves.calculate_hooke_jeeves(f=Funkcije.f2, print_progress=True).print_matrix()

    print(f"Funkcija f3:")

    print(f"\n\nPretraživanje po koordinatnim osima:")
    pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(x0=Matrica(elements=[[0, 0, 0, 0, 0]]),
                                                                                      n=5)
    pretrPoKoord.coordinate_search(f=Funkcije.f3, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex:")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[0, 0, 0, 0, 0]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f3, print_progress=True).print_matrix()

    print(f"\n\nHooke-Jeeves simplex:")
    hookeJeeves: HookeJeeves = HookeJeeves(x0=Matrica(elements=[[0, 0, 0, 0, 0]]))
    hookeJeeves.calculate_hooke_jeeves(f=Funkcije.f3, print_progress=True).print_matrix()

    print(f"Funkcija f4:")

    # print(f"\n\nPretraživanje po koordinatnim osima:")
    # pretrPoKoord: PretrazivanjePoKoordinatnimOsima = PretrazivanjePoKoordinatnimOsima(
    #     x0=Matrica(elements=[[5.1, 1.1]]),
    #     n=2)
    # pretrPoKoord.coordinate_search(f=Funkcije.f4, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex:")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[5.1, 1.1]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f4, print_progress=True).print_matrix()

    print(f"\n\nHooke-Jeeves simplex:")
    hookeJeeves: HookeJeeves = HookeJeeves(x0=Matrica(elements=[[5.1, 1.1]]))
    hookeJeeves.calculate_hooke_jeeves(f=Funkcije.f4, print_progress=True).print_matrix()


def zad3() -> None:
    print(f"\n\nHooke-Jeeves simplex:")
    hookeJeeves: HookeJeeves = HookeJeeves(x0=Matrica(elements=[[5.1, 1.1]]))
    hookeJeeves.calculate_hooke_jeeves(f=Funkcije.f4, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex:")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[5.1, 1.1]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f4, print_progress=True).print_matrix()


def zad4() -> None:
    print(f"\n\nNelder-Meadu simplex (0.5, 0.5):")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[0.5, 0.5]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex (3, 3):")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[3, 3]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex (5, 5):")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[5, 5]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex (8, 8):")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[20, 20]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex (10, 10):")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[10, 10]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex (15, 15):")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[15, 15]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()

    print(f"\n\nNelder-Meadu simplex (20, 20):")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[20, 20]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f1, print_progress=True).print_matrix()


def zad5() -> None:
    print(f"\n\nNelder-Meadu simplex:")
    nelderMeadu: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[0, 0]]))
    nelderMeadu.calculate_nelder_meadu_simplex(f=Funkcije.f6, print_progress=True).print_matrix()

    print(f"\n\nHooke-Jeeves simplex:")
    hookeJeeves: HookeJeeves = HookeJeeves(x0=Matrica(elements=[[0, 0]]))
    hookeJeeves.calculate_hooke_jeeves(f=Funkcije.f6, print_progress=True).print_matrix()


def main() -> None:
    # zad1()
    # zad2()
    # zad3()
    # zad4()
    zad5()


if __name__ == "__main__":
    main()
