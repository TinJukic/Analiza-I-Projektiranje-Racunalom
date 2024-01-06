"""
:author: Tin Jukić
"""
from Matrica import Matrica
from PostupciIFunkcije import Box
from PostupciIFunkcije import Funkcije
from PostupciIFunkcije import Ogranicenja


def zad1() -> None:
    print(f"ZAD1", end="\n\n")
    print(f"F1")
    box: Box = Box(
        x0=Matrica(elements=[[-1.9, 2]]),
        implicit=[Ogranicenja.implicit_1_1, Ogranicenja.implicit_1_2],
        explicit=[Ogranicenja.explicit_1_1],
        explicit_values=[-100, 100]
    )
    result: Matrica | None = box.calculate(f=Funkcije.f1)
    if result is not None:
        result.print_matrix()
    else:
        print("Result is None!")

    # print(f"\n\nF2")
    # box: Box = Box(
    #     x0=Matrica(elements=[[0.1, 0.3]]),
    #     implicit=[Ogranicenja.implicit_1_1, Ogranicenja.implicit_1_2],
    #     explicit=[Ogranicenja.explicit_1_1],
    #     explicit_values=[-100, 100]
    # )
    # result: Matrica | None = box.calculate(f=Funkcije.f2)
    # if result is not None:
    #     result.print_matrix()
    # else:
    #     print("Result is None!")


def zad2() -> None:
    ...


def zad3() -> None:
    ...


def main() -> None:
    zad1()
    # zad2()
    # zad3()


if __name__ == "__main__":
    main()
