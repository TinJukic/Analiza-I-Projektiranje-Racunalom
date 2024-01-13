"""
:author: Tin Jukić
"""
from Matrica import Matrica
from PostupciIFunkcije import Box
from PostupciIFunkcije import Funkcije
from PostupciIFunkcije import Ogranicenja
from PostupciIFunkcije import TransformacijaBezOgranicenja


def zad1() -> None:
    print(f"ZAD1", end="\n\n")
    print(f"F1")
    box: Box = Box(
        x0=Matrica(elements=[[-0.1, 2]]),
        implicit=[Ogranicenja.implicit_1_1, Ogranicenja.implicit_1_2],
        explicit=[Ogranicenja.explicit_1_1],
        explicit_values=[-100, 100]
    )
    result: Matrica | None = box.calculate(f=Funkcije.f1)
    if result is not None:
        result.print_matrix()
    else:
        print("Result is None!")

    print(f"\n\nF2")
    box: Box = Box(
        x0=Matrica(elements=[[0.1, 0.3]]),
        implicit=[Ogranicenja.implicit_1_1, Ogranicenja.implicit_1_2],
        explicit=[Ogranicenja.explicit_1_1],
        explicit_values=[-100, 100]
    )
    result: Matrica | None = box.calculate(f=Funkcije.f2)
    if result is not None:
        result.print_matrix()
    else:
        print("Result is None!")


def zad2() -> None:
    print(f"ZAD2", end="\n\n")
    print(f"F1")

    start_point: Matrica = Matrica(elements=[[-1.9, 2]])
    # start_point: Matrica = Matrica(elements=[[-0.1, 2]])
    print(f"Start point:")
    start_point.print_matrix()

    transformacija: TransformacijaBezOgranicenja = TransformacijaBezOgranicenja()
    result: Matrica = transformacija.transform(
        x=start_point, f=Funkcije.f1, g=[Ogranicenja.implicit_2_1, Ogranicenja.implicit_2_2], h=None
    )

    print(f"Solution:")
    result.print_matrix()

    print(f"F2")
    start_point: Matrica = Matrica(elements=[[0.1, 0.3]])
    print(f"Start point:")
    start_point.print_matrix()

    result: Matrica = transformacija.transform(
        x=start_point, f=Funkcije.f2, g=[Ogranicenja.implicit_2_1, Ogranicenja.implicit_2_2], h=None
    )
    print(f"Solution:")
    result.print_matrix()


def zad3() -> None:
    print(f"ZAD3", end="\n\n")
    print(f"F4")

    start_point: Matrica = Matrica(elements=[[0, 0]])
    print(f"Start point:")
    start_point.print_matrix()

    transformacija: TransformacijaBezOgranicenja = TransformacijaBezOgranicenja()
    result: Matrica = transformacija.transform(
        x=start_point,
        f=Funkcije.f4,
        g=[Ogranicenja.implicit_3_1, Ogranicenja.implicit_3_2],
        h=[Ogranicenja.explicit_3_1]
    )

    print(f"Solution:")
    result.print_matrix()


def main() -> None:
    zad1()
    # zad2()
    # zad3()


if __name__ == "__main__":
    main()
