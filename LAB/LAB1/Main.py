"""
File for lab implementation testing.
:author: Tin Jukić
"""
from LAB1.Matrica import Matrica


def zad1() -> None:
    print(f"ZAD1", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad1_1.txt")
    A.print_matrix()

    B: Matrica = Matrica(A.get_elements())
    B.print_matrix()

    print(f"A == B ==> {A == B}", end="\n\n")

    A *= 2.5
    A.print_matrix()

    A /= 3
    A.print_matrix()

    print(f"A == B ==> {A == B}", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad1_2.txt")
    A.print_matrix()

    B: Matrica = Matrica(A.get_elements())
    B.print_matrix()

    print(f"A == B ==> {A == B}")


def main() -> None:
    zad1()


if __name__ == "__main__":
    main()
