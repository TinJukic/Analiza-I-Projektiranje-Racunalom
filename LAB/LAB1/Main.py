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


def zad2() -> None:
    print(f"ZAD2", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad2A.txt")
    A.print_matrix()

    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad2b.txt")
    b.print_matrix()

    # print(f"LU decomposition:")
    # LU = A.LU_decomposition()
    # if LU is not None:
    #     LU.print_matrix()
    # else:
    #     A.print_matrix()
    #     print(f"LU decomposition is not possible!", end="\n\n")

    print(f"LUP decomposition:")
    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad2A.txt")
    LUP = A.LUP_decomposition()
    if LUP is not None:
        A, P, n = LUP
        A.print_matrix()
        P.print_matrix()
        print(f"Number of transforms = {n}", end="\n\n")

        perm_b = P * b
        perm_b.print_matrix()

        y: Matrica = A.forward_substitution(perm_b)
        y.print_matrix()

        x: Matrica = A.backward_substitution(y)
        x.print_matrix()
    else:
        A.print_matrix()
        print(f"LUP decomposition is not possible!", end="\n\n")


def main() -> None:
    # zad1()
    zad2()


if __name__ == "__main__":
    main()
