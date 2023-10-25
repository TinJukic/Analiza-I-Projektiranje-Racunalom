"""
File for lab implementation testing.
:author: Tin Jukić
"""
import sys

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

    print(f"LU decomposition:")
    LU = A.LU_decomposition()
    if LU is not None:
        LU.print_matrix()
    else:
        # A.print_matrix()
        print(f"LU decomposition is not possible!", end="\n\n")

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


def zad3() -> None:
    print(f"ZAD3", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad3A.txt")
    A.print_matrix()

    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad3b.txt")
    b.print_matrix()

    print(f"LU decomposition:")
    LU = A.LU_decomposition()
    try:
        if LU is not None:
            LU.print_matrix()

            y: Matrica = LU.forward_substitution(b)
            y.print_matrix()

            x: Matrica = LU.backward_substitution(y)
            x.print_matrix()
        else:
            print(f"LU decomposition is not possible!", end="\n\n")
    except ZeroDivisionError:
        sys.stderr.write(f"Cannot calculate x using LU! Pivot element cannot be zero!\n")

    print(f"LUP decomposition:")
    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad3A.txt")
    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad3b.txt")
    LUP = A.LUP_decomposition()
    try:
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
    except ZeroDivisionError:
        sys.stderr.write(f"Cannot calculate x using LUP! Pivot element cannot be zero!\n")


def zad4() -> None:
    print(f"ZAD4", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad4A.txt")
    A.print_matrix()

    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad4b.txt")
    b.print_matrix()

    print(f"LU decomposition:")
    LU = A.LU_decomposition()
    if LU is not None:
        LU.print_matrix()

        y: Matrica = LU.forward_substitution(b)
        y.print_matrix()

        x: Matrica = LU.backward_substitution(y)
        x.print_matrix()
    else:
        print(f"LU decomposition is not possible!", end="\n\n")

    print(f"LUP decomposition:")
    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad4A.txt")
    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad4b.txt")
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


def zad5() -> None:
    print(f"ZAD5", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad5A.txt")
    A.print_matrix()

    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad5b.txt")
    b.print_matrix()

    print(f"LU decomposition:")
    LU = A.LU_decomposition()
    if LU is not None:
        LU.print_matrix()

        y: Matrica = LU.forward_substitution(b)
        y.print_matrix()

        x: Matrica = LU.backward_substitution(y)
        x.print_matrix()
    else:
        print(f"LU decomposition is not possible!", end="\n\n")

    print(f"LUP decomposition:")
    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad5A.txt")
    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad5b.txt")
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


def zad6() -> None:
    print(f"ZAD6", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad6A.txt")
    A.print_matrix()

    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad6b.txt")
    b.print_matrix()

    print(f"LU decomposition:")
    LU = A.LU_decomposition()
    try:
        if LU is not None:
            LU.print_matrix()

            y: Matrica = LU.forward_substitution(b)
            y.print_matrix()

            x: Matrica = LU.backward_substitution(y)
            x.print_matrix()
        else:
            print(f"LU decomposition is not possible!", end="\n\n")
    except ZeroDivisionError:
        sys.stderr.write(f"Cannot calculate x using LU! Pivot element cannot be zero!\n")

    print(f"LUP decomposition:")
    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad6A.txt")
    b: Matrica = Matrica.load_matrix_from_file("Matrices/zad6b.txt")
    LUP = A.LUP_decomposition()
    try:
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
    except ZeroDivisionError:
        sys.stderr.write(f"Cannot calculate x using LUP! Pivot element cannot be zero!\n")


def zad7() -> None:
    print(f"ZAD7", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad7A.txt")
    A.print_matrix()

    A.inversion()
    # A.print_matrix()


def zad8() -> None:
    print(f"ZAD8", end="\n\n")

    A: Matrica = Matrica.load_matrix_from_file("Matrices/zad8A.txt")
    A.print_matrix()

    inv_A = A.inversion()
    inv_A.print_matrix()


def main() -> None:
    # zad1()
    # zad2()
    # zad3()
    # zad4()
    # zad5()
    # zad6()
    # zad7()
    zad8()


if __name__ == "__main__":
    main()
