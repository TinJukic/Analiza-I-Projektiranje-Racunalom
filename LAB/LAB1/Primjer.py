"""
:author: Tin Jukić
"""
from LAB1.Matrica import Matrica


def main():
    A = Matrica.load_matrix_from_file('A.txt')
    print("A")
    A.print_matrix()
    B = Matrica.load_matrix_from_file('B.txt')
    print("B")
    B.print_matrix()
    C = ~A
    print("C=~A")
    print(C == ~A)
    C.print_matrix()
    # print("B*2")
    # (B*2).print_matrix()
    # print("(A-B*2)")
    # (A-B*2).print_matrix()
    # print("B*(A-B*2)")
    # (B*(A-B*2)).print_matrix()
    # print("A*0.5")
    # (A*0.5).print_matrix()
    # print("A*0.5*B*(A-B*2)")
    # (A*0.5*B*(A-B*2)).print_matrix()
    C += A * 0.5 * B * (A - B * 2)
    C.print_matrix()
    x = C.get_element_at(position=(0, 0))
    print(f"Element at index (0, 0) = {x}")
    C.set_element_at(position=(1, 1), element=x)
    C.print_matrix()

    print(C == A)
    print(C == ~A)
    print(A == A)

    C.save_matrix_to_file(f"PrimjerCSave.txt")

    print(f"\nC-=C")
    C -= C
    C.print_matrix()

    print(f"\nA:")
    A.print_matrix()
    P = Matrica.identity_matrix(A.get_matrix_dimension())
    print(f"Identity matrix")
    P.print_matrix()
    n = A.switch_rows(P, 1, 2)  # rows 2 and 3 are switched
    n = A.switch_rows(P, 0, 1, n)
    print(f"Number of transformations = {n}")
    A.print_matrix()
    print(f"P:")
    P.print_matrix()

    print(f"Row-vectors of P")
    row_vectors_P = P.to_row_vectors()
    print(f"len_row_vectors_P = {len(row_vectors_P)}")
    print(row_vectors_P)
    for matrica in row_vectors_P:
        matrica.print_matrix()

    print(f"P to row-vectors")
    P = Matrica.row_vectors_to_matrix(row_vectors_P)
    P.print_matrix()

    A = Matrica.load_matrix_from_file('A.txt')
    print(f"A:")
    A.print_matrix()
    print(f"A/=2")
    A /= 2
    A.print_matrix()

    print(f"LUP => matA * x = matb")
    A: Matrica = Matrica.load_matrix_from_file("matA.txt")
    A.print_matrix()
    b: Matrica = Matrica.load_matrix_from_file("matb.txt")
    b.print_matrix()

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

    print(f"LU => matC * x = matd")
    C: Matrica = Matrica.load_matrix_from_file("matC.txt")
    C.print_matrix()
    d: Matrica = Matrica.load_matrix_from_file("matd.txt")
    d.print_matrix()

    print(f"LU decomposition:")
    LU = C.LU_decomposition()
    if LU is not None:
        LU.print_matrix()

        y: Matrica = LU.forward_substitution(d)
        y.print_matrix()

        x: Matrica = LU.backward_substitution(y)
        x.print_matrix()
    else:
        C.print_matrix()
        print(f"LU decomposition is not possible!", end="\n\n")

    print(f"Matrix inverse")
    A: Matrica = Matrica.load_matrix_from_file("inv_A.txt")
    A.print_matrix()

    inv_A = A.inversion()

    if inv_A is None:
        print(f"Inverse cannot be calculated!")
    else:
        inv_A.print_matrix()

    print(f"Matrix determinant")
    A: Matrica = Matrica.load_matrix_from_file("det_A.txt")
    A.print_matrix()

    det_A = A.determinant()

    if det_A is None:
        print(f"Determinant cannot be calculated!")
    else:
        print(f"det(A) = {det_A}")


if __name__ == '__main__':
    main()
