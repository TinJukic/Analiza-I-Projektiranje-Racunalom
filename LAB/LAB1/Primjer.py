"""
:author: Tin Jukic
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


if __name__ == '__main__':
    main()
