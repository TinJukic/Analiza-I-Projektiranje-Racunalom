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


if __name__ == '__main__':
    main()
