"""
:author: Tin Jukić
"""
from Matrica import Matrica
from PostupciIFunkcije import NelderMeaduSimplex


def main() -> None:
    a: Matrica = Matrica(elements=[[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])
    b: Matrica = Matrica(elements=[[2.1, 3.2, 4.3], [5.4, 6.5, 7.6]])

    print(a < b)  # works as expected

    c: Matrica = Matrica(elements=[[1.1, -2.2, -3.3], [-4.4, 5.5, 6.6]])
    abs_c = abs(c)  # works as expected
    abs_c.print_matrix()

    for element in c:
        print(element)

    d: Matrica = Matrica(elements=[[4, -5], [4, -5]])
    pow(d, 2)
    d.print_matrix()

    # nm: NelderMeaduSimplex = NelderMeaduSimplex(x0=Matrica(elements=[[0, 0, 0, 0]]))
    # for el in nm.calculate_starting_points():
    #     el.print_matrix()

    b: Matrica = Matrica(elements=[[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])
    a: Matrica = Matrica(elements=[[2.1, 3.2, 4.3], [5.4, 6.5, 7.6]])

    print(a >= b)
    print(a > b)


if __name__ == "__main__":
    main()
