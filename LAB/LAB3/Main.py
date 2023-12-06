"""
:author: Tin Jukić
"""
from Matrica import Matrica
from PostupciIFunkcije import Funkcije, GradijentniSpust, NewtonRaphson, GaussNewton


def proba() -> None:
    # grSp: GradijentniSpust = GradijentniSpust(
    #     x0=Matrica(elements=[[0.1, 0.3]]),
    #     f=Funkcije.f2,
    #     f_der1_x1=Funkcije.f2_der1_x1,
    #     f_der1_x2=Funkcije.f2_der1_x2,
    #     f_lambda=Funkcije.f2_lambda,
    #     f_lambda_der=Funkcije.f2_lambda_der,
    #     use_golden_section=True
    # )
    #
    # result: Matrica | None = grSp.calculate()
    # if result is not None:
    #     result.print_matrix()

    # nr: NewtonRaphson = NewtonRaphson(
    #     x0=Matrica(elements=[[0.1, 0.3]]),
    #     f=Funkcije.f2,
    #     f_der1_x1=Funkcije.f2_der1_x1,
    #     f_der1_x2=Funkcije.f2_der1_x2,
    #     f_der2_x1=Funkcije.f2_der2_x1,
    #     f_der2_x2=Funkcije.f2_der2_x2,
    #     f_lambda=Funkcije.f2_lambda,
    #     f_lambda_der=Funkcije.f2_lambda_der,
    #     use_golden_section=True
    # )
    #
    # result: Matrica | None = nr.calculate()
    # if result is not None:
    #     result.print_matrix()

    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[-2, 2]]),
        f1=Funkcije.f5_1,
        f1_der1_x1=Funkcije.f5_1_der1_x1,
        f1_der1_x2=Funkcije.f5_1_der1_x2,
        f2=Funkcije.f5_2,
        f2_der1_x1=Funkcije.f5_2_der1_x1,
        f2_der1_x2=Funkcije.f5_2_der1_x2,
        f_lambda=Funkcije.f5_lambda,
        use_golden_section=True
    )

    result: Matrica | None = gn.calculate()
    if result is not None:
        result.print_matrix()


def zad1() -> None:
    ...


def main() -> None:
    proba()


if __name__ == "__main__":
    main()
