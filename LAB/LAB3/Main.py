"""
:author: Tin Jukić
"""
from Matrica import Matrica
from PostupciIFunkcije import Funkcije, GradijentniSpust, NewtonRaphson, GaussNewton


def proba() -> None:
    grSp: GradijentniSpust = GradijentniSpust(
        x0=Matrica(elements=[[0.1, 0.3]]),
        f=Funkcije.f2,
        f_der1_x1=Funkcije.f2_der1_x1,
        f_der1_x2=Funkcije.f2_der1_x2,
        f_lambda=Funkcije.f2_lambda,
        f_lambda_der=Funkcije.f2_lambda_der,
        use_golden_section=True
    )

    result: Matrica | None = grSp.calculate()
    if result is not None:
        result.print_matrix()

    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[0.1, 0.3]]),
        f=Funkcije.f2,
        f_der1_x1=Funkcije.f2_der1_x1,
        f_der1_x2=Funkcije.f2_der1_x2,
        f_der2_x1=Funkcije.f2_der2_x1,
        f_der2_x2=Funkcije.f2_der2_x2,
        f_lambda=Funkcije.f2_lambda,
        f_lambda_der=Funkcije.f2_lambda_der,
        use_golden_section=True
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()

    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[-2, 2]]),
        f=Funkcije.f5,
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
    print(f"Gradijentni spust bez zlatnog reza.")
    grSp: GradijentniSpust = GradijentniSpust(
        x0=Matrica(elements=[[0, 0]]),
        f=Funkcije.f3,
        f_der1_x1=Funkcije.f3_der1_x1,
        f_der1_x2=Funkcije.f3_der1_x2,
        f_lambda=Funkcije.f3_lambda
    )

    result: Matrica | None = grSp.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\nGradijentni spust uz zlatni rez.")
    grSp: GradijentniSpust = GradijentniSpust(
        x0=Matrica(elements=[[0, 0]]),
        f=Funkcije.f3,
        f_der1_x1=Funkcije.f3_der1_x1,
        f_der1_x2=Funkcije.f3_der1_x2,
        f_lambda=Funkcije.f3_lambda,
        use_golden_section=True
    )

    result: Matrica | None = grSp.calculate()
    if result is not None:
        result.print_matrix()


def zad2() -> None:
    print(f"FUNKCIJA 1")

    print(f"Gradijentni spust bez zlatnog reza.")
    grSp: GradijentniSpust = GradijentniSpust(
        x0=Matrica(elements=[[-1.9, 2]]),
        f=Funkcije.f1,
        f_der1_x1=Funkcije.f1_der1_x1,
        f_der1_x2=Funkcije.f1_der1_x2,
        f_lambda=Funkcije.f1_lambda
    )

    result: Matrica | None = grSp.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\nGradijentni spust uz zlatni rez.")
    grSp: GradijentniSpust = GradijentniSpust(
        x0=Matrica(elements=[[-1.9, 2]]),
        f=Funkcije.f1,
        f_der1_x1=Funkcije.f1_der1_x1,
        f_der1_x2=Funkcije.f1_der1_x2,
        f_lambda=Funkcije.f1_lambda,
        use_golden_section=True
    )

    result: Matrica | None = grSp.calculate()
    if result is not None:
        result.print_matrix()

    print(f"Newton-Raphson spust bez zlatnog reza.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[-1.9, 2]]),
        f=Funkcije.f1,
        f_der1_x1=Funkcije.f1_der1_x1,
        f_der1_x2=Funkcije.f1_der1_x2,
        f_der1_x1_x2=Funkcije.f1_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f1_der1_x2_x1,
        f_der2_x1=Funkcije.f1_der2_x1,
        f_der2_x2=Funkcije.f1_der2_x2,
        f_lambda=Funkcije.f1_lambda
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\nNewton-Raphson uz zlatni rez.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[-1.9, 2]]),
        f=Funkcije.f1,
        f_der1_x1=Funkcije.f1_der1_x1,
        f_der1_x2=Funkcije.f1_der1_x2,
        f_der1_x1_x2=Funkcije.f1_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f1_der1_x2_x1,
        f_der2_x1=Funkcije.f1_der2_x1,
        f_der2_x2=Funkcije.f1_der2_x2,
        f_lambda=Funkcije.f1_lambda,
        use_golden_section=True
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\n\nFUNKCIJA 2")

    print(f"Gradijentni spust bez zlatnog reza.")
    grSp: GradijentniSpust = GradijentniSpust(
        x0=Matrica(elements=[[0.1, 0.3]]),
        f=Funkcije.f2,
        f_der1_x1=Funkcije.f2_der1_x1,
        f_der1_x2=Funkcije.f2_der1_x2,
        f_lambda=Funkcije.f2_lambda,
        f_lambda_der=Funkcije.f2_lambda_der
    )

    result: Matrica | None = grSp.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\nGradijentni spust uz zlatni rez.")
    grSp: GradijentniSpust = GradijentniSpust(
        x0=Matrica(elements=[[0.1, 0.3]]),
        f=Funkcije.f2,
        f_der1_x1=Funkcije.f2_der1_x1,
        f_der1_x2=Funkcije.f2_der1_x2,
        f_lambda=Funkcije.f2_lambda,
        f_lambda_der=Funkcije.f2_lambda_der,
        use_golden_section=True
    )

    result: Matrica | None = grSp.calculate()
    if result is not None:
        result.print_matrix()

    print(f"Newton-Raphson spust bez zlatnog reza.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[0.1, 0.3]]),
        f=Funkcije.f2,
        f_der1_x1=Funkcije.f2_der1_x1,
        f_der1_x2=Funkcije.f2_der1_x2,
        f_der1_x1_x2=Funkcije.f2_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f2_der1_x2_x1,
        f_der2_x1=Funkcije.f2_der2_x1,
        f_der2_x2=Funkcije.f2_der2_x2,
        f_lambda=Funkcije.f2_lambda,
        f_lambda_der=Funkcije.f2_lambda_der
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\nNewton-Raphson uz zlatni rez.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[0.1, 0.3]]),
        f=Funkcije.f2,
        f_der1_x1=Funkcije.f2_der1_x1,
        f_der1_x2=Funkcije.f2_der1_x2,
        f_der1_x1_x2=Funkcije.f2_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f2_der1_x2_x1,
        f_der2_x1=Funkcije.f2_der2_x1,
        f_der2_x2=Funkcije.f2_der2_x2,
        f_lambda=Funkcije.f2_lambda,
        f_lambda_der=Funkcije.f2_lambda_der,
        use_golden_section=True
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()


def zad3() -> None:
    print(f"\nNewton-Raphson uz zlatni rez.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[3, 3]]),
        f=Funkcije.f4,
        f_der1_x1=Funkcije.f4_der1_x1,
        f_der1_x2=Funkcije.f4_der1_x2,
        f_der1_x1_x2=Funkcije.f2_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f2_der1_x2_x1,
        f_der2_x1=Funkcije.f4_der2_x1,
        f_der2_x2=Funkcije.f4_der2_x2,
        f_lambda=Funkcije.f4_lambda,
        use_golden_section=True
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\nNewton-Raphson uz zlatni rez.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[1, 2]]),
        f=Funkcije.f4,
        f_der1_x1=Funkcije.f4_der1_x1,
        f_der1_x2=Funkcije.f4_der1_x2,
        f_der1_x1_x2=Funkcije.f2_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f2_der1_x2_x1,
        f_der2_x1=Funkcije.f4_der2_x1,
        f_der2_x2=Funkcije.f4_der2_x2,
        f_lambda=Funkcije.f4_lambda,
        use_golden_section=True
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()

    print(f"Newton-Raphson bez zlatnog reza.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[3, 3]]),
        f=Funkcije.f4,
        f_der1_x1=Funkcije.f4_der1_x1,
        f_der1_x2=Funkcije.f4_der1_x2,
        f_der1_x1_x2=Funkcije.f2_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f2_der1_x2_x1,
        f_der2_x1=Funkcije.f4_der2_x1,
        f_der2_x2=Funkcije.f4_der2_x2,
        f_lambda=Funkcije.f4_lambda
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()

    print(f"\nNewton-Raphson bez zlatnog reza.")
    nr: NewtonRaphson = NewtonRaphson(
        x0=Matrica(elements=[[1, 2]]),
        f=Funkcije.f4,
        f_der1_x1=Funkcije.f4_der1_x1,
        f_der1_x2=Funkcije.f4_der1_x2,
        f_der1_x1_x2=Funkcije.f2_der1_x1_x2,
        f_der1_x2_x1=Funkcije.f2_der1_x2_x1,
        f_der2_x1=Funkcije.f4_der2_x1,
        f_der2_x2=Funkcije.f4_der2_x2,
        f_lambda=Funkcije.f4_lambda
    )

    result: Matrica | None = nr.calculate()
    if result is not None:
        result.print_matrix()


def zad4() -> None:
    print(f"Gauss-Newton bez zlatnog reza.")
    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[-1.9, 2]]),
        f=Funkcije.f1,
        f1=Funkcije.f1_1,
        f1_der1_x1=Funkcije.f1_1_der1_x1,
        f1_der1_x2=Funkcije.f1_1_der1_x2,
        f2=Funkcije.f1_2,
        f2_der1_x1=Funkcije.f1_2_der1_x1,
        f2_der1_x2=Funkcije.f1_2_der1_x2,
        f_lambda=Funkcije.f1_lambda
    )

    result: Matrica | None = gn.calculate()
    if result is not None:
        result.print_matrix()

    print(f"Gauss-Newton uz zlatni rez.")
    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[-1.9, 2]]),
        f=Funkcije.f1,
        f1=Funkcije.f1_1,
        f1_der1_x1=Funkcije.f1_1_der1_x1,
        f1_der1_x2=Funkcije.f1_1_der1_x2,
        f2=Funkcije.f1_2,
        f2_der1_x1=Funkcije.f1_2_der1_x1,
        f2_der1_x2=Funkcije.f1_2_der1_x2,
        f_lambda=Funkcije.f1_lambda,
        use_golden_section=True
    )

    result: Matrica | None = gn.calculate()
    if result is not None:
        result.print_matrix()


def zad5() -> None:
    print(f"Gauss-Newton uz zlatni rez.")
    print(f"x0 = (-2, 2)")
    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[-2, 2]]),
        f=Funkcije.f5,
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

    print(f"Gauss-Newton uz zlatni rez.")
    print(f"x0 = (2, 2)")
    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[2, 2]]),
        f=Funkcije.f5,
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

    print(f"\nGauss-Newton uz zlatni rez.")
    print(f"x0 = (2, -2)")
    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[2, -2]]),
        f=Funkcije.f5,
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


def zad6() -> None:
    print(f"\nGauss-Newton uz zlatni rez.")

    elements: list[tuple[float, float]] = [(1, 3), (2, 4), (3, 4), (5, 5), (6, 6), (7, 8)]
    fs = []
    der_fs = []
    f_lams = Funkcije.f6_lambda(elements=elements)

    for t, y in elements:
        fs.append(Funkcije.f6(t=t, y=y))
        der_fs.append(
            (Funkcije.f6_1_der1_x1(t=t, y=y),
             Funkcije.f6_1_der1_x2(t=t, y=y),
             Funkcije.f6_1_der1_x3(t=t, y=y))
        )

    gn: GaussNewton = GaussNewton(
        x0=Matrica(elements=[[1, 1, 1]]),
        f=Funkcije.f6(t=t, y=y),
        f1=Funkcije.f6_1(t=t, y=y),
        f1_der1_x1=Funkcije.f6_1_der1_x1(t=t, y=y),
        f1_der1_x2=Funkcije.f6_1_der1_x2(t=t, y=y),
        f2=Funkcije.f6_2(t=t, y=y),
        f2_der1_x1=Funkcije.f6_2_der1_x1(t=t, y=y),
        f2_der1_x2=Funkcije.f6_2_der1_x2(t=t, y=y),
        use_golden_section=True
    )

    result: Matrica | None = gn.calculate_with_golden_section_3_points(fs=fs, f_der=der_fs, f_lams=f_lams)
    if result is not None:
        result.print_matrix()


def main() -> None:
    # proba()
    # zad1()
    # zad2()
    # zad3()
    zad4()
    # zad5()
    # zad6()


if __name__ == "__main__":
    main()
