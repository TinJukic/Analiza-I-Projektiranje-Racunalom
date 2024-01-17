"""
:author: Tin Jukić
"""
from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy

from Matrica import Matrica


class Funkcije:
    """
    Implemented all functions for this exercise.
    """

    @staticmethod
    def f1(x: Matrica, t: float) -> Matrica:
        x1: float = x.get_element_at(position=(0, 0)) * math.cos(t) + x.get_element_at(position=(0, 1)) * math.sin(t)
        x2: float = x.get_element_at(position=(0, 1)) * math.cos(t) - x.get_element_at(position=(0, 0)) * math.sin(t)
        return Matrica(elements=[[x1, x2]])


class Loader:
    """
    Class which loads all parameters from file.
    """

    @staticmethod
    def load_from(file: str, without_B: bool = False) -> tuple[Matrica, Matrica | None, Matrica]:
        """
        Loads data from a file.
        :param file: from which the data is loaded
        :param without_B: determines whether the B matrix is passed or not
        :return: data loaded from file
        :throws IndexError: if the file doesn't have the correct number of elements
        """
        A: Matrica = Matrica()
        B: Matrica | None = None  # does not always need to be passed
        x: Matrica = Matrica()

        with open(file, 'r', encoding="utf-8") as f:
            lines = [line.strip().split() for line in f.readlines() if line.strip() != ""]

            # check whether the correct number of elements is passed
            if (len(lines) != 3 or not without_B) and (len(lines) != 5 or without_B):
                raise IndexError(f"Number of passed elements is not correct")
            for i, line in enumerate(lines, start=1):
                if len(line) != 2:  # should always have 2 elements
                    raise IndexError(f"Number of passed elements in line {i} is not correct")

            A = Matrica(elements=[
                [float(lines[0][0]), float(lines[0][1])],
                [float(lines[1][0]), float(lines[1][1])]]
            )
            if not without_B:
                B = Matrica(elements=[
                    [float(lines[2][0]), float(lines[2][1])],
                    [float(lines[3][0]), float(lines[3][1])]
                ])
                x = Matrica(elements=[[float(lines[4][0]), float(lines[4][1])]])
            else:
                x = Matrica(elements=[[float(lines[2][0]), float(lines[2][1])]])

        return A, B, x

    @staticmethod
    def save_to(file: str, data: list[Matrica]) -> None:
        """
        Saves data to a file.
        :param file: at which the data is saved
        :param data: to be saved
        :return: None
        """
        with open(file, 'w', encoding="utf-8") as f:
            for element in data:
                f.write(str(element) + "\n")

    @staticmethod
    def load_solution_from(file: str) -> list[Matrica]:
        """
        Loads solution from a file.
        :param file: from which the solution is loaded
        :return: list of points representing a solution
        """
        with open(file, 'r', encoding="utf-8") as f:
            lines: list[list[str]] = [line.strip().split("\t") for line in f.readlines() if line.strip() != ""]
            result = [Matrica(elements=[[float(element[0]), float(element[1])]]) for element in lines]
        return result


class Drawer:
    """
    Class which draws the results of methods.
    """

    @staticmethod
    def draw_from(file: str, title: str, t_max: int, T: float) -> None:
        """
        Draws results of method.
        :param file: from which the data is loaded
        :param title: graph title
        :param t_max: time interval upper limit -> [0, t_max]
        :param T: integration step
        :return: None
        """
        res: list[Matrica] = Loader.load_solution_from(file=file)
        Drawer.draw_using(data=res, title=title, t_max=t_max, T=T)

    @staticmethod
    def draw_using(data: list[Matrica], title: str, t_max: int, T: float) -> None:
        """
        Draws results of method.
        :param data: to plot
        :param title: graph title
        :param t_max: time interval upper limit -> [0, t_max]
        :param T: integration step
        :return: None
        """
        x1: list[float] = [element.get_element_at(position=(0, 0)) for i, element in enumerate(data)]
        x2: list[float] = [element.get_element_at(position=(0, 1)) for i, element in enumerate(data)]
        time = numpy.linspace(0, t_max, int(t_max / T))

        Drawer.__draw(x1=x1, x2=x2, time=time, title=title)

    @staticmethod
    def __draw(x1: list[float], x2: list[float], time: numpy.linspace, title: str) -> None:
        """
        Draws result.
        :param x1: first variable
        :param x2: second variable
        :param time: interval
        :param title: graph title
        :return: None
        """
        plt.figure(figsize=(30, 30))
        plt.title(title)
        plt.plot(time, x1, label=f"x1")
        plt.plot(time, x2, label=f"x2")
        plt.legend(loc=f"best")
        plt.show()


class Euler:
    """
    Euler's method class.
    """

    @staticmethod
    def calculate(
            A: Matrica,
            B: Matrica | None,
            x0: Matrica,
            f_real: any,
            T: float,
            t_max: int,
            r: Matrica | None = None,
            update_r: bool = False,
            print_after: int = 100
    ) -> list[Matrica]:
        """
        Euler's method.
        :param A: matrix of the function
        :param B: matrix of the function
        :param x0: starting point at t=0
        :param f_real: real function used to calculate real new points | None if it should not be used
        :param T: integration step
        :param t_max: time interval upper limit -> [0, t_max]
        :param r: matrix used to calculate new points
        :param update_r: determines whether r value should be updated or not
        :param print_after: after how many iterations to print current solution
        :return: list of calculated matrices
        """
        result: list[Matrica] = []
        real_result: list[Matrica] = []

        x: Matrica = Matrica(elements=x0.get_elements())
        current_print_after: int = 0

        for t in numpy.linspace(0, t_max, int(t_max / T)):
            if f_real is not None:
                real_result.append(Euler.__calculate_real_next_point(xk=x0, f_real=f_real, T=T, t=t))

            if update_r and r is not None:
                for i in range(len(r.get_elements()[0])):
                    r.set_element_at(position=(0, i), element=t)
            else:
                r = Matrica(elements=[[t, t]])

            x += (Euler.__calculate_next_point(A=A, B=B, xk=x, r=r) * T)
            result.append(x)

            if current_print_after % print_after == 0:
                print(f"[{x.get_element_at(position=(0, 0))}, {x.get_element_at(position=(0, 1))}]")
            current_print_after += 1

        if f_real is not None:
            error: float = 0.0  # error at each time point
            for i in range(len(result)):
                for r in abs(result[i] - real_result[i]).get_elements()[0]:
                    error += r
            print(f"Error: {error / len(result)}")
            Drawer.draw_using(data=real_result, title=f"Euler - real solution", t_max=t_max, T=T)

        return result

    @staticmethod
    def __calculate_next_point(A: Matrica, B: Matrica | None, xk: Matrica, r: Matrica | None) -> Matrica:
        """
        Method used to calculate the next point of the Euler's method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: current point
        :return: calculated next point
        """
        a: Matrica = A * ~xk
        return ~a if B is None else ~(a + B * ~r)

    @staticmethod
    def __calculate_real_next_point(xk: Matrica, f_real: any, T: float, t: int) -> Matrica:
        """
        Method used to calculate the real next point of the Euler's method.
        :param xk: current point
        :param f_real: function used to calculate real new points
        :param T: integration step
        :param t: current time moment
        :return: calculated next real point
        """
        return xk + f_real(x=xk, t=t) * T


class ReversedEuler:
    """
    Reverse Euler's method class.
    """

    @staticmethod
    def calculate(
            A: Matrica,
            B: Matrica | None,
            x0: Matrica,
            f_real: any,
            T: float,
            t_max: int,
            r: Matrica | None = None,
            update_r: bool = False,
            print_after: int = 100
    ) -> list[Matrica]:
        """
        Reversed Euler's method.
        :param A: matrix of the function
        :param B: matrix of the function
        :param x0: starting point at t=0
        :param f_real: real function used to calculate real new points | None if it should not be used
        :param T: integration step
        :param t_max: time interval upper limit -> [0, t_max]
        :param r: matrix used to calculate new points
        :param update_r: determines whether r value should be updated or not
        :param print_after: after how many iterations to print current solution
        :return: list of calculated matrices
        """
        result: list[Matrica] = []
        real_result: list[Matrica] = []

        x: Matrica = Matrica(elements=x0.get_elements())
        current_print_after: int = 0

        for t in numpy.linspace(0, t_max, int(t_max / T)):
            if f_real is not None:
                real_result.append(ReversedEuler.__calculate_real_next_point(xk=x0, f_real=f_real, T=T, t=t))

            if update_r and r is not None:
                for i in range(len(r.get_elements()[0])):
                    r.set_element_at(position=(0, i), element=t)
            else:
                r = Matrica(elements=[[t, t]])

            x = ReversedEuler.__calculate_next_point(A=A, B=B, xk=x, T=T, r=r)
            result.append(x)

            if current_print_after % print_after == 0:
                print(f"[{x.get_element_at(position=(0, 0))}, {x.get_element_at(position=(0, 1))}]")
            current_print_after += 1

        if f_real is not None:
            error: float = 0.0  # error at each time point
            for i in range(len(result)):
                for r in abs(result[i] - real_result[i]).get_elements()[0]:
                    error += r
            print(f"Error: {error / len(result)}")
            Drawer.draw_using(data=real_result, title=f"Reversed Euler - real solution", t_max=t_max, T=T)

        return result

    @staticmethod
    def __calculate_next_point(A: Matrica, B: Matrica | None, xk: Matrica, T: float, r: Matrica | None) -> Matrica:
        """
        Method used to calculate the next point of the reversed Euler's method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: next point
        :param T: integration step
        :return: calculated next point
        """
        identity_matrix: Matrica = Matrica.identity_matrix(dimension=A.get_matrix_dimension())
        divider: Matrica = identity_matrix - A * T

        return xk * ~divider.inversion() if B is None else (xk + ~(B * T * ~r)) * ~divider.inversion()

    @staticmethod
    def __calculate_real_next_point(xk: Matrica, f_real: any, T: float, t: int) -> Matrica:
        """
        Method used to calculate the real next point of the reversed Euler's method.
        :param xk: next point
        :param f_real: function used to calculate real new points
        :param T: integration step
        :param t: current time moment
        :return: calculated next real point
        """
        return xk + f_real(x=xk, t=t) * T


class Trapeze:
    """
    Trapeze method class.
    """

    @staticmethod
    def calculate(
            A: Matrica,
            B: Matrica | None,
            x0: Matrica,
            f_real: any,
            T: float,
            t_max: int,
            r: Matrica | None = None,
            update_r: bool = False,
            print_after: int = 100
    ) -> list[Matrica]:
        """
        Trapeze method.
        :param A: matrix of the function
        :param B: matrix of the function
        :param x0: starting point at t=0
        :param f_real: real function used to calculate real new points | None if it should not be used
        :param T: integration step
        :param t_max: time interval upper limit -> [0, t_max]
        :param r: matrix used to calculate new points
        :param update_r: determines whether r value should be updated or not
        :param print_after: after how many iterations to print current solution
        :return: list of calculated matrices
        """
        result: list[Matrica] = []
        real_result: list[Matrica] = []

        x: Matrica = Matrica(elements=x0.get_elements())
        current_print_after: int = 0

        for t in numpy.linspace(0, t_max, int(t_max / T)):
            if f_real is not None:
                real_result.append(Trapeze.__calculate_real_next_point(xk=x0, f_real=f_real, T=T, t=t))

            if update_r and r is not None:
                for i in range(len(r.get_elements()[0])):
                    r.set_element_at(position=(0, i), element=t)
            else:
                r = Matrica(elements=[[t, t]])

            x = Trapeze.__calculate_next_point(A=A, B=B, xk=x, T=T, r=r)
            result.append(x)

            if current_print_after % print_after == 0:
                print(f"[{x.get_element_at(position=(0, 0))}, {x.get_element_at(position=(0, 1))}]")
            current_print_after += 1

        if f_real is not None:
            error: float = 0.0  # error at each time point
            for i in range(len(result)):
                for r in abs(result[i] - real_result[i]).get_elements()[0]:
                    error += r
            print(f"Error: {error / len(result)}")
            Drawer.draw_using(data=real_result, title=f"Trapeze - real solution", t_max=t_max, T=T)

        return result

    @staticmethod
    def __calculate_next_point(A: Matrica, B: Matrica | None, xk: Matrica, T: float, r: Matrica | None) -> Matrica:
        """
        Method used to calculate the next point of the trapeze method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: next point
        :param T: integration step
        :return: calculated next point
        """
        identity_matrix: Matrica = Matrica.identity_matrix(dimension=A.get_matrix_dimension())
        R: Matrica = ~(identity_matrix + A * T / 2) * ~(identity_matrix - A * T / 2).inversion()
        other_r: Matrica = Matrica(elements=[[t + T for t in elements] for elements in r.get_elements()])

        return xk * R if B is None else xk * R + ~((identity_matrix - A * T / 2) * T / 2 * B * ~(r + other_r))

    @staticmethod
    def __calculate_real_next_point(xk: Matrica, f_real: any, T: float, t: int) -> Matrica:
        """
        Method used to calculate the real next point of the trapeze method.
        :param xk: next point
        :param f_real: function used to calculate real new points
        :param T: integration step
        :param t: current time moment
        :return: calculated next real point
        """
        return xk + f_real(x=xk, t=t) * T


class RungeKutta:
    """
    Runge-Kutta 4th order method class.
    """

    @staticmethod
    def calculate(
            A: Matrica,
            B: Matrica | None,
            x0: Matrica,
            f_real: any,
            T: float,
            t_max: int,
            r: Matrica | None = None,
            update_r: bool = False,
            print_after: int = 100
    ) -> list[Matrica]:
        """
        Runge-Kutta 4th order method.
        :param A: matrix of the function
        :param B: matrix of the function
        :param x0: starting point at t=0
        :param f_real: real function used to calculate real new points | None if it should not be used
        :param T: integration step
        :param t_max: time interval upper limit -> [0, t_max]
        :param r: matrix used to calculate new points
        :param update_r: determines whether r value should be updated or not
        :param print_after: after how many iterations to print current solution
        :return: list of calculated matrices
        """
        result: list[Matrica] = []
        real_result: list[Matrica] = []

        x: Matrica = Matrica(elements=x0.get_elements())
        current_print_after: int = 0

        for t in numpy.linspace(0, t_max, int(t_max / T)):
            if f_real is not None:
                real_result.append(RungeKutta.__calculate_real_next_point(xk=x0, f_real=f_real, T=T, t=t))

            if update_r and r is not None:
                for i in range(len(r.get_elements()[0])):
                    r.set_element_at(position=(0, i), element=t)
            else:
                r = Matrica(elements=[[t, t]])

            x = RungeKutta.__calculate_next_point(A=A, B=B, xk=x, T=T, r=r)
            result.append(x)

            if current_print_after % print_after == 0:
                print(f"[{x.get_element_at(position=(0, 0))}, {x.get_element_at(position=(0, 1))}]")
            current_print_after += 1

        if f_real is not None:
            error: float = 0.0  # error at each time point
            for i in range(len(result)):
                for r in abs(result[i] - real_result[i]).get_elements()[0]:
                    error += r
            print(f"Error: {error / len(result)}")
            Drawer.draw_using(data=real_result, title=f"Runge-Kutta - real solution", t_max=t_max, T=T)

        return result

    @staticmethod
    def __calculate_next_point(A: Matrica, B: Matrica | None, xk: Matrica, T: float, r: Matrica | None) -> Matrica:
        """
        Method used to calculate the next point of the Runge-Kutta method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: current point
        :param T: integration step
        :return: calculated next point
        """
        return xk + (
                ~RungeKutta.__m1(A=A, B=B, xk=xk, r=r) +
                ~RungeKutta.__m2(A=A, B=B, xk=xk, r=r, T=T) * 2 +
                ~RungeKutta.__m3(A=A, B=B, xk=xk, r=r, T=T) * 2 +
                ~RungeKutta.__m4(A=A, B=B, xk=xk, r=r, T=T)
        ) * T / 6

    @staticmethod
    def __calculate_real_next_point(xk: Matrica, f_real: any, T: float, t: int) -> Matrica:
        """
        Method used to calculate the real next point of the Runge-Kutta method.
        :param xk: current point
        :param f_real: function used to calculate real new points
        :param T: integration step
        :param t: current time moment
        :return: calculated next real point
        """
        return xk + f_real(x=xk, t=t) * T

    @staticmethod
    def __m(A: Matrica, B: Matrica | None, xk: Matrica, r: Matrica) -> Matrica:
        """
        M function of the Runge-Kutta.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: current point
        :param r: vector
        :return: calculated point
        """
        return xk * A if B is None else xk * A + B * r

    @staticmethod
    def __m1(A: Matrica, B: Matrica | None, xk: Matrica, r: Matrica) -> Matrica:
        """
        M1 function of the Runge-Kutta.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: current point
        :param r: vector
        :return: calculated point
        """
        return A * ~xk if B is None else A * ~xk + B * ~r

    @staticmethod
    def __m2(A: Matrica, B: Matrica | None, xk: Matrica, r: Matrica, T: float) -> Matrica:
        """
        M2 function of the Runge-Kutta.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: current point
        :param r: vector
        :param T: integration step
        :return: calculated point
        """
        new_r: Matrica = Matrica(elements=[[t + T / 2 for t in elements] for elements in r.get_elements()])
        x: Matrica = A * ~(xk + ~RungeKutta.__m1(A=A, B=B, xk=xk, r=r) * T / 2)

        return x if B is None else x + B * ~new_r

    @staticmethod
    def __m3(A: Matrica, B: Matrica | None, xk: Matrica, r: Matrica, T: float) -> Matrica:
        """
        M3 function of the Runge-Kutta.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: current point
        :param r: vector
        :param T: integration step
        :return: calculated point
        """
        new_r: Matrica = Matrica(elements=[[t + T / 2 for t in elements] for elements in r.get_elements()])
        x: Matrica = A * ~(xk + ~RungeKutta.__m2(A=A, B=B, xk=xk, r=r, T=T) * T / 2)

        return x if B is None else x + B * ~new_r

    @staticmethod
    def __m4(A: Matrica, B: Matrica | None, xk: Matrica, r: Matrica, T: float) -> Matrica:
        """
        M4 function of the Runge-Kutta.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: current point
        :param r: vector
        :param T: integration step
        :return: calculated point
        """
        new_r: Matrica = Matrica(elements=[[t + T for t in elements] for elements in r.get_elements()])
        x: Matrica = A * ~(xk + ~RungeKutta.__m3(A=A, B=B, xk=xk, r=r, T=T) * T)

        return x if B is None else x + B * ~new_r


class PECE2:
    """
    Prediktorsko korektorski postupak (Euler + reversed Euler).
    """

    @staticmethod
    def calculate(
            A: Matrica,
            B: Matrica | None,
            x0: Matrica,
            f_real: any,
            T: float,
            t_max: int,
            r: Matrica | None = None,
            update_r: bool = False,
            print_after: int = 100
    ) -> list[Matrica]:
        """
        PE(CE)^2 method with Euler as predictor and reversed Euler as corrector.
        :param A: matrix of the function
        :param B: matrix of the function
        :param x0: starting point at t=0
        :param f_real: real function used to calculate real new points | None if it should not be used
        :param T: integration step
        :param t_max: time interval upper limit -> [0, t_max]
        :param r: matrix used to calculate new points
        :param update_r: determines whether r value should be updated or not
        :param print_after: after how many iterations to print current solution
        :return: list of calculated matrices
        """
        result: list[Matrica] = []
        real_result: list[Matrica] = []

        x: Matrica = Matrica(elements=x0.get_elements())
        current_print_after: int = 0

        for t in numpy.linspace(0, t_max, int(t_max / T)):
            if f_real is not None:
                real_result.append(PECE2.__calculate_real_next_point(xk=x0, f_real=f_real, T=T, t=t))

            if update_r and r is not None:
                for i in range(len(r.get_elements()[0])):
                    r.set_element_at(position=(0, i), element=t)
            else:
                r = Matrica(elements=[[t, t]])

            r_T: Matrica | None = None
            if r is not None:
                r_T: Matrica = Matrica(elements=[[t + T for t in elements] for elements in r.get_elements()])

            # calculating next point using Euler and reversed Euler for two times
            predicted_x: Matrica = x + PECE2.__predictor(A=A, B=B, x=x, T=T, r=r)
            predicted_x = x + PECE2.__corrector(A=A, B=B, xk=predicted_x, T=T, r=r_T)
            x += PECE2.__corrector(A=A, B=B, xk=predicted_x, T=T, r=r_T)
            result.append(x)

            if current_print_after % print_after == 0:
                print(f"[{x.get_element_at(position=(0, 0))}, {x.get_element_at(position=(0, 1))}]")
            current_print_after += 1

        if f_real is not None:
            error: float = 0.0  # error at each time point
            for i in range(len(result)):
                for r in abs(result[i] - real_result[i]).get_elements()[0]:
                    error += r
            print(f"Error: {error / len(result)}")
            Drawer.draw_using(data=real_result, title=f"PE(CE)^2 - real solution", t_max=t_max, T=T)

        return result

    @staticmethod
    def __predictor(A: Matrica, B: Matrica | None, x: Matrica, T: float, r: Matrica | None) -> Matrica:
        """
        Predictor method used to calculate the next point using Euler's method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param x: current point
        :param T: integration step
        :param r: vector (optional)
        :return: calculated next point
        """
        a: Matrica = A * ~x
        return ~a * T if B is None else ~(a + B * ~r) * T

    @staticmethod
    def __corrector(A: Matrica, B: Matrica | None, xk: Matrica, T: float, r: Matrica | None) -> Matrica:
        """
        Method used to calculate the next point of the reversed Euler's method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param xk: next point
        :param T: integration step
        :param r: vector (optional)
        :return: calculated next point
        """
        a: Matrica = A * ~xk
        return ~a * T if B is None else ~(a + B * ~r) * T

    @staticmethod
    def __calculate_real_next_point(xk: Matrica, f_real: any, T: float, t: int) -> Matrica:
        """
        Method used to calculate the real next point of the PE(CE)^2 method.
        :param xk: next point
        :param f_real: function used to calculate real new points
        :param T: integration step
        :param t: current time moment
        :return: calculated next real point
        """
        return xk + f_real(x=xk, t=t) * T


class PECE:
    """
    Prediktorsko korektorski postupak (Euler + reversed Euler).
    """

    @staticmethod
    def calculate(
            A: Matrica,
            B: Matrica | None,
            x0: Matrica,
            f_real: any,
            T: float,
            t_max: int,
            r: Matrica | None = None,
            update_r: bool = False,
            print_after: int = 100
    ) -> list[Matrica]:
        """
        PECE method with Euler as predictor and trapeze as corrector.
        :param A: matrix of the function
        :param B: matrix of the function
        :param x0: starting point at t=0
        :param f_real: real function used to calculate real new points | None if it should not be used
        :param T: integration step
        :param t_max: time interval upper limit -> [0, t_max]
        :param r: matrix used to calculate new points
        :param update_r: determines whether r value should be updated or not
        :param print_after: after how many iterations to print current solution
        :return: list of calculated matrices
        """
        result: list[Matrica] = []
        real_result: list[Matrica] = []

        x: Matrica = Matrica(elements=x0.get_elements())
        current_print_after: int = 0

        for t in numpy.linspace(0, t_max, int(t_max / T)):
            if f_real is not None:
                real_result.append(PECE.__calculate_real_next_point(xk=x0, f_real=f_real, T=T, t=t))

            if update_r and r is not None:
                for i in range(len(r.get_elements()[0])):
                    r.set_element_at(position=(0, i), element=t)
            else:
                r = Matrica(elements=[[t, t]])

            r_T: Matrica | None = None
            if r is not None:
                r_T: Matrica = Matrica(elements=[[t + T for t in elements] for elements in r.get_elements()])

            # calculating next point using Euler and trapeze
            predicted_x: Matrica = x + PECE.__predictor(A=A, B=B, x=x, T=T, r=r)
            x += PECE.__corrector(A=A, B=B, x=x, xk=predicted_x, T=T, r=r, r_T=r_T)
            result.append(x)

            if current_print_after % print_after == 0:
                print(f"[{x.get_element_at(position=(0, 0))}, {x.get_element_at(position=(0, 1))}]")
            current_print_after += 1

        if f_real is not None:
            error: float = 0.0  # error at each time point
            for i in range(len(result)):
                for r in abs(result[i] - real_result[i]).get_elements()[0]:
                    error += r
            print(f"Error: {error / len(result)}")
            Drawer.draw_using(data=real_result, title=f"PECE - real solution", t_max=t_max, T=T)

        return result

    @staticmethod
    def __predictor(A: Matrica, B: Matrica | None, x: Matrica, T: float, r: Matrica | None) -> Matrica:
        """
        Predictor method used to calculate the next point using Euler's method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param x: current point
        :param T: integration step
        :param r: vector (optional)
        :return: calculated next point
        """
        a: Matrica = A * ~x
        return ~a * T if B is None else ~(a + B * ~r) * T

    @staticmethod
    def __corrector(
            A: Matrica,
            B: Matrica | None,
            x: Matrica,
            xk: Matrica,
            T: float,
            r: Matrica | None,
            r_T: Matrica | None
    ) -> Matrica:
        """
        Method used to calculate the next point of the trapeze method.
        :param A: matrix of the function
        :param B: matrix of the function (optional)
        :param x: current point
        :param xk: next point
        :param T: integration step
        :param r: current vector (optional)
        :param r_T: next vector (optional)
        :return: calculated next point
        """
        current_a: Matrica = A * ~x
        next_a: Matrica = A * ~xk
        return (~current_a + ~next_a) * T / 2 if B is None else ~(current_a + B * ~r + next_a + B * ~r_T) * T / 2

    @staticmethod
    def __calculate_real_next_point(xk: Matrica, f_real: any, T: float, t: int) -> Matrica:
        """
        Method used to calculate the real next point of the PECE method.
        :param xk: next point
        :param f_real: function used to calculate real new points
        :param T: integration step
        :param t: current time moment
        :return: calculated next real point
        """
        return xk + f_real(x=xk, t=t) * T
