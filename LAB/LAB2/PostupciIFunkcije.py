"""
:author: Tin Jukić
"""
from __future__ import annotations

import math

from Matrica import Matrica
import sys


class Funkcije:
    """
    Implemented all functions for this exercise.
    """

    @staticmethod
    def uni_modal_test(x: float):
        return pow(x, 2) - 2

    @staticmethod
    def golden_section_test(x: float):
        return pow(x - 4, 2)

    @staticmethod
    def f(x: float):
        return pow(x - 3, 2)  # min = 3

    @staticmethod
    def f1(x: Matrica):
        # min = (1, 1), f_min = 0
        return ((100 * pow(x.get_element_at(position=(0, 1)) - pow(x.get_element_at(position=(0, 0)), 2), 2))
                + pow(1 - x.get_element_at(position=(0, 0)), 2))

    @staticmethod
    def f2(x: Matrica):
        # min = (4, 2), f_min = 0
        return pow((x.get_element_at(position=(0, 0)) - 4), 2) + 4 * pow((x.get_element_at(position=(0, 1)) - 2), 2)

    @staticmethod
    def f3(x: Matrica):
        # min = (0, 1, 2, 3, ..., n), f_min = 0
        summ = 0.0
        for i, xi in enumerate(x.get_elements()[0]):
            summ += pow(xi - i, 2)
        return summ

    @staticmethod
    def f4(x: Matrica):
        # min = (0, 0), f_min = 0
        x.get_element_at(position=(0, 0))
        return (abs(
            (x.get_element_at(position=(0, 0)) - x.get_element_at(position=(0, 1))
             ) * (x.get_element_at(position=(0, 0)) + x.get_element_at(position=(0, 1)))) +
                math.sqrt(pow(x.get_element_at(position=(0, 0)), 2) + pow(x.get_element_at(position=(0, 1)), 2)))

    @staticmethod
    def f6(x: Matrica):
        # min = (0, 0, 0, ..., 0), 1...n, f_min = 0
        summm = 0.0
        for xi in x.get_elements()[0]:
            summm += pow(xi, 2)
        return 0.5 + (pow(math.sin(math.sqrt(summm)), 2) - 0.5) / pow(1 + 0.001 * summm, 2)


class ZlatniRez:
    """
    Golden section class with all necessary functionality implemented.
    """

    def __init__(self, x0: float = None, h: float = None, f=None, a: float = None, b: float = None, e: float = 10e-6):
        """
        *ZlatniRez* constructor.
        :param x0: starting point of the uni-modal interval
        :param h: shift of an interval
        :param f: function of an interval
        :param e: precision
        :param a: lower boundary of the uni-modal interval
        :param b: upper boundary of the uni-modal interval
        :param e: precision
        :raise Exception: if neither x0 nor a and b are set
        """
        while x0 is None and a is None and b is None:
            print(f"None of the arguments were given. Please provide them.")
            user_selection: str
            user_selection = input(f"x0=")
            if user_selection != '\n':
                x0 = float(user_selection)

            user_selection = input(f"a=")
            if user_selection != '\n':
                a = float(user_selection)
                b = float(input(f"b="))

            user_selection = input(f"e=")
            if user_selection != '\n':
                e = float(user_selection)

        self.__e: float = e
        self.__interval: Matrica
        self.__k: float

        if x0 is not None:
            self.__interval = ZlatniRez.find_uni_modal_interval(x0=x0, h=0.1, f=f)
            self.__k = 0.5 * (math.sqrt(5) - 1)
        else:
            self.__interval = Matrica([[a, b]])

    @classmethod
    def create_uni_modal_interval(cls, x0: float, h: float, f, e: float):
        """
        Creates *ZlatniRez* class with starting point of an interval.
        :param x0: starting point of an interval
        :param h: shift of an interval
        :param f: function of an interval
        :param e: precision
        :return: new *ZlatniRez* object
        """
        ...

    @staticmethod
    def load_from_file(file: str) -> ZlatniRez | None:
        """
        Loads data for *ZlatniRez* class from file.
        :param file: file from which the data is loaded
        :return: new *ZlatniRez* if the file exists | *None* if the file does not exist or sent values are incorrect
        """
        try:
            with open(file, 'r', encoding='utf-8') as file_golden_section:
                lines: list[str] = file_golden_section.readline().strip().split()
                if len(lines) == 2:
                    # x0, e
                    return ZlatniRez(x0=float(lines[0]), h=1, e=float(lines[1]))
                elif len(lines) == 3:
                    # a, b, e
                    return ZlatniRez(a=float(lines[0]), b=float(lines[1]), e=float(lines[2]))
                else:
                    sys.stderr.write(f"You gave the program too many elements as input! Input should be either 'x0' and"
                                     f"'e' or points 'a', 'b' and precision 'e'.\n")
                    return None
        except FileNotFoundError:
            sys.stderr.write(f"Provided file does not exist!\n")
            return None

    def golden_section(self, f, print_progress: bool = False) -> Matrica:
        """
        Calculates golden section from the interval of this class.
        :param f: function to be used in golden section calculation
        :param print_progress: tells the program whether the progress should be printed or not
        :return: calculated golden section
        """
        num_of_iters: int = 0

        a, b = self.__interval.get_elements()[0]
        c, d = b - self.__k * (b - a), a + self.__k * (b - a)
        fc, fd = f(c), f(d)

        while (b - a) > self.__e:
            num_of_iters += 1

            # if print_progress:
            #     print(f"a = {a}, b = {b}, c = {c}, d = {d}, fc = {fc}, fd = {fd}")

            if fc < fd:
                b, d = d, c
                c = b - self.__k * (b - a)
                fd, fc = fc, f(c)
            else:
                a, c = c, d
                d = a + self.__k * (b - a)
                fc, fd = fd, f(d)

        if print_progress:
            print(f"Number of iterations for golden_section is {num_of_iters}.")

        return Matrica([[a, b]])

    @staticmethod
    def find_uni_modal_interval(x0: float, h: float, f, print_progress: bool = False) -> Matrica:
        """
        Finds a uni-modal interval using starting point, shift and interval function.
        :param x0: starting point of a uni-modal interval
        :param h: shift of a uni-modal interval
        :param f: function of a uni-modal interval
        :param print_progress: tells the program whether the progress should be printed or not
        :return: uni-modal interval as a new *Matrica* object
        """
        l: float = x0 - h
        r: float = x0 + h
        m: float = x0
        fm: float
        fl: float
        fr: float
        step: int = 1

        fm, fl, fr = f(x0), f(l), f(r)

        if fm > fr:
            while fm > fr:
                if print_progress:
                    print(f"l = {l}, r = {r}, m = {m}, fm = {fm}, fl = {fl}, fr = {fr}, step = {step}")

                l, m, fm = m, r, fr
                step *= 2
                r = x0 + h * step
                fr = f(r)
        elif fm > fl:
            while fm > fl:
                if print_progress:
                    print(f"l = {l}, r = {r}, m = {m}, fm = {fm}, fl = {fl}, fr = {fr}, step = {step}")

                r, m, fm = m, l, fl
                step *= 2
                l = x0 - h * step
                fl = f(l)

        return Matrica([[l, r]])


class PretrazivanjePoKoordinatnimOsima:
    """
    Coordinate axis search algorithm with all necessary functionality implemented.
    """

    def __init__(self, x0: Matrica, n: int, e: float = 10e-6):
        """
        *PretrazivanjePoKoordinatnimOsima* constructor.
        :param x0: starting point
        :param n: number of dimensions
        :param e: precision vector
        """
        self.__x0: Matrica = x0
        self.__n: int = n
        self.__e: Matrica = Matrica([[e for _ in range(n)]])

    @staticmethod
    def load_from_file(file: str) -> PretrazivanjePoKoordinatnimOsima | None:
        """
        Loads data for *PretrazivanjePoKoordinatnimOsima* class from file.
        :param file: file from which the data is loaded
        :return: new *PretrazivanjePoKoordinatnimOsima* if the file exists | *None* if the file does not exist
                 or sent values are incorrect
        """
        try:
            with open(file, 'r', encoding='utf-8') as file_coordinate_axis_search:
                lines: list[str] = file_coordinate_axis_search.readline().strip().split()
                if len(lines) == 3:
                    # x0, n, e
                    # return PretrazivanjePoKoordinatnimOsima(x0=float(lines[0]), n=int(lines[1]), e=float(lines[2]))
                    ...
                else:
                    sys.stderr.write(f"You gave the program too many elements as input! Input should be either 'e' and"
                                     f"'x0' or points 'a' and 'b'.\n")
                    return None
        except FileNotFoundError:
            sys.stderr.write(f"Provided file does not exist!\n")
            return None

    def coordinate_search(self, f, e: float = 10e-6, print_progress: bool = False) -> Matrica:
        """
        Runs coordinate axis search algorithm on this class.
        :param f: function that needs to be minimised
        :param e: precision for this search
        :param print_progress: tells the program whether the progress should be printed or not
        :return: found coordinate
        """
        num_of_iters: int = 0

        x: Matrica = Matrica(elements=self.__x0.get_elements())

        while True:
            num_of_iters += 1

            xs: Matrica = Matrica(elements=x.get_elements())

            for i in range(self.__n):
                # minimization in one dimension
                func = lambda l: f(Matrica(elements=[[
                    x.get_element_at(position=(0, j)) + l * self.__e.get_element_at(position=(0, j))
                    if i != j else l  # i == j => e == 1
                    for j in range(len(x.get_elements()[0]))
                ]]))

                selected_x: float = x.get_element_at(position=(0, i))

                interval: Matrica = ZlatniRez(x0=selected_x, f=func).golden_section(f=func)
                lam: float = (interval.get_element_at(position=(0, 0)) + interval.get_element_at(position=(0, 1))) / 2

                x.set_element_at(position=(0, i), element=lam)

            if abs(x - xs) < self.__e:
                break

        if print_progress:
            print(f"Number of iterations for coordinate_search is {num_of_iters}.")

        return x


class NelderMeaduSimplex:
    """
    Nelder-Meadu simplex algorithm with all necessary functionality implemented.
    """

    def __init__(self, x0: Matrica, e: float = 10e-6, delta_x: float = 1.0, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 2.0, sigma: float = 0.5):
        """
        *NelderMeaduSimplex* constructor.
        :param x0: starting point
        :param e: precision
        :param delta_x: shift
        :param alpha: parameter alpha
        :param beta: parameter beta
        :param gamma: parameter gamma
        :param sigma: parameter sigma
        """
        self.__x0: Matrica = x0
        self.__e: float = e
        self.__delta_x: float = delta_x
        self.__alpha: float = alpha
        self.__beta: float = beta
        self.__gamma: float = gamma
        self.__sigma: float = sigma

    @staticmethod
    def load_from_file(file: str) -> NelderMeaduSimplex | None:
        """
        Loads data for *PretrazivanjePoKoordinatnimOsima* class from file.
        :param file: file from which the data is loaded
        :return: new *NelderMeaduSimplex* if the file exists | *None* if the file does not exist
                 or sent values are incorrect
        """
        try:
            with open(file, 'r', encoding='utf-8') as file_coordinate_axis_search:
                lines: list[str] = file_coordinate_axis_search.readline().strip().split()
                if len(lines) > 1:
                    # x0, n, e
                    # return NelderMeaduSimplex(x0=float(lines[0]))
                    ...
                else:
                    sys.stderr.write(f"You gave the program too many elements as input! Input should be either 'e' and"
                                     f"'x0' or points 'a' and 'b'.\n")
                    return None
        except FileNotFoundError:
            sys.stderr.write(f"Provided file does not exist!\n")
            return None

    def calculate_nelder_meadu_simplex(self, f, print_progress: bool = False) -> Matrica:
        """
        Runs Nelder-Meadu algorithm on this class.
        :param f: function that needs to be minimised
        :param print_progress: tells the program whether the progress should be printed or not
        :return: found min of the function
        """
        num_of_iters: int = 0

        xs: list[Matrica] = self.__calculate_starting_points()  # vector X[i] -> starting simplex

        while True:
            num_of_iters += 1

            l: int = self.__argmin(f=f, xs=xs)
            h: int = self.__argmax(f=f, xs=xs)
            # s: int = self.__argmin(f=f, xs=xs, h=h)

            k: float = (1 + math.sqrt(5)) / 2
            xc: Matrica = NelderMeaduSimplex.__find_centroid(x0=self.__x0, xs=xs, h=h)
            xr: Matrica = NelderMeaduSimplex.__reflexion(alpha=self.__alpha, xc=xc, xh=xs[h])

            if f(xr) < f(xs[l]):
                xe: Matrica = NelderMeaduSimplex.__expansion(gamma=self.__gamma, xc=xc, xr=xr)
                xs[h] = xe if f(xe) < f(xs[l]) else xr
            else:
                all_xr_smaller: bool = True
                for i in range(len(xs)):
                    if i != h:
                        if f(xr) >= f(xs[i]):
                            all_xr_smaller = False
                            break

                if all_xr_smaller:
                    xs[h] = xr
                else:
                    xk: Matrica = NelderMeaduSimplex.__contraction(alpha=self.__alpha, beta=self.__beta, xc=xc, xr=xr) \
                        if f(xr) < f(xs[h]) \
                        else NelderMeaduSimplex.__contraction(alpha=self.__alpha, beta=self.__beta, xc=xc, xh=xs[h])

                    if f(xk) < f(xs[h]):
                        xs[h] = xk
                    else:
                        NelderMeaduSimplex.__move_points_to_l(xs=xs, l=l)

            result: float = 0.0
            for xi in xs:
                element = pow(f(xi) - f(xc), 2)  # should always have only one value - scalar
                if isinstance(element, float):
                    result += element
                else:
                    element: Matrica
                    result += element.get_element_at(position=(0, 0))
            result = math.sqrt(result / len(self.__x0.get_elements()[0]))

            print(f"xc = {xc.get_elements()}, result = {result}")

            if result <= self.__e:
                if print_progress:
                    print(f"Number of iterations for Nelder-Meadu algorithm is {num_of_iters}.")
                return (xs[l] + xs[h]) / 2  # (a + b) / 2

    def __calculate_starting_points(self) -> list[Matrica]:
        """
        Calculates starting points of Nelder-Meadu simplex algorithm.
        :return: starting points
        """
        # starting points are calculated by moving starting point on each axis by delta_x value
        xs: list[Matrica] = [self.__x0]

        for x in self.__x0:
            x: list[float | int]
            for i in range(len(x)):
                xs.append(Matrica(elements=[[element + 1 if i == j else element for j, element in enumerate(x)]]))
        return xs

    def __argmin(self, f, xs: list[Matrica]) -> int:
        """
        Finds the argmin of the function.
        :param f: desired function
        :param xs: values for which the min is calculated
        :return: argmin
        """
        x_function_call: dict[int:Matrica] = {i: f(x) for i, x in enumerate(xs)}
        # x_function_call: dict[int:Matrica] = {i: lambda l: f(x + l * self.__e) for i, x in enumerate(xs)}

        argmin: int = 0
        for i in range(len(x_function_call) - 1):
            if x_function_call[i] < x_function_call[argmin]:
                argmin = i
            for j in range(i + 1, len(x_function_call)):
                if x_function_call[j] < x_function_call[i]:
                    argmin = j
        return argmin

    def __argmax(self, f, xs: list[Matrica]) -> int:
        """
        Finds the argmax of the function.
        :param f: desired function
        :param xs: values for which the max is calculated
        :return: argmax
        """
        x_function_call: dict[int:Matrica] = {i: f(x) for i, x in enumerate(xs)}
        # x_function_call: dict[int:Matrica] = {i: lambda l: f(x + l * self.__e) for i, x in enumerate(xs)}

        argmax: int = 0
        for i in range(len(x_function_call) - 1):
            if x_function_call[i] > x_function_call[argmax]:
                argmax = i
            for j in range(i + 1, len(x_function_call)):
                if x_function_call[j] > x_function_call[i]:
                    argmax = j
        return argmax

    @staticmethod
    def __find_centroid(x0: Matrica, xs: list[Matrica], h: int) -> Matrica:
        """
        Finds the centroid.
        :param x0: starting point
        :param xs: list of vectors
        :param h: argmax value
        :return: found centroid
        """
        xc: Matrica = Matrica(elements=[[0 for _ in range(len(xs[0].get_elements()[0]))]])
        n: int = len(xs)

        for i in range(n):
            if i != h:
                xc += xs[i]
        return xc / len(x0.get_elements()[0])

    @staticmethod
    def __reflexion(alpha: float, xc: Matrica, xh: Matrica) -> Matrica:
        """
        Performs reflexion.
        :param alpha: coefficient alpha
        :param xc: centroid
        :param xh: max value for the argument h (argmax)
        :return: reflexion point
        """
        return xc * (1 + alpha) - xh * alpha

    @staticmethod
    def __expansion(gamma: float, xc: Matrica, xr: Matrica) -> Matrica:
        """
        Performs expansion.
        :param gamma: coefficient gamma
        :param xc: centroid
        :param xr: reflexion point
        :return: expansion point
        """
        return xc * (1 - gamma) - xr * gamma

    @staticmethod
    def __contraction(alpha: float, beta: float, xc: Matrica, xr: Matrica = None, xh: Matrica = None) -> Matrica:
        """
        Performs contraction.
        :param beta: coefficient beta
        :param xc: centroid
        :param xr: reflexion point
        :param xh: max value for the argument h (argmax)
        :return: contraction point
        """
        return xc * (1 - beta) - xr * alpha if xr is not None else xc * (1 - beta) - xh * alpha

    @staticmethod
    def __move_points_to_l(xs: list[Matrica], l: int) -> None:
        """
        Moves all point to l.
        :param xs: all points in this iteration
        :param l: argmin value
        :return: None
        """
        for i in range(len(xs)):
            if i != l:
                xs[i] = (xs[i] + xs[l]) / 2  # (pointer, no need for return)


class HookeJeeves:
    def __init__(self, x0: Matrica, delta_x: float = 0.5, e: float = 10e-6):
        """
        *HookeJeeves* constructor.
        :param x0: starting point
        :param delta_x: delta x for every example
        :param e: precision
        """
        self.__x0: Matrica = x0
        self.__dx: Matrica = Matrica(elements=[[delta_x for _ in range(len(x0.get_elements()[0]))]])
        self.__e: Matrica = Matrica(elements=[[e for _ in range(len(x0.get_elements()[0]))]])

    def calculate_hooke_jeeves(self, f, print_progress: bool = False) -> Matrica:
        """
        Runs Hooke-Jeeves algorithm on this class.
        :param f: function that needs to be minimised
        :param print_progress: tells the program whether the progress should be printed or not
        :return: found min of the function
        """
        num_of_iters: int = 0

        xp: Matrica = Matrica(elements=self.__x0.get_elements())
        xb: Matrica = Matrica(elements=self.__x0.get_elements())

        while self.__dx > self.__e:
            num_of_iters += 1

            xn: Matrica = self.__search_procedure(xp=xp, f=f)

            if f(xn) < f(xb):
                xp = xn * 2 - xb
                xb = xn
            else:
                for i, dx in enumerate(self.__dx.get_elements()[0]):
                    self.__dx.set_element_at(position=(0, i), element=dx / 2)
                xp = xb

            fb, fp, fn = f(xb), f(xp), f(xn)

            if isinstance(fb, float):
                print(f"xb = {xb.get_elements()}, f(xb) = {fb} ; "
                      f"xp = {xp.get_elements()}, f(xp) = {fp} ; "
                      f"xn = {xn.get_elements()}, f(xn) = {fn}")
            else:
                print(f"xb = {xb.get_elements()}, f(xb) = {fb.get_elements()} ; "
                      f"xp = {xp.get_elements()}, f(xp) = {fp.get_elements()} ; "
                      f"xn = {xn.get_elements()}, f(xn) = {fn.get_elements()}")

        if print_progress:
            print(f"Number of iterations for Hooke-Jeeves algorithm is {num_of_iters}.")

        return xb

    def __search_procedure(self, xp: Matrica, f) -> Matrica:
        """
        Searches for solution.
        :param xp: starting point for the search procedure
        :param f: function that needs to be minimised
        :return: found solution
        """
        x: Matrica = Matrica(elements=xp.get_elements())

        for i in range(len(xp.get_elements()[0])):
            p: Matrica = f(x)

            x.set_element_at(
                position=(0, i),
                element=x.get_element_at(position=(0, i)) + self.__dx.get_element_at(position=(0, i))
            )

            n: Matrica = f(x)

            if n > p:
                x.set_element_at(
                    position=(0, i),
                    element=x.get_element_at(position=(0, i)) - 2 * self.__dx.get_element_at(position=(0, i))
                )

                n = f(x)

                if n > p:
                    x.set_element_at(
                        position=(0, i),
                        element=x.get_element_at(position=(0, i)) + self.__dx.get_element_at(position=(0, i))
                    )

        return x
