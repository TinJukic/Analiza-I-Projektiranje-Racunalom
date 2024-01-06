"""
:author: Tin Jukić
"""
from __future__ import annotations

import math
import random

from Matrica import Matrica


class Funkcije:
    """
    Implemented all functions for this exercise.
    """

    @staticmethod
    def f1(x: Matrica):
        # min: (1, 1), f_min: 0, start: (-1.9, 2)
        return 100 * pow(x.get_element_at(position=(0, 1)) - pow(x.get_element_at(position=(0, 0)), 2), 2) + \
            pow(1 - x.get_element_at(position=(0, 0)), 2)

    @staticmethod
    def f2(x: Matrica):
        # min: (4, 2), f_min: 0, start: (0.1, 0.3)
        return pow(x.get_element_at(position=(0, 0)) - 4, 2) + 4 * pow(x.get_element_at(position=(0, 1)) - 2, 2)

    @staticmethod
    def f3(x: Matrica):
        # min: (2, -3), f_min: 0, start: (0, 0, ..., 0)
        return pow(x.get_element_at(position=(0, 0)) - 2, 2) + pow(x.get_element_at(position=(0, 1)) + 3, 2)

    @staticmethod
    def f5(x: Matrica):
        # min: (3, 0), f_min: 0, start: (0, 0)s
        return pow(x.get_element_at(position=(0, 0)) - 3, 2) + pow(x.get_element_at(position=(0, 1)), 2)


class Ogranicenja:
    """
    Contains all boundaries needed for the *Box* algorithm.
    """

    @staticmethod
    def implicit_1_1(x: Matrica) -> bool:
        return x.get_element_at(position=(0, 1)) - x.get_element_at(position=(0, 0)) >= 0

    @staticmethod
    def implicit_1_2(x: Matrica) -> bool:
        return 2 - x.get_element_at(position=(0, 0)) >= 0

    @staticmethod
    def explicit_1_1(x: Matrica) -> bool:
        for element in x.get_elements():
            for e in element:
                if e < -100 or e > 100:
                    return False
        return True  # [-100, 100]

    @staticmethod
    def implicit_3_1(x: Matrica) -> bool:
        return 3 - x.get_element_at(position=(0, 0)) - x.get_element_at(position=(0, 1)) >= 0

    @staticmethod
    def implicit_3_2(x: Matrica) -> bool:
        return 3 + 1.5 * x.get_element_at(position=(0, 0)) - x.get_element_at(position=(0, 1)) >= 0

    @staticmethod
    def explicit_3_1(x: Matrica) -> bool:
        return x.get_element_at(position=(0, 1)) - 1 >= 0


class Box:
    """
    Box algorithm class with all necessary functionality implemented.
    """

    def __init__(
            self,
            x0: Matrica,
            implicit: list,
            explicit: list,
            explicit_values: list[int],
            e: float = 10e-6,
            alpha: float = 1.3,
            max_num_of_iters: int = 1000
    ):
        """
        *Box* constructor.
        :param x0: starting point
        :param implicit: implicit boundaries (functions)
        :param explicit: explicit boundaries (functions)
        :param explicit_values: explicit values (interval of numbers)
        :param e: precision
        :param alpha: parameter alpha
        :param max_num_of_iters: maximum number of iterations
        """
        self.__x0: Matrica = x0
        self.__implicit: list = implicit
        self.__explicit: list = explicit
        self.__explicit_values: list[int] = explicit_values
        self.__e: float = e
        self.__alpha: float = alpha
        self.__max_num_of_iters: int = max_num_of_iters

    def calculate(self, f, print_progress: bool = False) -> Matrica | None:
        """
        Runs Box algorithm on this class.
        :param f: function that needs to be minimised
        :param print_progress: tells the program whether the progress should be printed or not
        :return: found min of the function | *None* if min could not be found
        """
        xc: Matrica = Matrica(elements=self.__x0.get_elements())  # copy the starting point
        X: list[Matrica] = []

        # first check initial boundaries (xc == x0)
        if not self.__check_implicit_boundaries(x=xc) or not self.__check_explicit_boundaries(x=xc):
            return None

        n: int = len(self.__x0.get_elements()[0])

        # defining l and h parameters
        l: int
        h: int

        for j in range(2 * n):
            # create new empty matrix to store into X list
            x: Matrica = Matrica(elements=[[0 for _ in range(len(xc.get_elements()[0]))]])

            for i in range(n):
                # random values store in matrix x
                r = random.random()
                x.set_element_at(
                    position=(0, i),
                    element=self.__explicit_values[0] + r * (self.__explicit_values[1] - self.__explicit_values[0]),
                )

            X.append(x)

            limit: int = 100  # to prevent infinite loop
            while not self.__check_implicit_boundaries(x=X[j]) and limit > 0:
                X[j] = Box.__move_to_centroid(x=X[j], xc=xc)
                limit -= 1

            # calculate new centroid using all points
            xc = Box.__find_centroid(xs=X, j=j)

        num_of_iters: int = 0  # to prevent infinite loop
        diverges: int = 0  # to prevent divergence
        prev_result: float = 0.0  # result of the previous iteration

        while num_of_iters < self.__max_num_of_iters + 1:
            num_of_iters += 1

            # find min and max element
            l = Box.__argmin(f=f, xs=X)
            h = Box.__argmax(f=f, xs=X)
            second_h: int = Box.__second_argmax(f=f, xs=X, h=h)

            # find current min of the function
            current_min: float = f(x=X[l])

            # calculate new centroid without xh point
            xc = Box.__find_centroid(xs=X, h=h)

            # reflexion point
            xr: Matrica = Box.__reflexion(alpha=self.__alpha, xc=xc, xh=X[h])

            # move boundary to explicit boundaries
            for i in range(n):
                if xr.get_element_at(position=(0, i)) < self.__explicit_values[0]:
                    xr.set_element_at(position=(0, i), element=self.__explicit_values[0])
                elif xr.get_element_at(position=(0, i)) > self.__explicit_values[1]:
                    xr.set_element_at(position=(0, i), element=self.__explicit_values[1])

            # check implicit boundaries for xr
            limit: int = 100  # to prevent infinite loop
            while not self.__check_implicit_boundaries(x=xr) and limit > 0:
                xr = Box.__move_to_centroid(x=xr, xc=xc)
                limit -= 1

            # if xr is still the worse, update it once more
            if f(x=xr) > f(x=X[second_h]):
                xr = Box.__move_to_centroid(x=xr, xc=xc)

            X[h] = xr

            if diverges == 100:
                print(f"Problem diverges!")
                print(f"Expected min: (1, 1), f_min = {f(x=Matrica(elements=[[1, 1]]))}")
                return (X[l] + X[h]) / 2

            result: float = 0.0
            for xi in X:
                element = pow(f(xi) - f(xc), 2)  # should always have only one value - scalar
                if isinstance(element, float):
                    result += element
                else:
                    element: Matrica
                    result += element.get_element_at(position=(0, 0))
            result = math.sqrt(result / 2)

            print(f"X[l] = {X[l].get_elements()}, result = {result}")

            if result < self.__e:
                if print_progress:
                    print(f"Number of iterations for Nelder-Meadu algorithm is {num_of_iters}.")
                return (X[l] + X[h]) / 2

            # didn't return - diverges?
            l = Box.__argmin(f=f, xs=X)  # calculate new min point
            if f(x=X[l]) >= current_min or abs(prev_result - result) < self.__e:
                diverges += 1
            else:
                diverges = 0

            # set result as new previous result for next iteration
            prev_result = result

    def __check_implicit_boundaries(self, x: Matrica) -> bool:
        """
        Checks whether the implicit boundaries are satisfied by point x.
        :param x: point for which the implicit boundaries are checked
        :return: *True* if the boundaries are satisfied, *False* otherwise
        """
        for implicit in self.__implicit:
            # check all implicit boundaries
            if not implicit(x=x):
                return False
        return True

    def __check_explicit_boundaries(self, x: Matrica) -> bool:
        """
        Checks whether the explicit boundaries are satisfied by point x.
        :param x: point for which the explicit boundaries are checked
        :return: *True* if the explicit boundaries are satisfied, *False* otherwise
        """
        for explicit in self.__explicit:
            # check all explicit boundaries
            if not explicit(x=x):
                return False
        return True

    @staticmethod
    def __move_to_centroid(x: Matrica, xc: Matrica) -> Matrica:
        """
        Moves point x to the centroid xc.
        :param x: point to be moved
        :param xc: centroid point
        :return: moved point to centroid point
        """
        return (x + xc) / 2

    @staticmethod
    def __argmin(f, xs: list[Matrica]) -> int:
        """
        Finds the argmin of the function.
        :param f: desired function
        :param xs: values for which the min is calculated
        :return: argmin
        """
        x_function_call: dict[int:Matrica] = {i: f(x) for i, x in enumerate(xs)}

        argmin: int = 0
        for i in range(len(x_function_call) - 1):
            if x_function_call[i] < x_function_call[argmin]:
                argmin = i
            for j in range(i + 1, len(x_function_call)):
                if x_function_call[j] < x_function_call[argmin]:
                    argmin = j
        return argmin

    @staticmethod
    def __argmax(f, xs: list[Matrica]) -> int:
        """
        Finds the argmax of the function.
        :param f: desired function
        :param xs: values for which the max is calculated
        :return: argmax
        """
        x_function_call: dict[int:Matrica] = {i: f(x) for i, x in enumerate(xs)}

        argmax: int = 0
        for i in range(len(x_function_call) - 1):
            if x_function_call[i] > x_function_call[argmax]:
                argmax = i
            for j in range(i + 1, len(x_function_call)):
                if x_function_call[j] > x_function_call[argmax]:
                    argmax = j
        return argmax

    @staticmethod
    def __second_argmax(f, xs: list[Matrica], h: int) -> int:
        """
        Finds the argmax of the function.
        :param f: desired function
        :param xs: values for which the max is calculated
        :param h: argmax
        :return: second argmax
        """
        x_function_call: dict[int:Matrica] = {i: f(x) for i, x in enumerate(xs)}

        argmax: int = 0
        for i in range(len(x_function_call) - 1):
            if i == h:
                continue
            if x_function_call[argmax] < x_function_call[i] != x_function_call[h]:
                argmax = i
            for j in range(i + 1, len(x_function_call)):
                if j == h:
                    continue
                if x_function_call[argmax] < x_function_call[j] != x_function_call[h]:
                    argmax = j
        return argmax if argmax != h else (argmax + 1) % len(xs)

    @staticmethod
    def __find_centroid(xs: list[Matrica], j: int | None = None, h: int | None = None) -> Matrica:
        """
        Finds the centroid.
        :param xs: list of vectors
        :param j: upper limit | None if the limit is n
        :param h: argmax value | None if it should not be used
        :return: found centroid
        """
        xc: Matrica = Matrica(elements=[[0 for _ in range(len(xs[0].get_elements()[0]))]])
        n: int = j if j is not None else len(xs)

        for i in range(n):
            if h is None or i != h:
                xc += xs[i]
        return xc / n if h is not None else xc / 2

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
