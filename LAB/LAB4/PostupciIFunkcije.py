"""
:author: Tin Jukić
"""
from __future__ import annotations

import math
import random

from Matrica import Matrica
import sys


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
    def f4(x: Matrica):
        # min: (3, 0), f_min: 0, start: (0, 0)
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
    def implicit_2_1(x: Matrica) -> float:
        return x.get_element_at(position=(0, 1)) - x.get_element_at(position=(0, 0))

    @staticmethod
    def implicit_2_2(x: Matrica) -> float:
        return 2 - x.get_element_at(position=(0, 0))

    @staticmethod
    def implicit_3_1(x: Matrica) -> float:
        return 3 - x.get_element_at(position=(0, 0)) - x.get_element_at(position=(0, 1))

    @staticmethod
    def implicit_3_2(x: Matrica) -> float:
        return 3 + 1.5 * x.get_element_at(position=(0, 0)) - x.get_element_at(position=(0, 1))

    @staticmethod
    def explicit_3_1(x: Matrica) -> float:
        return x.get_element_at(position=(0, 1)) - 1


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
            r = random.random()

            for i in range(n):
                # random values store in matrix x
                x.set_element_at(
                    position=(0, i),
                    element=self.__explicit_values[0] + r * (self.__explicit_values[1] - self.__explicit_values[0]),
                )

            X.append(x)

            limit: int = 100  # to prevent infinite loop
            while ((not self.__check_explicit_boundaries(x=X[j]) or not self.__check_implicit_boundaries(x=X[j]))
                   and limit > 0):
                X[j] = Box.__move_to_centroid(x=X[j], xc=xc)
                limit -= 1

            # calculate new centroid using all points
            xc = Box.__find_centroid(xs=X, j=j+1, h=Box.__argmax(f=f, xs=X))

        num_of_iters: int = 0  # to prevent infinite loop
        diverges: int = 0  # to prevent divergence
        prev_result: float = 0.0  # result of the previous iteration

        while num_of_iters < self.__max_num_of_iters + 1:
            num_of_iters += 1

            # find min and max element
            l = Box.__argmin(f=f, xs=X)
            h = Box.__argmax(f=f, xs=X)
            second_h: int = Box.__second_argmax(f=f, xs=X, h=h)

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
            if abs(prev_result - result) < self.__e:
                diverges += 1
            else:
                # does not diverge -> reset
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


class TransformacijaBezOgranicenja:
    """
    Transformation without boundaries with mixed way.
    """

    def __init__(self, max_num_of_iters: int = 1000, e: float = 10e-6):
        """
        *TransformacijaBezOgranicenja* constructor.
        :param max_num_of_iters: maximum number of iterations
        :param e: precision
        """
        self.__max_num_of_iters: int = max_num_of_iters
        self.__e: float = e

    def transform(self, x: Matrica, f: any, g: list, h: list | None, r: float = 1) -> Matrica:
        """
        Transforms g and h functions,
        iteratively uses Nelder-Meadu simplex and Hooke-Jeeves algorithm to find the optimal point.
        :param x: point for which the transformation is done
        :param f: function to be minimised
        :param g: list of implicit boundaries
        :param h: list of explicit boundaries | None if the boundaries should be ignored
        :param r: parameter used for calculating the transformation
        :return: transformed point
        """
        prev_solution: Matrica = x  # result of the previous iteration

        # check boundaries for the initial point x
        if not (TransformacijaBezOgranicenja.__check_implicit_boundaries(x=x, g=g) and
                TransformacijaBezOgranicenja.__check_explicit_boundaries(x=x, h=h)):
            # find new starting point using Nelder-Meadu simplex or Hooke-Jeeves algorithm -> using: Hooke-Jeeves
            hooke_jeeves: HookeJeeves = HookeJeeves(x0=x)
            prev_solution = hooke_jeeves.calculate_hooke_jeeves(
                f=lambda inner: TransformacijaBezOgranicenja.__calculate_using_inner_point(x=inner, g=g, r=r)
            )

        num_of_iters: int = 0  # to prevent infinite loop
        diverges: int = 0  # to prevent divergence

        while num_of_iters < self.__max_num_of_iters + 1:
            num_of_iters += 1

            hooke_jeeves: HookeJeeves = HookeJeeves(x0=prev_solution)
            new_solution: Matrica = hooke_jeeves.calculate_hooke_jeeves(
                f=lambda inner: TransformacijaBezOgranicenja.__u(x=inner, f=f, g=g, h=h, r=r)
            )

            # printing results to check the progress
            # print(f"Iteration: {num_of_iters}")
            # print(f"Previous solution: ")
            # prev_solution.print_matrix()
            # print(f"New solution: ")
            # new_solution.print_matrix()

            if diverges == 100:
                print(f"Problem diverges!")
                return prev_solution

            # calculate distances between prev_solution and new_solution points
            if TransformacijaBezOgranicenja.__calculate_distance(a=new_solution, b=prev_solution) < self.__e:
                print(f"Solution found at iteration {num_of_iters}")
                return new_solution

            # solution not found -> diverges?
            if (TransformacijaBezOgranicenja.__u(x=new_solution, f=f, g=g, h=h, r=r) >=
                    TransformacijaBezOgranicenja.__u(x=prev_solution, f=f, g=g, h=h, r=r)):
                diverges += 1
            else:
                # does not diverge -> reset
                diverges = 0

            prev_solution = new_solution
            r /= 10

        print(f"Maximum number of iterations reached!")
        return prev_solution

    @staticmethod
    def __u(x: Matrica, f: any, g: list, h: list | None, r: float) -> float:
        """
        Calculates the U function for the transformation process.
        :param x: point for which the transformation is done
        :param f: function to be minimised
        :param g: list of implicit boundaries
        :param h: list of explicit boundaries | None if h should be ignored
        :param r: parameter used for calculating the transformation
        :return: calculated result
        """
        implicit_sum: float = 0.0
        for implicit in g:
            # boundary not satisfied?
            sol: float = implicit(x=x)
            if sol < 0:
                return math.inf  # return infinite if boundary is not satisfied

            try:
                implicit_sum += math.log(implicit(x=x))
            except ValueError:
                return math.inf
        implicit_sum *= r

        explicit_sum: float = 0.0
        if h is not None:
            for explicit in h:
                explicit_sum += math.pow(explicit(x=x), 2)
            explicit_sum /= r

        return f(x=x) - implicit_sum + explicit_sum

    @staticmethod
    def __calculate_using_inner_point(x: Matrica, g: list, r: float) -> float:
        """
        Calculates the result using inner point function for the transformation process.
        :param x: point for which the transformation is done
        :param g: list of implicit boundaries
        :param r: parameter used for calculating the transformation
        :return: calculated result
        """
        # transform using inner point
        implicit_sum: float = 0.0
        for implicit in g:
            try:
                implicit_sum += math.log(implicit(x=x))
            except ValueError:
                return math.inf
        return -(implicit_sum / r)

    @staticmethod
    def __calculate_distance(a: Matrica, b: Matrica) -> float:
        """
        Calculates the distance between two points.
        :param a: first point
        :param b: second point
        :return: calculated distance
        """
        distance: float = 0.0
        for i in range(len(a.get_elements())):
            distance += math.pow(
                a.get_element_at(position=(0, i)) - b.get_element_at(position=(0, i)), 2
            )
        return math.sqrt(distance)

    @staticmethod
    def __check_implicit_boundaries(x: Matrica, g: list) -> bool:
        """
        Checks whether the implicit boundaries are satisfied by point x.
        :param x: point for which the implicit boundaries are checked
        :param g: list of implicit boundaries
        :return: *True* if the boundaries are satisfied, *False* otherwise
        """
        for implicit in g:
            # check all implicit boundaries
            if implicit(x=x) < 0:
                return False
        return True

    @staticmethod
    def __check_explicit_boundaries(x: Matrica, h: list | None) -> bool:
        """
        Checks whether the explicit boundaries are satisfied by point x.
        :param x: point for which the explicit boundaries are checked
        :param h: list of explicit boundaries | None if h should be ignored
        :return: *True* if the explicit boundaries are satisfied, *False* otherwise
        """
        # h should be ignored
        if h is None:
            return True

        for explicit in h:
            # check all explicit boundaries
            if explicit(x=x) != 0:
                return False
        return True


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
                all_xr_larger: bool = True
                for i in range(len(xs)):
                    if i != h:
                        if f(xr) <= f(xs[i]):
                            all_xr_larger = False
                            break

                if all_xr_larger:
                    if f(xr) < f(xs[h]):
                        xs[h] = xr

                    xk: Matrica = NelderMeaduSimplex.__contraction(alpha=self.__alpha, beta=self.__beta, xc=xc,
                                                                   xh=xs[h])

                    if f(xk) < f(xs[h]):
                        xs[h] = xk
                    else:
                        NelderMeaduSimplex.__move_points_to_l(sigma=self.__sigma, xs=xs, l=l)
                else:
                    xs[h] = xr

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

        argmin: int = 0
        for i in range(len(x_function_call) - 1):
            if x_function_call[i] < x_function_call[argmin]:
                argmin = i
            for j in range(i + 1, len(x_function_call)):
                if x_function_call[j] < x_function_call[argmin]:
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

        argmax: int = 0
        for i in range(len(x_function_call) - 1):
            if x_function_call[i] > x_function_call[argmax]:
                argmax = i
            for j in range(i + 1, len(x_function_call)):
                if x_function_call[j] > x_function_call[argmax]:
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
        return xc * (1 + gamma) - xr * gamma

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
        return xc * (1 - beta) + xh * beta

    @staticmethod
    def __move_points_to_l(sigma: float, xs: list[Matrica], l: int) -> None:
        """
        Moves all point to l.
        :param sigma: coefficient sigma
        :param xs: all points in this iteration
        :param l: argmin value
        :return: None
        """
        for i in range(len(xs)):
            if i != l:
                xs[i] = (xs[i] + xs[l]) * sigma  # (pointer, no need for return)


class HookeJeeves:
    def __init__(self, x0: Matrica, delta_x: float = 0.5, e: float = 10e-6, max_num_of_iters: int = 1000):
        """
        *HookeJeeves* constructor.
        :param x0: starting point
        :param delta_x: delta x for every example
        :param e: precision
        :param max_num_of_iters: maximum number of iterations
        """
        self.__x0: Matrica = x0
        self.__dx: Matrica = Matrica(elements=[[delta_x for _ in range(len(x0.get_elements()[0]))]])
        self.__e: Matrica = Matrica(elements=[[e for _ in range(len(x0.get_elements()[0]))]])
        self.__max_num_of_iters: int = max_num_of_iters

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

        while self.__dx > self.__e and num_of_iters < self.__max_num_of_iters + 1:
            num_of_iters += 1

            xn: Matrica = self.__search_procedure(xp=xp, f=f)

            if f(xn) < f(xb):
                xp = xn * 2 - xb
                xb = xn
            else:
                for i, dx in enumerate(self.__dx.get_elements()[0]):
                    self.__dx.set_element_at(position=(0, i), element=dx / 2)
                xp = xb

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
