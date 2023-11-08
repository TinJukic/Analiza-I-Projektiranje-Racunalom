"""
:author: Tin Jukić
"""
from __future__ import annotations

import math

from LAB2.Matrica import Matrica
from typing import Self
import sys


class Funkcije:
    """
    Implemented all functions for this exercise.
    """
    @staticmethod
    def f1(x: float):
        return math.pow(x - 3, 2)  # min = 3


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
            self.__interval = ZlatniRez.find_uni_modal_interval(x0=x0, h=1, f=f)
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
                fd, fc = fc,  f(c)
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
                selected_x: float = x.get_element_at(position=(0, i))
                selected_e: float = self.__e.get_element_at(position=(0, i))
                func = lambda l: f(selected_x + l * selected_e)

                interval: Matrica = ZlatniRez(x0=selected_x, f=func).golden_section(f=func)
                lam: float = (interval.get_element_at(position=(0, 0)) + interval.get_element_at(position=(0, 1))) / 2

                new_x = x.get_element_at(position=(0, i)) + lam * self.__e.get_element_at(position=(0, i))
                x.set_element_at(position=(0, i), element=new_x)

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
        self.__x0 = x0
        self.__e = e
        self.__delta_x = delta_x
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__sigma = sigma

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

    def calculate_nelder_meadu_simplex(self, f) -> Matrica:
        # starting points are calculated by moving starting point on each axis by delta_x value
        xs: list[Matrica] = [
            Matrica(elements=[
                [element + 1 if i == j else element for j, element in enumerate(self.__x0.get_elements())]
            ]) for i in range(self.__x0.get_matrix_dimension())
        ]  # vector X[i] -> starting simplex
