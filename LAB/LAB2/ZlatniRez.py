"""
:author: Tin Jukić
"""
from __future__ import annotations

import math

from LAB2.Matrica import Matrica
from typing import Self
import sys


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
        if (x0 and a and b) is None:
            sys.stderr.write(f"You need to pass x0 or a and b to be able to use this class!\n")
            raise Exception

        self.__e: float = e
        self.__interval: Matrica
        self.__k: float

        if (x0 and h and f) is not None:
            self.__interval = ZlatniRez.find_uni_modal_interval(x0=x0, h=h, f=f)
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
        :return: new *ZlatniRez* if the file exists | *None* if the file does not exist
        """
        try:
            with open(file, 'r', encoding='utf-8') as file_golden_section:
                # ovdje je potrebno jos provjeriti koliko podataka imamo, 2 ili 3
                # ovisno o tome, mozda se treba pozvati i metoda find_uni_modal_interval
                a, b, e = file_golden_section.readline().strip().split()
                return ZlatniRez(a=a, b=b, e=e)
        except FileNotFoundError:
            sys.stderr.write(f"Provided file does not exist!\n")
            return None

    def golden_section(self, f) -> Matrica:
        """
        Calculates golden section from the interval of this class.
        :param f: function to be used in golden section calculation
        :return: calculated golden section
        """
        a, b = self.__interval.get_elements()[0]
        c, d = b - self.__k * (b - a), a + self.__k * (b - a)
        fc, fd = f(c), f(d)

        while (b - a) > self.__e:
            if fc < fd:
                b, d = d, c
                c = b - self.__k * (b - a)
                fd, fc = fc,  f(c)
            else:
                a, c = c, d
                d = a + self.__k * (b - a)
                fc, fd = fd, f(d)

        return Matrica([[a, b]])

    @staticmethod
    def find_uni_modal_interval(x0: float, h: float, f) -> Matrica:
        """
        Finds a uni-modal interval using starting point, shift and interval function.
        :param x0: starting point of a uni-modal interval
        :param h: shift of a uni-modal interval
        :param f: function of a uni-modal interval
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
                l, m, fm = m, r, fr
                step *= 2
                r = x0 + h * step
                fr = f(r)
        elif fm > fl:
            while fm > fl:
                r, m, fm = f, l, fl
                step *= 2
                l = x0 - h * step
                fl = f(l)

        return Matrica([[l, r]])
