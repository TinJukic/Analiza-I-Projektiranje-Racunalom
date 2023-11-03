"""
:author: Tin Jukić
"""
from __future__ import annotations
from typing import Self
import sys

from LAB2.Matrica import Matrica


class ZlatniRez:
    """
    Golden section class with all necessary functionality implemented.
    """
    def __init__(self, a: float, b: float, e: float):
        """
        *ZlatniRez* constructor.
        :param a: lower boundary of the uni-modal interval
        :param b: upper boundary of the uni-modal interval
        :param e: precision
        """
        self.__a = a
        self.__b = b
        self.__e = e

    @staticmethod
    def load_from_file(file: str) -> ZlatniRez | None:
        """
        Loads data for *ZlatniRez* class from file.
        :param file: file from which the data is loaded
        :return: new *ZlatniRez* if the file exists | *None* if the file does not exist
        """
        try:
            with open(file, 'r', encoding='utf-8') as file_golden_section:
                a, b, e = file_golden_section.readline().strip().split()
                return ZlatniRez(a=a, b=b, e=e)
        except FileNotFoundError:
            sys.stderr.write(f"Provided file does not exist!\n")
            return None
