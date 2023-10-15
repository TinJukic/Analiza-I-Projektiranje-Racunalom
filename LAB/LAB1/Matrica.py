"""
:author: Tin Jukic
"""
import sys


class Matrica:
    pass


class Matrica:
    """
    Matrix class with implemented all necessary functionality.
    """

    def __init__(self, elements: list[list[float]] | None = None):
        """
        Matrix constructor.
        :param elements: elements of the previous matrix
        """
        self.elements: list[list[float]] = \
            [] if elements.copy() is None else elements  # in Python, float is double precision

    def get_elements(self) -> list[list[float]]:
        """
        Gets matrix elements.
        :return: matrix elements
        """
        return self.elements

    def set_elements(self, elements: list[list[float]]) -> None:
        """
        Copies all elements into matrix.
        :param elements: to be copied
        :return: None
        """
        self.elements = elements.copy()

    def get_element_at(self, position: tuple[int, int]) -> float | None:
        """
        Gets element from desired position if it exists.
        :param position: from which to get the element
        :return: float if the element at desired position exists | None otherwise
        """
        i, j = position
        try:
            return self.elements[i][j]
        except IndexError as error:
            sys.stderr.write(f"Position out of range\n{error}\n")
            return None

    def set_element_at(self, position: tuple[int, int],  element: float) -> None:
        """
        Sets element at desired position.
        :param position: at which to put the element
        :param element: to be put into matrix
        :return: None
        """
        i, j = position
        self.elements[i][j] = element

    @staticmethod
    def load_matrix_from_file(file: str) -> Matrica | None:
        """
        Loads matrix from file into memory.
        :param file: from which to load matrix
        :return: new matrix if it could be created | None otherwise
        """
        try:
            elements: list[list[float]] = []
            with open(file, 'r', encoding='utf-8') as file_matrix:
                line, i = file_matrix.readline(), 0
                while line:
                    line, row = line.strip().split(), []
                    for element in line:
                        try:
                            row.append(float(element))
                        except ValueError:
                            sys.stderr.write(f"Value cannot be converted to float.\n")
                            return None
                    elements.append(row)
            return Matrica(elements=elements)

        except FileNotFoundError:
            sys.stderr.write(f"Provided file does not exist.")
            return None
