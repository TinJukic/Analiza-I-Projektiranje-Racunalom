"""
:author: Tin Jukić
"""
from __future__ import annotations
from typing import Self
import sys


class Matrica:
    """
    Matrix class with all necessary functionality implemented.
    Supports only quadratic matrices.
    """

    def __init__(self, elements: list[list[float | int]] | None = None):
        """
        Matrix constructor.
        :param elements: elements of the previous matrix
        """
        self.elements: list[list[float | int]] = \
            [] if elements.copy() is None else elements  # in Python, float is double precision

    def get_elements(self) -> list[list[float | int]]:
        """
        Gets matrix elements.
        :return: matrix elements
        """
        return self.elements

    def set_elements(self, elements: list[list[float | int]]) -> None:
        """
        Copies all elements into matrix.
        :param elements: to be copied
        :return: None
        """
        self.elements = elements.copy()

    def get_element_at(self, position: tuple[int, int]) -> float | int | None:
        """
        Gets element from desired position if it exists.
        :param position: from which to get the element
        :return: *float* or *int* if the element at desired position exists | *None* otherwise
        """
        i, j = position
        try:
            return self.elements[i][j]
        except IndexError as error:
            sys.stderr.write(f"Position out of range\n{error}\n")
            return None

    def set_element_at(self, position: tuple[int, int], element: float | int) -> None:
        """
        Sets element at desired position.
        :param position: at which to put the element
        :param element: to be put into matrix
        :return: None
        """
        i, j = position
        self.elements[i][j] = element

    def get_matrix_dimension(self) -> int:
        """
        Calculates dimension of the matrix.
        :return: dimension of the matrix as *int*
        """
        return len(self.elements)

    @staticmethod
    def load_matrix_from_file(file: str) -> Matrica | None:
        """
        Loads matrix from file into memory.
        :param file: from which to load matrix
        :return: new *Matrica* if it could be created | *None* otherwise
        """
        try:
            elements: list[list[float]] = []
            with open(file, 'r', encoding='utf-8') as file_matrix:
                line, i = file_matrix.readline(), 0
                while line:
                    line, row = line.split(), []
                    for element in line:
                        try:
                            row.append(float(element.strip()))
                        except ValueError:
                            sys.stderr.write(f"Value cannot be converted to float.\n")
                            return None
                    elements.append(row)
                    line = file_matrix.readline()
            return Matrica(elements=elements)

        except FileNotFoundError:
            sys.stderr.write(f"Provided file does not exist.")
            return None

    def save_matrix_to_file(self, file: str) -> None:
        """
        Saves matrix into desired file.
        :param file: in which to save the matrix
        :return: None
        """
        last_element_index: int = len(self.elements[0]) - 1
        try:
            with open(file, 'w', encoding='utf-8') as file_matrix:
                for row in self.elements:
                    for j, element in enumerate(row):
                        if j < last_element_index:
                            file_matrix.write(f"{str(element)} ")
                        else:
                            file_matrix.write(f"{str(element)}\n")
        except ValueError:
            sys.stderr.write(f"Value cannot be converted to string.\n")

    def print_matrix(self) -> None:
        """
        Prints matrix on screen.
        :return: None
        """
        last_element_index: int = len(self.elements[0]) - 1
        for row in self.elements:
            for j, element in enumerate(row):
                if j < last_element_index:
                    print(f"{element} ")
                else:
                    print(f"{element}\n")

    def __add__(self, other) -> Self:
        """
        Adds all elements from one matrix to another.
        :param other: matrix to be added
        :return: new *Matrica* object
        """
        rows, cols = len(self.elements), len(self.elements[0])
        if isinstance(other, float | int):
            for i in range(rows):
                for j in range(cols):
                    self.elements[i][j] += other
        else:
            for i in range(rows):
                for j in range(cols):
                    self.elements[i][j] += other.elements[i][j]
        return self

    def __sub__(self, other) -> Self:
        """
        Subtracts all elements from one matrix to another.
        :param other: matrix to be subtracted
        :return: new *Matrix* object
        """
        rows, cols = len(self.elements), len(self.elements[0])
        if isinstance(other, float | int):
            for i in range(rows):
                for j in range(cols):
                    self.elements[i][j] -= other
        else:
            for i in range(rows):
                for j in range(cols):
                    self.elements[i][j] -= other.elements[i][j]
        return self

    def __mul__(self, other) -> Self:
        """
        Multiplies all elements from one matrix to another.
        :param other: matrix to be multiplied
        :return: new *Matrica* object
        """
        rows, cols = len(self.elements), len(self.elements[0])
        if isinstance(other, float | int):
            for i in range(rows):
                for j in range(cols):
                    self.elements[i][j] *= other
        else:
            for i in range(rows):
                for j in range(cols):
                    self.elements[i][j] *= other.elements[i][j]
        return self

    def __invert__(self) -> Self:
        """
        Transposes the matrix.
        :return: new *Matrica* object
        """
        elements: list[list[float]] = []
        width, height = len(self.elements), len((self.elements[0]))
        for j in range(height):
            col: list[float] = []
            for i in range(width):
                col.append(self.elements[i][j])
            elements.append(col)
        return Matrica(elements=elements)

    def __eq__(self, other) -> bool:
        """
        Compares two matrices.
        :param other: matrix to be compared to
        :return: *True* if matrices are equal, *False* otherwise
        """
        if type(self) is not type(other):
            return False

        width1, height1 = len(self.elements), len(self.elements[0])
        width2, height2 = len(other.elements), len(other.elements[0])
        if width1 != width2 or height1 != height2:
            return False

        for i in range(width1):
            for j in range(height1):
                if self.elements[i][j] != other.elements[i][j]:
                    return False

        return True

    def __switch_rows(self, P: Matrica, row1: int, row2: int) -> None:
        """
        Helper method for switching rows inside matrix.
        :type P: current identity matrix
        :type row1: first row, *int*
        :type row2: second row, *int*
        :return: new *Matrica* object, which represents P matrix | *None* if the switch cannot be executed
        """
        N: int = self.get_matrix_dimension()

        if row1 >= N or row2 >= N:
            return None

        tmp: list[float] = self.elements[row1]
        self.elements[row1] = self.elements[row2]
        self.elements[row2] = tmp

        tmp: list[int] = P.elements[row1]
        P.elements[row1] = P.elements[row2]
        P.elements[row2] = tmp

    @staticmethod
    def identity_matrix(dimension: int) -> Matrica:
        """
        Creates new unit matrix with specified dimensions.
        :param dimension: of the matrix
        :return: new *Matrica* object
        """
        elements: list[list[float]] = [[1 if i == j else 0 for i in range(dimension)] for j in range(dimension)]
        return Matrica(elements=elements)

    def forward_substitution(self, b: Matrica) -> Self:
        """
        Performs forward substitution algorithm.\n
        Solves the equation Ly=Pb.\n
        Algorithm complexity: *O(n^2)*
        :param b: vector which multiplies the matrix
        :return: vector as new *Matrica* object, which is the equation solution
        """
        N: int = self.get_matrix_dimension()

        for i in range(0, N - 1):
            for j in range(i + 1, N):
                b.elements[0][j] -= self.elements[j][i] * b.elements[0][i]

        return b

    def backward_substitution(self, b: Matrica) -> Self:
        """
        Performs backward substitution algorithm.\n
        Solves the equation Ux=y.\n
        Algorithm complexity: *O(n^2)*
        :return: vector as new *Matrica* object, which is the equation solution
        """
        N: int = self.get_matrix_dimension()

        for i in range(0, N - 1):
            b.elements[0][i] /= self.elements[i][i]
            for j in range(i + 1, N):
                b.elements[0][j] -= self.elements[j][i] * b.elements[0][i]

        return b

    def LU_decomposition(self) -> Self | None:
        """
        Performs LU-decomposition of the matrix.\n
        Algorithm complexity: *O(n^3)*
        :return: *Matrica* which is LU-decomposition of the matrix | *None* if error occurred
        """
        N: int = self.get_matrix_dimension()

        try:
            for i in range(0, N - 1):
                for j in range(i + 1, N):
                    pivot: float = self.elements[i][i]
                    if pivot == 0:
                        raise ZeroDivisionError
                    self.elements[j][i] /= pivot

                    for k in range(i + 1, N):
                        self.elements[j][k] -= self.elements[j][i] * self.elements[i][k]
        except ZeroDivisionError:
            sys.stderr.write(f"Pivot element cannot be zero!")
            return None

        return self

    def LUP_decomposition(self) -> tuple[Self, Self] | None:
        """
        Performs LUP-decomposition of the matrix.\n
        Algorithm complexity: *O(n^3)*
        :return: *tuple* with two *Matrica* objects: LUP-decomposition of the matrix
                 and P matrix | *None* if error occurred
        """
        try:
            N: int = self.get_matrix_dimension()
            P: Matrica = Matrica.identity_matrix(dimension=N)

            for i in range(0, N - 1):
                # pivot selection: i == row, i == j initially
                max_element: tuple[float, int] = self.elements[i][i], i
                for j in range(i, N - 1):
                    if self.elements[j][i] > max_element[0]:
                        max_element = self.elements[j][i], j
                self.__switch_rows(P, i, max_element[1])  # references are sent here, return not required
                if self.elements[i][i] == 0:
                    raise ZeroDivisionError

                # pivot selected at index (i, i)
                for j in range(i + 1, N):
                    self.elements[j][i] /= self.elements[i][j]

                    for k in range(i + 1, N):
                        self.elements[j][k] -= self.elements[j][i] * self.elements[i][k]
        except ZeroDivisionError:
            sys.stderr.write(f"Pivot element cannot be zero!")
            return None

        return self, P

    def inversion(self):
        ...

    def determinant(self):
        ...
