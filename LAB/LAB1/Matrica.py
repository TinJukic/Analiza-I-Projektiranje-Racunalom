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
        self.__elements: list[list[float | int]] = \
            [] if elements is None else elements.copy()  # in Python, float is double precision

    def get_elements(self) -> list[list[float | int]]:
        """
        Gets matrix elements.
        :return: matrix elements
        """
        return self.__elements

    def set_elements(self, elements: list[list[float | int]]) -> None:
        """
        Copies all elements into matrix.
        :param elements: to be copied
        :return: None
        """
        self.__elements = elements.copy()

    def get_element_at(self, position: tuple[int, int]) -> float | int | None:
        """
        Gets element from desired position if it exists.
        :param position: from which to get the element
        :return: *float* or *int* if the element at desired position exists | *None* otherwise
        """
        i, j = position
        try:
            return self.__elements[i][j]
        except IndexError as error:
            sys.stderr.write(f"Position out of range\n{error}\n")
            return None

    def set_element_at(self, position: tuple[int, int], element: float | int, N: int = 0) -> None:
        """
        Sets element at desired position.
        :param position: at which to put the element
        :param element: to be put into matrix
        :param N: number of columns that the new row-vector has
        :return: None
        """
        i, j = position
        try:
            self.__elements[i][j] = element
        except IndexError:
            # only used for row-vectors
            if len(self.__elements) == 0:
                self.__elements = [[0] for _ in range(N)]
            self.__elements[j][i] = element

    def get_matrix_dimension(self) -> int:
        """
        Calculates dimension of the matrix.
        :return: dimension of the matrix as *int*
        """
        return len(self.__elements)

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
        last_element_index: int = len(self.__elements[0]) - 1
        try:
            with open(file, 'w', encoding='utf-8') as file_matrix:
                for row in self.__elements:
                    for j, element in enumerate(row):
                        if j < last_element_index:
                            file_matrix.write(f"{str(element):10} ")
                        else:
                            file_matrix.write(f"{str(element):10}\n")
        except ValueError:
            sys.stderr.write(f"Value cannot be converted to string.\n")

    def print_matrix(self) -> None:
        """
        Prints matrix on screen.
        :return: None
        """
        last_element_index: int = len(self.__elements[0]) - 1
        for row in self.__elements:
            for j, element in enumerate(row):
                if j < last_element_index:
                    print(f"{element:10}", end=" ")
                else:
                    print(f"{element:10}")
        print()

    def __add__(self, other) -> Self:
        """
        Adds all elements from one matrix to another.
        :param other: matrix to be added
        :return: new *Matrica* object
        """
        elements: list[list[float]] = []
        rows, cols = len(self.__elements), len(self.__elements[0])
        if isinstance(other, float | int):
            for i in range(rows):
                col: list[float] = []
                for j in range(cols):
                    col.append(self.__elements[i][j] + other)
                elements.append(col)
        else:
            other: Matrica
            if rows != len(other.__elements) and cols != len(other.__elements[0]):
                sys.stderr.write(f"Dimensions of two matrices must be identical!")
                raise Exception

            for i in range(rows):
                col: list[float] = []
                for j in range(cols):
                    col.append(self.__elements[i][j] + other.__elements[i][j])
                elements.append(col)
        return Matrica(elements=elements)

    def __sub__(self, other) -> Self:
        """
        Subtracts all elements from one matrix to another.
        :param other: matrix to be subtracted
        :return: new *Matrix* object
        """
        elements: list[list[float]] = []
        rows, cols = len(self.__elements), len(self.__elements[0])
        if isinstance(other, float | int):
            for i in range(rows):
                col: list[float] = []
                for j in range(cols):
                    col.append(self.__elements[i][j] - other)
                elements.append(col)
        else:
            other: Matrica
            if rows != len(other.__elements) and cols != len(other.__elements[0]):
                sys.stderr.write(f"Dimensions of two matrices must be identical!")
                raise Exception

            for i in range(rows):
                col: list[float] = []
                for j in range(cols):
                    col.append(self.__elements[i][j] - other.__elements[i][j])
                elements.append(col)
        return Matrica(elements=elements)

    def __mul__(self, other) -> Self:
        """
        Multiplies all elements from one matrix to another.
        :param other: matrix to be multiplied
        :return: new *Matrica* object
        """
        elements: list[list[float]] = []
        rows, cols = len(self.__elements), len(self.__elements[0])
        if isinstance(other, float | int):
            for i in range(rows):
                col: list[float] = []
                for j in range(cols):
                    col.append(self.__elements[i][j] * other)
                elements.append(col)
        else:
            other: Matrica
            if rows != len(other.__elements[0]):
                sys.stderr.write(f"Dimensions of two matrices (rows and columns) must be identical "
                                 f"in order to multiply them!")
                raise Exception

            for i in range(rows):
                col: list[float] = []
                for j in range(cols):
                    summing: float = 0.0
                    for k in range(cols):
                        summing += self.__elements[i][k] * other.__elements[k][j]
                    col.append(summing)
                elements.append(col)
        return Matrica(elements=elements)

    def __invert__(self) -> Self:
        """
        Transposes the matrix.
        :return: new *Matrica* object
        """
        elements: list[list[float]] = []
        width, height = len(self.__elements), len((self.__elements[0]))
        for j in range(height):
            col: list[float] = []
            for i in range(width):
                col.append(self.__elements[i][j])
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

        width1, height1 = len(self.__elements), len(self.__elements[0])
        width2, height2 = len(other.__elements), len(other.__elements[0])
        if width1 != width2 or height1 != height2:
            return False

        for i in range(width1):
            for j in range(height1):
                if self.__elements[i][j] != other.__elements[i][j]:
                    return False

        return True

    def switch_rows(self, P: Matrica, row1: int, row2: int, num_of_transforms: int = 0) -> int | None:
        """
        Helper method for switching rows inside matrix.
        :param P: current identity matrix
        :param row1: first row, *int*
        :param row2: second row, *int*
        :param num_of_transforms: number of transformations that the matrix has gone through, 1 initially
        :return: number of transformations that the matrix has gone through | *None* if the transformation cannot be
                 executed
        """
        N: int = self.get_matrix_dimension()

        if row1 >= N or row2 >= N:
            return None

        tmp: list[float] = self.__elements[row1]
        self.__elements[row1] = self.__elements[row2]
        self.__elements[row2] = tmp

        tmp: list[int] = P.__elements[row1]
        P.__elements[row1] = P.__elements[row2]
        P.__elements[row2] = tmp

        return num_of_transforms + 1

    def to_row_vectors(self) -> list[Self]:
        """
        Separates the matrix into row-vectors. Matrix must be quadratic.
        :return: list of *Matrica* elements, which are row-vectors
        """
        N: int = self.get_matrix_dimension()
        row_vectors: list[Matrica] = [Matrica() for _ in range(N)]

        for i in range(N):
            for j in range(N):
                row_vectors[j].set_element_at((0, i), self.__elements[i][j], N)

        return row_vectors

    @staticmethod
    def row_vectors_to_matrix(row_vectors: list[Matrica]) -> Matrica:
        """
        Converts row-vectors into matrix.
        :param row_vectors: to be converted
        :return: new *quadratic Matrica* object with row_vectors as its elements
        """
        N: int = len(row_vectors)
        A: Matrica = Matrica([[0 for _ in range(N)] for _ in range(N)])

        for i in range(N):
            for j in range(N):
                A.set_element_at((i, j), row_vectors[j].get_element_at((i, 0)))

        return A

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
                b.__elements[0][j] -= self.__elements[j][i] * b.__elements[0][i]

        return b  # ne trebam return?

    def backward_substitution(self, b: Matrica) -> Self:
        """
        Performs backward substitution algorithm.\n
        Solves the equation Ux=y.\n
        Algorithm complexity: *O(n^2)*
        :return: vector as new *Matrica* object, which is the equation solution
        """
        N: int = self.get_matrix_dimension()

        for i in range(0, N - 1):
            b.__elements[0][i] /= self.__elements[i][i]
            for j in range(i + 1, N):
                b.__elements[0][j] -= self.__elements[j][i] * b.__elements[0][i]

        return b  # ne trebam return?

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
                    pivot: float = self.__elements[i][i]
                    if pivot == 0:
                        raise ZeroDivisionError
                    self.__elements[j][i] /= pivot

                    for k in range(i + 1, N):
                        self.__elements[j][k] -= self.__elements[j][i] * self.__elements[i][k]
        except ZeroDivisionError:
            sys.stderr.write(f"Pivot element cannot be zero!")
            return None

        return self

    def LUP_decomposition(self) -> tuple[Self, Self, int] | None:
        """
        Performs LUP-decomposition of the matrix.\n
        Algorithm complexity: *O(n^3)*
        :return: *tuple* with two *Matrica* objects and an *int*: LUP-decomposition of the matrix
                 and P matrix, and number of executed transformations | *None* if error occurred
        """
        try:
            N: int = self.get_matrix_dimension()
            P: Matrica = Matrica.identity_matrix(dimension=N)
            num_of_transforms: int = 0

            for i in range(0, N - 1):
                # pivot selection: i == row, i == j initially
                max_element: tuple[float, int] = self.__elements[i][i], i
                for j in range(i, N - 1):
                    if self.__elements[j][i] > max_element[0]:
                        max_element = self.__elements[j][i], j
                # references are sent here, return not required
                num_of_transforms = self.switch_rows(P, i, max_element[1], num_of_transforms)
                if self.__elements[i][i] == 0:
                    raise ZeroDivisionError

                # pivot selected at index (i, i)
                for j in range(i + 1, N):
                    self.__elements[j][i] /= self.__elements[i][j]

                    for k in range(i + 1, N):
                        self.__elements[j][k] -= self.__elements[j][i] * self.__elements[i][k]
        except ZeroDivisionError:
            sys.stderr.write(f"Pivot element cannot be zero!")
            return None

        return self, P, num_of_transforms

    def inversion(self) -> Self | None:
        """
        Makes the inverse of the quadratic matrix using LUP decomposition.\n
        One LUP-decomposition, n forward and backward substitutions.\n
        Algorithm complexity: *O(n^3)*
        :return: matrix inverse as *Matrica* | *None* if the inverse cannot be determined (if matrix is singular)
        """
        LUP, P, num_of_transformations = self.LUP_decomposition()
        N: int = self.get_matrix_dimension()

        if LUP is None:
            sys.stderr.write(f"Cannot calculate inverse of a singular matrix!")
            return None

        P: Matrica
        row_vectors_P: list[Matrica] = P.to_row_vectors()

        # N forward substitutions
        for i in range(N):
            self.forward_substitution(b=row_vectors_P[i])

        # N backward substitutions
        for i in range(N):
            self.backward_substitution(b=row_vectors_P[i])

        return Matrica.row_vectors_to_matrix(row_vectors=row_vectors_P)

    def determinant(self) -> int | None:
        """
        Calculates the determinant of the matrix.
        :return: determinant of the matrix | *None* if it cannot be calculated (singular matrix)
        """
        LUP, _, k = self.LUP_decomposition()
        N: int = self.get_matrix_dimension()

        if LUP is None:
            sys.stderr.write(f"Cannot calculate determinant of a singular matrix!")
            return None

        upper_determinant: int = 1
        for i in range(N):
            upper_determinant *= LUP.__elements[i][i]  # product of diagonal elements

        return pow(-1, k) * 1 * upper_determinant
