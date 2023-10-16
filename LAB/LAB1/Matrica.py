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

    def set_element_at(self, position: tuple[int, int], element: float) -> None:
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

    def __add__(self, other) -> Matrica:
        """
        Adds all elements from one matrix to another.
        :param other: matrix to be added
        :return: new *Matrica* object
        """
        elements: list[list[float]] = []
        if isinstance(other, float | int):
            for row in self.elements:
                col: list[float] = []
                for element in row:
                    col.append(element + other)
                elements.append(col)
        else:
            for row1, row2 in zip(self.elements, other.elements):
                col: list[float] = []
                for element1, element2 in zip(row1, row2):
                    col.append(element1 + element2)
                elements.append(col)
        return Matrica(elements=elements)

    def __sub__(self, other) -> Matrica:
        """
        Subtracts all elements from one matrix to another.
        :param other: matrix to be subtracted
        :return: new *Matrix* object
        """
        elements: list[list[float]] = []
        if isinstance(other, float | int):
            for row in self.elements:
                col: list[float] = []
                for element in row:
                    col.append(element - other)
                elements.append(col)
        else:
            for row1, row2 in zip(self.elements, other.elements):
                col: list[float] = []
                for element1, element2 in zip(row1, row2):
                    col.append(element1 - element2)
                elements.append(col)
        return Matrica(elements=elements)

    def __mul__(self, other) -> Matrica:
        """
        Multiplies all elements from one matrix to another.
        :param other: matrix to be multiplied
        :return: new *Matrica* object
        """
        elements: list[list[float]] = []
        if isinstance(other, float | int):
            for row in self.elements:
                col: list[float] = []
                for element in row:
                    col.append(element * other)
                elements.append(col)
        else:
            for row1, row2 in zip(self.elements, other.elements):
                col: list[float] = []
                for element1, element2 in zip(row1, row2):
                    col.append(element1 * element2)
                elements.append(col)
        return Matrica(elements=elements)

    def __invert__(self) -> Matrica:
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
