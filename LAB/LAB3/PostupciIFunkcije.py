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
    def f1(x: Matrica):
        return 100 * pow(x.get_element_at(position=(0, 1)) - pow(x.get_element_at(position=(0, 0)), 2), 2) + \
               pow(1 - x.get_element_at(position=(0, 0)), 2)

    @staticmethod
    def f1_der1_x1(x: Matrica):
        return -400 * x.get_element_at(position=(0, 1)) * x.get_element_at(position=(0, 0)) + \
               400 * pow(x.get_element_at(position=(0, 0)), 3) - 2 + 2 * x.get_element_at(position=(0, 0))

    @staticmethod
    def f1_der1_x2(x: Matrica):
        return 200 * x.get_element_at(position=(0, 1)) - 200 * pow(x.get_element_at(position=(0, 0)), 2)

    @staticmethod
    def f1_der2_x1(x: Matrica):
        return -400 * x.get_element_at(position=(0, 1)) + 1200 * pow(x.get_element_at(position=(0, 0)), 2) + 2

    @staticmethod
    def f1_der2_x2(x: Matrica):
        return 200

    @staticmethod
    def f1_lambda(x: Matrica, v: Matrica):
        def f(lam: float) -> float:
            return 100 * pow(x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1)) -
                   pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)), 2), 2) + \
                   pow(1 - (x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0))), 2)
        return f

    @staticmethod
    def f1_1(x: Matrica):
        return 10 * (x.get_element_at(position=(0, 1)) - pow(x.get_element_at(position=(0, 0)), 2))

    @staticmethod
    def f1_1_der1_x1(x: Matrica):
        return -20 * x.get_element_at(position=(0, 0))

    @staticmethod
    def f1_1_der1_x2(x: Matrica):
        return 10

    @staticmethod
    def f1_1_der2_x1(x: Matrica):
        return -20

    @staticmethod
    def f1_1_der2_x2(x: Matrica):
        return 0

    @staticmethod
    def f1_1_lambda(x: Matrica, v: Matrica):
        def f(lam: float) -> float:
            return 10 * (x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1)) -
                         pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)), 2))
        return f

    @staticmethod
    def f1_2(x: Matrica):
        return 1 - x.get_element_at(position=(0, 0))

    @staticmethod
    def f1_2_der1_x1(x: Matrica):
        return -1

    @staticmethod
    def f1_2_der1_x2(x: Matrica):
        return 0

    @staticmethod
    def f1_2_der2_x1(x: Matrica):
        return 0

    @staticmethod
    def f1_2_der2_x2(x: Matrica):
        return 0

    @staticmethod
    def f1_2_lambda(x: Matrica, v: Matrica):
        def f(lam: float) -> float:
            return 1 - (x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)))
        return f

    @staticmethod
    def f2(x: Matrica):
        return pow(x.get_element_at(position=(0, 0)) - 4, 2) + 4 * pow(x.get_element_at(position=(0, 1)) - 2, 2)

    @staticmethod
    def f2_der1_x1(x: Matrica):
        return 2 * x.get_element_at(position=(0, 0)) - 8

    @staticmethod
    def f2_der1_x2(x: Matrica):
        return 8 * x.get_element_at(position=(0, 1)) - 16

    @staticmethod
    def f2_der2_x1(x: Matrica):
        return 2

    @staticmethod
    def f2_der2_x2(x: Matrica):
        return 8

    @staticmethod
    def f2_lambda(x: Matrica, v: Matrica):
        def f(lam: float) -> float:
            return pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)) - 4, 2) + \
                4 * pow(x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1)) - 2, 2)
        return f

    @staticmethod
    def f2_lambda_der(x: Matrica, v: Matrica):
        return ((4 * v.get_element_at(position=(0, 0)) + 8 * v.get_element_at(position=(0, 1)) -
                 x.get_element_at(position=(0, 0)) * v.get_element_at(position=(0, 0)) -
                 4 * x.get_element_at(position=(0, 1)) * v.get_element_at(position=(0, 1))) /
                (pow(v.get_element_at(position=(0, 0)), 2) + 4 * pow(v.get_element_at(position=(0, 1)), 2)))

    @staticmethod
    def f3(x: Matrica):
        return pow(x.get_element_at(position=(0, 0)) - 2, 2) + pow(x.get_element_at(position=(0, 1)) + 3, 2)

    @staticmethod
    def f3_der1_x1(x: Matrica):
        return 2 * (x.get_element_at(position=(0, 0)) - 2)

    @staticmethod
    def f3_der1_x2(x: Matrica):
        return 2 * (x.get_element_at(position=(0, 1)) + 3)

    @staticmethod
    def fr_der2_x1(x: Matrica):
        return 2

    @staticmethod
    def f3_der2_x2(x: Matrica):
        return 2

    @staticmethod
    def f3_lambda(x: Matrica, v: Matrica):
        def f(lam: float) -> float:
            return pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)) - 2, 2) + \
                   pow(x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1)) + 3, 2)
        return f

    @staticmethod
    def f4(x: Matrica):
        return (1 / 4) * pow(x.get_element_at(position=(0, 0)), 4) - pow(x.get_element_at(position=(0, 0)), 2) + \
               2 * x.get_element_at(position=(0, 0)) + pow(x.get_element_at(position=(0, 1)) - 1, 2)

    @staticmethod
    def f4_der1_x1(x: Matrica):
        return pow(x.get_element_at(position=(0, 0)), 3) - 2 * x.get_element_at(position=(0, 0)) + 2

    @staticmethod
    def f4_der1_x2(x: Matrica):
        return 2 * x.get_element_at(position=(0, 1)) - 2

    @staticmethod
    def f4_der2_x1(x: Matrica):
        return 3 * pow(x.get_element_at(position=(0, 0)), 2) - 2

    @staticmethod
    def f4_der2_x2(x: Matrica):
        return 2

    @staticmethod
    def f4_lambda(x: Matrica, v: Matrica):
        def f(lam: float) -> float:
            return (1 / 4) * pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)), 4) - \
                   pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)), 2) + \
                   2 * (x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0))) + \
                   pow(x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1)) - 1, 2)
        return f

    @staticmethod
    def f5_1(x: Matrica):
        return pow(x.get_element_at(position=(0, 0)), 2) + pow(x.get_element_at(position=(0, 1)), 2) - 1

    @staticmethod
    def f5_2(x: Matrica):
        return x.get_element_at(position=(0, 1)) - pow(x.get_element_at(position=(0, 0)), 2)

    @staticmethod
    def f5_1_der1_x1(x: Matrica):
        return 2 * x.get_element_at(position=(0, 0))

    @staticmethod
    def f5_1_der1_x2(x: Matrica):
        return 2 * x.get_element_at(position=(0, 1))

    @staticmethod
    def f5_2_der1_x1(x: Matrica):
        return -2 * x.get_element_at(position=(0, 0))

    @staticmethod
    def f5_2_der1_x2(x: Matrica):
        return 1

    @staticmethod
    def f5_lambda(x: Matrica, v: Matrica):
        def f(lam: float) -> float:
            return (pow(
                pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)), 2) +
                pow(x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1)), 2) - 1, 2) +
                    pow(
                        x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1)) -
                        pow(x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)), 2), 2)
                    )

        return f


class GradijentniSpust:
    """
    Gradient descent algorithm class with all necessary functionality implemented.
    """

    def __init__(
            self,
            x0: Matrica,
            f=None,
            f_der1_x1=None,
            f_der1_x2=None,
            f_lambda=None,
            f_lambda_der=None,
            e: float = 10e-6,
            use_golden_section: bool = False,
            max_num_of_iter: int = 10000
    ):
        """
        *GradijentniSpust* constructor.
        :param x0: starting point
        :param f: function for which the calculation is done
        :param f_der1_x1: first derivative of the function f in first element
        :param f_der1_x2: first derivative of the function f in second element
        :param f_lambda: lambda function for the function f
        :param f_lambda_der: minimum of the lambda f function in points x and v
        :param e: precision
        :param use_golden_section: determines which method the class will use for solving the problem
        :param max_num_of_iter: maximum number of iterations
        """
        self.__x0: Matrica = x0
        self.__f = f
        self.__f_der1_x1 = f_der1_x1
        self.__f_der1_x2 = f_der1_x2
        self.__f_lambda = f_lambda
        self.__f_lambda_der = f_lambda_der
        self.__e: float = e
        self.__use_golden_section: bool = use_golden_section
        self.__max_num_of_iter: int = max_num_of_iter

    def calculate(self) -> Matrica | None:
        """
        Calculates point using desired method passed through the constructor.
        :return: calculated point | *None* if the solution could not be found
        """
        return self.__calculate_with_golden_section() if self.__use_golden_section else self.__move_for_fixed_value()

    def __move_for_fixed_value(self) -> Matrica | None:
        """
        Calculates point by moving it for the whole offset.
        :return: calculated point | *None* if the solution could not be found
        """
        num_of_iter: int = 0
        num_of_fun_calls: int = 0
        num_of_grad_calls: int = 0
        number_of_non_improvements: int = 0
        x: Matrica = Matrica(elements=self.__x0.get_elements())

        while num_of_iter < self.__max_num_of_iter:
            try:
                num_of_iter += 1

                first_der_in_x1: float = self.__f_der1_x1(x=x)
                first_der_in_x2: float = self.__f_der1_x2(x=x)

                v: Matrica = Matrica(elements=[[-first_der_in_x1, -first_der_in_x2]])

                new_x: Matrica = Matrica(
                    elements=[[x.get_element_at(position=(0, 0)) + v.get_element_at(position=(0, 0)),
                               x.get_element_at(position=(0, 1)) + v.get_element_at(position=(0, 1))]]
                )
                num_of_grad_calls += 1

                result_der_x1: float = self.__f_der1_x1(x=new_x)
                result_der_x2: float = self.__f_der1_x2(x=new_x)
                num_of_grad_calls += 1

                if math.sqrt(pow(result_der_x1, 2) + pow(result_der_x2, 2)) < self.__e:
                    print(f"Number of function calls = {num_of_fun_calls}\n"
                          f"Number of gradient calls = {num_of_grad_calls}")
                    return new_x

                if number_of_non_improvements == 10:
                    print(
                        f"Number of function calls = {num_of_fun_calls}\n"
                        f"Number of gradient calls = {num_of_grad_calls}")
                    print(f"Problem diverges!")
                    return None
                elif self.__f(x=x) < self.__f(x=new_x):
                    number_of_non_improvements += 1
                else:
                    number_of_non_improvements = 0
                num_of_fun_calls += 2

                x = new_x
            except OverflowError:
                print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
                print(f"Problem diverges!")
                return None

        print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
        print(f"Problem diverges!")
        return None

    def __calculate_with_golden_section(self) -> Matrica | None:
        """
        Calculates point by calculating the optimal offset on the line using *ZlatniRez* class.
        :return: calculated point | *None* if the solution could not be found
        """
        num_of_iter: int = 0
        number_of_non_improvements: int = 0
        num_of_fun_calls: int = 0
        num_of_grad_calls: int = 0
        x: Matrica = Matrica(elements=self.__x0.get_elements())

        while num_of_iter < self.__max_num_of_iter:
            num_of_iter += 1

            first_der_in_x1: float = self.__f_der1_x1(x=x)
            first_der_in_x2: float = self.__f_der1_x2(x=x)

            v: Matrica = Matrica(elements=[[-first_der_in_x1, -first_der_in_x2]])

            goldSec: ZlatniRez = ZlatniRez(x0=0, f=self.__f_lambda(x=x, v=v))
            goldSecRes: tuple[Matrica, int] = goldSec.golden_section(f=self.__f_lambda(x=x, v=v))
            lam: float = goldSecRes[0].get_element_at(position=(0, 0))
            num_of_fun_calls += goldSecRes[1]

            new_x: Matrica = Matrica(
                elements=[[x.get_element_at(position=(0, 0)) + lam * v.get_element_at(position=(0, 0)),
                           x.get_element_at(position=(0, 1)) + lam * v.get_element_at(position=(0, 1))]]
            )
            num_of_grad_calls += 1

            result_der_x1: float = self.__f_der1_x1(x=new_x)
            result_der_x2: float = self.__f_der1_x2(x=new_x)
            num_of_grad_calls += 1

            if math.sqrt(pow(result_der_x1, 2) + pow(result_der_x2, 2)) < self.__e:
                print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
                return new_x

            if number_of_non_improvements == 10:
                print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
                print(f"Problem diverges!")
                return None
            elif self.__f(x=x) < self.__f(x=new_x):
                number_of_non_improvements += 1
            else:
                number_of_non_improvements = 0
            num_of_fun_calls += 2

            x = new_x

        print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
        print(f"Problem diverges!")
        return None


class NewtonRaphson:
    """
    Newton-Raphson algorithm class with all necessary functionality implemented.
    """

    def __init__(
            self,
            x0: Matrica,
            f=None,
            f_der1_x1=None,
            f_der1_x2=None,
            f_der2_x1=None,
            f_der2_x2=None,
            f_lambda=None,
            f_lambda_der=None,
            e: float = 10e-6,
            use_golden_section: bool = False,
            max_num_of_iter: int = 10000
    ):
        """
        *NewtonRaphson* constructor.
        :param x0: starting point
        :param f: function for which the calculation is done
        :param f_der1_x1: first derivative of the function f in first element
        :param f_der1_x2: first derivative of the function f in second element
        :param f_der2_x1: second derivative of the function f in first element
        :param f_der2_x2: second derivative of the function f in second element
        :param f_lambda: lambda function for the function f
        :param f_lambda_der: minimum of the lambda f function in points x and v
        :param e: precision
        :param use_golden_section: determines which method the class will use for solving the problem
        :param max_num_of_iter: maximum number of iterations
        """
        self.__x0: Matrica = x0
        self.__f = f
        self.__f_der1_x1 = f_der1_x1
        self.__f_der1_x2 = f_der1_x2
        self.__f_der2_x1 = f_der2_x1
        self.__f_der2_x2 = f_der2_x2
        self.__f_lambda = f_lambda
        self.__f_lambda_der = f_lambda_der
        self.__e: float = e
        self.__use_golden_section: bool = use_golden_section
        self.__max_num_of_iter: int = max_num_of_iter

    def calculate(self) -> Matrica | None:
        """
        Calculates point using desired method passed through the constructor.
        :return: calculated point | *None* if the solution could not be found
        """
        return self.__calculate_with_golden_section() if self.__use_golden_section else self.__move_for_fixed_value()

    def __move_for_fixed_value(self) -> Matrica | None:
        """
        Calculates point by moving it for the whole offset.
        :return: calculated point | *None* if the solution could not be found
        """
        num_of_iter: int = 0
        number_of_non_improvements: int = 0
        num_of_fun_calls: int = 0
        num_of_grad_calls: int = 0
        num_of_hess_calls: int = 0
        x: Matrica = Matrica(elements=self.__x0.get_elements())

        while num_of_iter < self.__max_num_of_iter:
            try:
                num_of_iter += 1

                first_der_in_x1: float = self.__f_der1_x1(x=x)
                first_der_in_x2: float = self.__f_der1_x2(x=x)
                second_der_in_x1: float = self.__f_der2_x1(x=x)
                second_der_in_x2: float = self.__f_der2_x2(x=x)

                hess: Matrica = Matrica(elements=[[-second_der_in_x1, 0], [0, -second_der_in_x2]])
                v: Matrica = Matrica(elements=[[first_der_in_x1, first_der_in_x2]])
                delta_x: Matrica = v * hess.inversion()
                num_of_grad_calls += 1
                num_of_hess_calls += 1

                new_x: Matrica = Matrica(
                    elements=[[x.get_element_at(position=(0, 0)) + delta_x.get_element_at(position=(0, 0)),
                               x.get_element_at(position=(0, 1)) + delta_x.get_element_at(position=(0, 1))]]
                )

                result_der_x1: float = self.__f_der1_x1(x=new_x)
                result_der_x2: float = self.__f_der1_x2(x=new_x)
                num_of_grad_calls += 1

                if math.sqrt(pow(result_der_x1, 2) + pow(result_der_x2, 2)) < self.__e:
                    print(
                        f"Number of function calls = {num_of_fun_calls}\n"
                        f"Number of gradient calls = {num_of_grad_calls}\n"
                        f"Number of Hess calls = {num_of_hess_calls}")
                    return new_x

                if number_of_non_improvements == 10:
                    print(
                        f"Number of function calls = {num_of_fun_calls}\n"
                        f"Number of gradient calls = {num_of_grad_calls}\n"
                        f"Number of Hess calls = {num_of_hess_calls}")
                    print(f"Problem diverges!")
                    return None
                elif self.__f(x=x) < self.__f(x=new_x):
                    number_of_non_improvements += 1
                else:
                    number_of_non_improvements = 0
                num_of_fun_calls += 2

                x = new_x
            except OverflowError:
                print(
                    f"Number of function calls = {num_of_fun_calls}\n"
                    f"Number of gradient calls = {num_of_grad_calls}\n"
                    f"Number of Hess calls = {num_of_hess_calls}")
                print(f"Problem diverges!")
                return None

        print(
            f"Number of function calls = {num_of_fun_calls}\n"
            f"Number of gradient calls = {num_of_grad_calls}\n"
            f"Number of Hess calls = {num_of_hess_calls}")
        print(f"Problem diverges!")
        return None

    def __calculate_with_golden_section(self) -> Matrica | None:
        """
        Calculates point by calculating the optimal offset on the line using *ZlatniRez* class.
        :return: calculated point | *None* if the solution could not be found
        """
        num_of_iter: int = 0
        number_of_non_improvements: int = 0
        num_of_fun_calls: int = 0
        num_of_grad_calls: int = 0
        num_of_hess_calls: int = 0
        x: Matrica = Matrica(elements=self.__x0.get_elements())

        while num_of_iter < self.__max_num_of_iter:
            num_of_iter += 1

            first_der_in_x1: float = self.__f_der1_x1(x=x)
            first_der_in_x2: float = self.__f_der1_x2(x=x)
            second_der_in_x1: float = self.__f_der2_x1(x=x)
            second_der_in_x2: float = self.__f_der2_x2(x=x)

            hess: Matrica = Matrica(elements=[[-second_der_in_x1, 0], [0, -second_der_in_x2]])
            v: Matrica = Matrica(elements=[[first_der_in_x1, first_der_in_x2]])
            delta_x: Matrica = v * hess.inversion()
            num_of_grad_calls += 1
            num_of_hess_calls += 1

            goldSec: ZlatniRez = ZlatniRez(x0=0, f=self.__f_lambda(x=x, v=delta_x))
            goldSecRes: tuple[Matrica, int] = goldSec.golden_section(f=self.__f_lambda(x=x, v=delta_x))
            lam: float = goldSecRes[0].get_element_at(position=(0, 0))
            num_of_fun_calls += goldSecRes[1]

            new_x: Matrica = Matrica(
                elements=[[x.get_element_at(position=(0, 0)) + lam * delta_x.get_element_at(position=(0, 0)),
                           x.get_element_at(position=(0, 1)) + lam * delta_x.get_element_at(position=(0, 1))]]
            )

            result_der_x1: float = self.__f_der1_x1(x=new_x)
            result_der_x2: float = self.__f_der1_x2(x=new_x)
            num_of_grad_calls += 1

            if math.sqrt(pow(result_der_x1, 2) + pow(result_der_x2, 2)) < self.__e:
                print(
                    f"Number of function calls = {num_of_fun_calls}\n"
                    f"Number of gradient calls = {num_of_grad_calls}\n"
                    f"Number of Hess calls = {num_of_hess_calls}")
                return new_x

            if number_of_non_improvements == 10:
                print(
                    f"Number of function calls = {num_of_fun_calls}\n"
                    f"Number of gradient calls = {num_of_grad_calls}\n"
                    f"Number of Hess calls = {num_of_hess_calls}")
                print(f"Problem diverges!")
                return None
            elif self.__f(x=x) < self.__f(x=new_x):
                number_of_non_improvements += 1
            else:
                number_of_non_improvements = 0
            num_of_fun_calls += 2

            x = new_x

        print(
            f"Number of function calls = {num_of_fun_calls}\n"
            f"Number of gradient calls = {num_of_grad_calls}\n"
            f"Number of Hess calls = {num_of_hess_calls}")
        print(f"Problem diverges!")
        return None


class GaussNewton:
    """
        Gauss-Newton algorithm class with all necessary functionality implemented.
    """

    def __init__(
            self,
            x0: Matrica,
            f1=None,
            f1_der1_x1=None,
            f1_der1_x2=None,
            f2=None,
            f2_der1_x1=None,
            f2_der1_x2=None,
            f_lambda=None,
            e: float = 10e-6,
            use_golden_section: bool = False,
            max_num_of_iter: int = 10000
    ):
        """
        *GaussNewton* constructor.
        :param x0: starting point
        :param f1: first function for which the calculation is done
        :param f1_der1_x1: first derivative of the first function f in first element
        :param f1_der1_x2: first derivative of the first function f in second element
        :param f2: second function for which the calculation is done
        :param f2_der1_x1: first derivative of the second function f in first element
        :param f2_der1_x2: first derivative of the second function f in second element
        :param f_lambda: lambda function for the function f
        :param e: precision
        :param use_golden_section: determines which method the class will use for solving the problem
        :param max_num_of_iter: maximum number of iterations
        """
        self.__x0: Matrica = x0
        self.__f1 = f1
        self.__f1_der1_x1 = f1_der1_x1
        self.__f1_der1_x2 = f1_der1_x2
        self.__f2 = f2
        self.__f2_der1_x1 = f2_der1_x1
        self.__f2_der1_x2 = f2_der1_x2
        self.__f_lambda = f_lambda
        self.__e: float = e
        self.__use_golden_section: bool = use_golden_section
        self.__max_num_of_iter: int = max_num_of_iter

    def calculate(self) -> Matrica | None:
        """
        Calculates point using desired method passed through the constructor.
        :return: calculated point | *None* if the solution could not be found
        """
        return self.__calculate_with_golden_section() if self.__use_golden_section else self.__move_for_fixed_value()

    def __move_for_fixed_value(self) -> Matrica | None:
        """
        Calculates point by moving it for the whole offset.
        :return: calculated point | *None* if the solution could not be found
        """
        num_of_iter: int = 0
        number_of_non_improvements: int = 0
        num_of_fun_calls: int = 0
        num_of_grad_calls: int = 0
        x: Matrica = Matrica(elements=self.__x0.get_elements())

        while num_of_iter < self.__max_num_of_iter:
            try:
                num_of_iter += 1

                f1_first_der_in_x1: float = self.__f1_der1_x1(x=x)
                f1_first_der_in_x2: float = self.__f1_der1_x2(x=x)
                f2_first_der_in_x1: float = self.__f2_der1_x1(x=x)
                f2_first_der_in_x2: float = self.__f2_der1_x2(x=x)

                j: Matrica = Matrica(elements=[[f1_first_der_in_x1, f1_first_der_in_x2],
                                               [f2_first_der_in_x1, f2_first_der_in_x2]])
                G: Matrica = Matrica(elements=[[self.__f1(x=x), self.__f2(x=x)]])
                num_of_fun_calls += 2
                num_of_grad_calls += 2

                a: Matrica = ~j * j
                g: Matrica = G * ~j
                g *= -1

                # solving the equation
                LUP = a.LUP_decomposition()
                A, P, n = LUP
                perm: Matrica = g * P
                y: Matrica = a.forward_substitution(b=~perm)
                delta_x: Matrica = a.backward_substitution(b=y)

                new_x: Matrica = Matrica(
                    elements=[[x.get_element_at(position=(0, 0)) + (~delta_x).get_element_at(position=(0, 0)),
                               x.get_element_at(position=(0, 1)) + (~delta_x).get_element_at(position=(0, 1))]]
                )

                if ((~delta_x).get_element_at(position=(0, 0)) < self.__e and
                        (~delta_x).get_element_at(position=(0, 1)) < self.__e):
                    print(f"Number of function calls = {num_of_fun_calls}\n"
                          f"Number of gradient calls = {num_of_grad_calls}")
                    return new_x

                if number_of_non_improvements == 10:
                    print(f"Number of function calls = {num_of_fun_calls}\n"
                          f"Number of gradient calls = {num_of_grad_calls}")
                    print(f"Problem diverges!")
                    return None
                elif self.__f1(x=x) < self.__f1(x=new_x) or self.__f2(x=x) < self.__f2(x=new_x):
                    number_of_non_improvements += 1
                else:
                    number_of_non_improvements = 0
                num_of_fun_calls += 2

                x = new_x
            except OverflowError:
                print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
                print(f"Problem diverges!")
                return None

        print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
        print(f"Problem diverges!")
        return None

    def __calculate_with_golden_section(self) -> Matrica | None:
        """
        Calculates point by calculating the optimal offset on the line using *ZlatniRez* class.
        :return: calculated point | *None* if the solution could not be found
        """
        num_of_iter: int = 0
        number_of_non_improvements: int = 0
        num_of_fun_calls: int = 0
        num_of_grad_calls: int = 0
        x: Matrica = Matrica(elements=self.__x0.get_elements())

        while num_of_iter < self.__max_num_of_iter:
            num_of_iter += 1

            f1_first_der_in_x1: float = self.__f1_der1_x1(x=x)
            f1_first_der_in_x2: float = self.__f1_der1_x2(x=x)
            f2_first_der_in_x1: float = self.__f2_der1_x1(x=x)
            f2_first_der_in_x2: float = self.__f2_der1_x2(x=x)

            j: Matrica = Matrica(elements=[[f1_first_der_in_x1, f1_first_der_in_x2],
                                           [f2_first_der_in_x1, f2_first_der_in_x2]])
            G: Matrica = Matrica(elements=[[self.__f1(x=x), self.__f2(x=x)]])
            num_of_fun_calls += 2
            num_of_grad_calls += 2

            a: Matrica = ~j * j
            g: Matrica = (~j) * (~G)
            g *= -1

            # solving the equation
            LUP = a.LUP_decomposition()
            A, P, n = LUP
            perm: Matrica = P * g
            y: Matrica = a.forward_substitution(b=perm)
            delta_x: Matrica = a.backward_substitution(b=y)

            goldSec: ZlatniRez = ZlatniRez(x0=0, f=self.__f_lambda(x=x, v=(~delta_x)))
            goldSecRes: tuple[Matrica, int] = goldSec.golden_section(f=self.__f_lambda(x=x, v=(~delta_x)))
            lam: float = goldSecRes[0].get_element_at(position=(0, 0))
            num_of_fun_calls += goldSecRes[1]

            new_x: Matrica = Matrica(
                elements=[[x.get_element_at(position=(0, 0)) + lam * (~delta_x).get_element_at(position=(0, 0)),
                           x.get_element_at(position=(0, 1)) + lam * (~delta_x).get_element_at(position=(0, 1))]]
            )

            if ((~delta_x).get_element_at(position=(0, 0)) < self.__e and
                    (~delta_x).get_element_at(position=(0, 1)) < self.__e):
                print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
                return new_x

            if number_of_non_improvements == 10:
                print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
                print(f"Problem diverges!")
                return None
            elif self.__f1(x=x) < self.__f1(x=new_x) or self.__f2(x=x) < self.__f2(x=new_x):
                number_of_non_improvements += 1
            else:
                number_of_non_improvements = 0
            num_of_fun_calls += 2

            x = new_x

        print(f"Number of function calls = {num_of_fun_calls}\nNumber of gradient calls = {num_of_grad_calls}")
        print(f"Problem diverges!")
        return None


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

    def golden_section(self, f, print_progress: bool = False) -> tuple[Matrica, int]:
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

        return Matrica([[a, b]]), num_of_iters

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
