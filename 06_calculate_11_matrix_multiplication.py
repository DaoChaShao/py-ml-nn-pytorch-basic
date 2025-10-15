#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:38
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_11_matrix_multiplication.py
# @Desc     :   

from torch import Tensor, tensor, matmul, mm

from utils.decorator import beautifier


@beautifier
def matrix_multiplication(a: Tensor, b: Tensor) -> Tensor:
    """ Matrix Multiplication of two tensors """
    print(f"Matrix A:\n{a}")
    print(f"Matrix B:\n{b}")

    c1 = matmul(a, b)
    print(f"Matrix Multiplication using matmul (matmul(A, B)):\n{c1}")

    c2 = mm(a, b)
    print(f"Matrix Multiplication using mm (A.mm(B)):\n{c2}")

    c3 = a @ b
    print(f"Matrix Multiplication using @ operator (A @ B):\n{c3}")

    return c1


def main() -> None:
    """ Main Function """
    a = tensor([[1, 2, 3], [4, 5, 6]])
    b = tensor([[1, 2], [3, 4], [5, 6]])
    # print(f"a:\n{a}")
    # print(f"b:\n{b}")

    matrix_multiplication(a, b)


if __name__ == "__main__":
    main()
