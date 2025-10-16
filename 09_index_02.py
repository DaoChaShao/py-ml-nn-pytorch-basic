#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 13:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   09_index_02.py
# @Desc     :   

from numpy import random as rand, ndarray
from random import randint
from torch import from_numpy

from utils.decorator import beautifier


@beautifier
def tensor_modifier():
    """ Modify Tensor and Numpy Array """
    arr: ndarray = rand.randint(1, 10, size=(randint(2, 5), randint(2, 5)))
    print(f"The random array is\n{arr}")

    t = from_numpy(arr)

    # Modify a value in arr and see if it affects tensor
    # - Method I
    # arr[0][0] = 0
    # - Method II
    arr[0, 0] = 0

    print(f"Modified Array：\n{arr}")
    print(f"Tensor：\n{t}")

    # Modify a value in tensor and see if it affects arr
    # - Method I
    # tensor[0][0] = -1
    # - Method II
    t[0, 0] = -1

    print(f"Modified Tensor：\n{t}")
    print(f"Array：\n{arr}")


def main() -> None:
    """ Main Function """
    tensor_modifier()


if __name__ == "__main__":
    main()
