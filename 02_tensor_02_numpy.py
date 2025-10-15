#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_02_numpy.py
# @Desc     :   

from numpy import ndarray, random as rand, set_printoptions as np_precision
from torch import from_numpy, set_printoptions as pt_precision
from random import randint

from utils.decorator import beautifier


@beautifier
def numpy_tensor():
    """ Create a tensor using numpy directly """
    arr: ndarray = rand.rand(randint(1, 3), randint(2, 5)) * 10
    print(f"The random array is {arr}")
    t = from_numpy(arr)
    print(t)
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


def main() -> None:
    """ Main Function """
    np_precision(precision=6)
    pt_precision(precision=6)
    numpy_tensor()


if __name__ == "__main__":
    main()
