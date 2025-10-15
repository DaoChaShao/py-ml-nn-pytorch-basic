#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:16
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_04_formal.py
# @Desc     :   

from numpy import ndarray, random as rand
from random import randint
from torch import tensor

from utils.decorator import beautifier


@beautifier
def formal_tensor():
    """ Create a tensor from a numpy array """
    # Get numpy array
    arr: ndarray = rand.randint(1, 10, size=(randint(2, 5), randint(2, 5)))
    # Transform a numpy into a tensor
    t = tensor(
        data=arr,
        dtype=None,  # Data type, default (None) is inferred from data
        device=None,  # Device to place the tensor on, default (None) is CPU
        requires_grad=False,  # If True, will record operations on the tensor for automatic differentiation
        pin_memory=False,  # If True, the returned tensor would be allocated in page-locked memory
    )
    print(t)
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


def main() -> None:
    """ Main Function """
    formal_tensor()


if __name__ == "__main__":
    main()
