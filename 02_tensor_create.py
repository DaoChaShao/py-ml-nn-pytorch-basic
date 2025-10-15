#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 11:24
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_create.py
# @Desc     :   

from numpy import ndarray, random as rand
from random import randint
from torch import Tensor, tensor, from_numpy, arange, linspace, logspace

from utils.decorator import beautifier


@beautifier
def scalar_tensor() -> Tensor:
    """ Create a tensor from a number """
    num = randint(1, 11)
    t = tensor(num)
    print(f"The random number is {num}.")
    print(f"The tensor is {t}.")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


@beautifier
def numpy_tensor():
    """ Create a tensor using numpy directly """
    arr: ndarray = rand.randint(1, 10, size=(randint(1, 1), randint(2, 5)))
    t = from_numpy(arr)
    print(t)
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


@beautifier
def empty_tensor():
    """ Create a tensor focusing torch size """
    dimensions: int = randint(1, 6)
    rows: int = randint(1, 6)
    cols: int = randint(1, 6)
    t = Tensor(dimensions, rows, cols)
    print(f"The empty tensor with size ({dimensions}, {rows}, {cols}) is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")


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


@beautifier
def range_tensor():
    """ Create a tensor using a range of numbers """
    start: int = randint(1, 11)
    end: int = randint(11, 21)
    step: int = randint(1, 3)
    t = arange(start, end, step)
    print(f"The range tensor from {start} to {end} with step {step} is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


@beautifier
def linear_space_tensor():
    """ Create a tensor using linear space """
    start: int = randint(11, 21)
    end: int = randint(21, 31)
    steps: int = randint(1, 11)
    t = linspace(start, end, steps)
    print(f"The linear space tensor from {start} to {end} with {steps} steps is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


@beautifier
def log_space_tensor():
    """ Create a tensor using logarithmic space """
    start: int = randint(1, 3)  # Exponent start
    end: int = randint(3, 5)  # Exponent end
    steps: int = randint(1, 11)  # Number of steps
    base: int = 10  # Base of the logarithm
    t = logspace(start, end, steps, base=base)
    print(f"The logarithmic space tensor from {base}^{start} to {base}^{end} with {steps} steps is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


def main() -> None:
    """ Main Function """
    scalar_tensor()

    numpy_tensor()

    empty_tensor()

    formal_tensor()

    range_tensor()

    linear_space_tensor()

    log_space_tensor()


if __name__ == "__main__":
    main()
