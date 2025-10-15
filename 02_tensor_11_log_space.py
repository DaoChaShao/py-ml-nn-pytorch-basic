#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:24
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_11_log_space.py
# @Desc     :   

from random import randint
from torch import logspace, strided

from utils.decorator import beautifier


@beautifier
def log_space_tensor():
    """ Create a tensor using logarithmic space """
    start: int = randint(1, 3)  # Exponent start
    end: int = randint(3, 5)  # Exponent end
    steps: int = randint(1, 11)  # Number of steps
    base: int = 10  # Base of the logarithm
    t = logspace(
        start,
        end,
        steps,
        base=base,
        dtype=None,
        layout=strided,
        device=None,
        requires_grad=False,
    )
    print(f"The logarithmic space tensor from {base}^{start} to {base}^{end} with {steps} steps is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


def main() -> None:
    """ Main Function """
    log_space_tensor()


if __name__ == "__main__":
    main()
