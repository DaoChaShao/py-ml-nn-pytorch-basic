#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 13:03
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   11_reshape_01_reshape.py
# @Desc     :   

from torch import Tensor, randperm, reshape

from utils.decorator import beautifier
from utils.highlighter import green, red, yellow


@beautifier
def reshape_tensor_auto():
    """ This function is used to reshape a tensor with automatic dimension calculation. """
    tensor_input: Tensor = randperm(8)
    print(f"tensor_input: \n{green(tensor_input)}")
    print(f"tensor_input length: {green(len(tensor_input))}")

    t = reshape(
        input=tensor_input,
        shape=[-1, 2, 2],  # -1 means automatically calculate the dimension
    )
    print(f"tensor: \n{red(t)}")
    print(f"tensor_input length: {red(len(t))}")

    print(f"tensor_input address: {yellow(id(tensor_input.data))}")
    print(f"tensor address: {yellow(id(t.data))}")


@beautifier
def reshape_tensor():
    """ This function is used to reshape a tensor. """
    tensor_input: Tensor = randperm(8)  # create a random tensor of size 10
    print(f"tensor_input: \n{green(tensor_input)}")
    print(f"tensor_input length: {green(len(tensor_input))}")

    t = reshape(
        input=tensor_input,
        shape=[2, 4],
        # shape=[4, 2],
    )
    print(f"tensor: \n{red(t)}")
    print(f"tensor_input length: {red(len(t))}")


def main() -> None:
    """ Main Function """
    reshape_tensor_auto()
    reshape_tensor()


if __name__ == "__main__":
    main()
