#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 14:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   16_split.py
# @Desc     :   

from random import randint
from torch import rand, split

from utils.decorator import beautifier
from utils.highlighter import green, red, yellow


@beautifier
def split_tensor_int_row():
    """ Split Tensor in equal parts by Row """
    tensor_full = rand(3, 7)
    print(f"tensor_full: \n{green(tensor_full)}")
    print(f"tensor_full shape: {green(tensor_full.shape)}")

    random_section = randint(1, 3)
    print(f"random num: {random_section}")

    dim_row = 0

    tensor_list = split(
        tensor_full,
        split_size_or_sections=random_section,  # the length of each row after splitting
        dim=dim_row,
    )
    for tensor in tensor_list:
        print(f"tensor: \n{red(tensor)}")
        print(f"tensor length: {red(len(tensor))}")


@beautifier
def split_tensor_int_col():
    """ Split Tensor in equal parts by Column """
    tensor_full = rand(3, 7)
    print(f"tensor_full: \n{green(tensor_full)}")
    print(f"tensor_full shape: {green(tensor_full.shape)}")

    random_section = randint(1, 3)
    print(f"random num: {random_section}")

    dim_row = 1

    tensor_list = split(
        tensor_full,
        split_size_or_sections=random_section,
        dim=dim_row,
    )
    for tensor in tensor_list:
        print(f"tensor: \n{red(tensor)}")
        print(f"tensor length: {red(len(tensor))}")


@beautifier
def split_tensor_list_row():
    """ Split Tensor in Unequal Parts by Row """
    tensor_full = rand(5, 7)
    print(f"tensor_full: \n{green(tensor_full)}")
    print(f"tensor_full shape: {green(tensor_full.shape)}")

    random_section = [2, 1, 2]
    dim_row = 0

    tensor_list = split(
        tensor_full,
        split_size_or_sections=random_section,
        dim=dim_row,
    )
    for tensor in tensor_list:
        print(f"tensor: \n{red(tensor)}")
        print(f"tensor length: {red(len(tensor))}")


@beautifier
def split_tensor_list_col():
    """ Split Tensor in Unequal Parts by Column """
    tensor_full = rand(5, 7)
    print(f"tensor_full: \n{green(tensor_full)}")
    print(f"tensor_full shape: {green(tensor_full.shape)}")

    random_section = [1, 2, 3, 1, ]
    dim_row = 1

    tensor_list = split(
        tensor_full,
        split_size_or_sections=random_section,
        dim=dim_row,
    )
    for tensor in tensor_list:
        print(f"tensor: \n{red(tensor)}")
        print(f"tensor length: {red(len(tensor))}")


def main() -> None:
    """ Main Function """
    split_tensor_int_row()
    split_tensor_int_col()
    split_tensor_list_row()
    split_tensor_list_col()


if __name__ == "__main__":
    main()
