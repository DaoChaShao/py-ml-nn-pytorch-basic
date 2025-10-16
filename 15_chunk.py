#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 14:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   15_chunk.py
# @Desc     :   

from torch import chunk, rand

from utils.decorator import beautifier
from utils.highlighter import green, red, yellow


@beautifier
def chunk_tensor_row():
    """ Chunk Tensor by Row """
    input_tensor = rand(3, 7)
    print(f"input_tensor: \n{green(input_tensor)}")
    print(f"input_tensor shape: {green(input_tensor.shape)}")

    dim_row = 0

    tensor_list = chunk(
        input_tensor,
        chunks=3,
        dim=dim_row,
    )
    for tensor in tensor_list:
        print(f"tensor: \n{red(tensor)}")
        print(f"tensor shape: {red(tensor.shape)}")

    print(f"tensor_list: \n{yellow(tensor_list)}")
    print(f"tensor_list length: {yellow(len(tensor_list))}")


@beautifier
def chunk_tensor_col():
    """ Chunk Tensor by Column """
    input_tensor = rand(3, 7)
    print(f"input_tensor: \n{green(input_tensor)}")
    print(f"input_tensor shape: {green(input_tensor.shape)}")

    dim_col = 1

    tensor_list = chunk(
        input_tensor,
        chunks=7,
        dim=dim_col,
    )
    for tensor in tensor_list:
        print(f"tensor: \n{red(tensor)}")
        print(f"tensor shape: {red(tensor.shape)}")

    print(f"tensor_list: \n{yellow(tensor_list)}")
    print(f"tensor_list length: {yellow(len(tensor_list))}")


def main() -> None:
    """ Main Function """
    chunk_tensor_row()
    chunk_tensor_col()


if __name__ == "__main__":
    main()
