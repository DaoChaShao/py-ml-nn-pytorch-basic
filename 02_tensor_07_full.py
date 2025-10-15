#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:04
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_07_full.py
# @Desc     :   

from torch import tensor, ones, float32, strided, full, full_like

from utils.decorator import beautifier


@beautifier
def full_creator():
    """ Create a tensor filled with a specified value """
    out = ones([1])
    print(f"out_tensor: {out}")

    t = full(
        size=(3, 5),
        fill_value=7,
        out=out,
        dtype=float32,
        layout=strided,
        device=None,
        requires_grad=False,
    )
    print(f"tensor：\n{t}")
    print(f"tensor.shape: {t.shape}")


@beautifier
def full_like_creator():
    """ Create a tensor filled with a specified value, with the same shape as a given tensor """
    input_tensor = tensor([[1, 2, 3], [4, 5, 6]])
    t = full_like(input_tensor, fill_value=7)

    print(f"input_tensor: \n{input_tensor}")

    print(f"tensor: \n{t}")
    print(f"tensor.shape: {t.shape}")


def main() -> None:
    """ Main Function """
    full_creator()
    full_like_creator()


if __name__ == "__main__":
    main()
