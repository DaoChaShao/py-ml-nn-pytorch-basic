#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 17:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_06_ones.py
# @Desc     :   

from torch import tensor, ones, strided, ones_like

from utils.decorator import beautifier


@beautifier
def ones_creator():
    """ Create a tensor filled with the scalar value 1 """
    out = tensor([1])
    print(f"out_tensor: {out}")

    t = ones(
        size=(2, 3),
        out=out,  # If provided, the result will be stored in this tensor
        dtype=None,
        layout=strided,
        device=None,  # "cuda" or "cpu"
        requires_grad=False,
    )
    print(f"2 x 3 tensor:\n{t}")
    print(f"2 x 3 tensor shape: {t.shape}")

    print(f"out_tensor: \n{out}")
    print(f"out_tensor shape: {out.shape}")

    print(f"2 x 3 tensor id: {id(out)}")
    print(f"out_tensor id: {id(t)}")


@beautifier
def ones_like_creator():
    """ Create a tensor filled with the scalar value 1, with the same size as input """
    input_tensor = ones(size=(2, 3))
    t = ones_like(
        input=input_tensor,
        dtype=None,
        layout=strided,
        device=None,  # "cuda" or "cpu"
        requires_grad=False,
    )
    print(f"2 x 3 tensor:\n{t}")
    print(f"2 x 3 tensor shape: {t.shape}")

    print(f"input_tensor:\n{input_tensor}")
    print(f"input_tensor shape: {input_tensor.shape}")

    print(f"2 x 3 tensor id: {id(t)}")
    print(f"input_tensor id: {id(input_tensor)}")


def main() -> None:
    """ Main Function """
    ones_creator()
    ones_like_creator()


if __name__ == "__main__":
    main()
