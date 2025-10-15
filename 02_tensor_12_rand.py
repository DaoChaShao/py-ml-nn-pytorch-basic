#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_12_rand.py
# @Desc     :   

from torch import ones, rand, float32, strided, tensor, rand_like

from utils.decorator import beautifier


@beautifier
def rand_creator():
    """ Create a tensor with random numbers from a uniform distribution over [0, 1) """
    out = ones([1])
    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")

    tensor = rand(
        size=(3, 5),
        out=out,
        dtype=float32,
        layout=strided,
        device=None,
        requires_grad=False,
    )
    print(f"tensor：\n{tensor}")
    print(f"tensor.shape: {tensor.shape}")

    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")


@beautifier
def rand_like_creator():
    """ Create a tensor with random numbers from a uniform distribution over [0, 1), with the same size as input """
    input_tensor = tensor([[1, 2, 3], [4, 5, 6]])
    print(f"input_tensor: \n{input_tensor}")
    print(f"input_tensor.shape: {input_tensor.shape}")

    t = rand_like(
        input=input_tensor,
        dtype=float32,
        layout=strided,
        device=None,
        requires_grad=False,
    )

    print(f"tensor: \n{t}")
    print(f"tensor.shape: {t.shape}")


def main() -> None:
    """ Main Function """
    rand_creator()
    rand_like_creator()


if __name__ == "__main__":
    main()
