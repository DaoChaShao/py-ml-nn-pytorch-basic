#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_14_randn.py
# @Desc     :   

from torch import tensor, ones, randint, float32, strided, randint_like

from utils.decorator import beautifier


@beautifier
def randint_creator():
    """ Create a tensor filled with random integers from low (inclusive) to high (exclusive) """
    out = ones([1])
    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")

    t = randint(
        low=0,
        high=10,
        size=(3, 5),
        out=out,
        dtype=float32,
        layout=strided,
        device=None,
        requires_grad=False,
    )
    print(f"tensor：\n{t}")
    print(f"tensor.shape: {t.shape}")

    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")


@beautifier
def randint_like_creator():
    """ 在 [0, 1) 区间内，生成整数均匀分布 """
    input_tensor = tensor([[1, 2, 3], [4, 5, 6]])
    print(f"input_tensor: \n{input_tensor}")
    print(f"input_tensor.shape: {input_tensor.shape}")

    t = randint_like(
        low=0,
        high=10,
        input=input_tensor,
        dtype=float32,
    )

    print(f"tensor: \n{t}")
    print(f"tensor.shape: {t.shape}")


def main() -> None:
    """ Main Function """
    randint_creator()
    randint_like_creator()


if __name__ == "__main__":
    main()
