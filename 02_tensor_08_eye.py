#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_08_eye.py
# @Desc     :   

from torch import tensor, eye, strided

from utils.decorator import beautifier


@beautifier
def eye_creator():
    """ Create a 2-D tensor with ones on the diagonal and zeros elsewhere """
    out = tensor([1])
    print(f"out_tensor: \n{out}")

    t = eye(
        n=3,  # 矩阵行数
        m=3,  # 矩阵列数
        out=out,
        dtype=None,
        layout=strided,
        device=None,
        requires_grad=False,
    )
    print(f"tensor: \n{t}")
    print(f"tensor shape: {t.shape}")

    print(f"out_tensor: \n{out}")
    print(f"out_tensor shape: {out.shape}")


def main() -> None:
    """ Main Function """
    eye_creator()


if __name__ == "__main__":
    main()
