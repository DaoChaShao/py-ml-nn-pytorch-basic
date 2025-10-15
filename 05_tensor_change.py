#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 19:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   05_tensor_change.py
# @Desc     :   

from torch import Tensor

from utils.decorator import beautifier


@beautifier
def tensor2array_same_id(t: Tensor):
    """ Tensor and numpy array share the same id """
    arr = t.numpy().copy()
    print(f"Tensor's id: {id(arr)}")
    print(f"Numpy array's id: {id(arr)}")


@beautifier
def tensor2array_diff_id(t: Tensor):
    """ Tensor and numpy array have different id """
    arr = t.numpy()
    print(f"Tensor's id: {id(t)}")
    print(f"Numpy array's id: {id(arr)}")

    arr = t.clone().numpy()
    print(f"Tensor's id: {id(t)}")
    print(f"Numpy array's id: {id(arr)}")


def main() -> None:
    """ Main Function """
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    tensor2array_same_id(t)
    tensor2array_diff_id(t)


if __name__ == "__main__":
    main()
