#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 19:01
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   04_type_change.py
# @Desc     :   

from torch import Tensor, tensor, int64, float64, complex64

from utils.decorator import beautifier


@beautifier
def type_transformer(t: Tensor, category: str) -> Tensor:
    """ Transform the type of tensor """
    print(f"The original tensor is:\n{t}")
    print(f"The original tensor type is {t.dtype}")

    if category == "int":
        t_transformed = t.to(int64)
    elif category == "float":
        t_transformed = t.to(float64)
    else:
        raise ValueError("Category must be 'int' or 'float'")

    print(f"The transformed tensor is:\n{t_transformed}")
    print(f"The transformed tensor type is {t_transformed.dtype}")
    return t_transformed


@beautifier
def type_switcher(t: Tensor, category) -> Tensor:
    """ Switch the type of tensor """
    print(f"The original tensor is:\n{t}")
    print(f"The original tensor type is {t.dtype}")

    t_switched = t.to(category)

    print(f"The switched tensor is:\n{t_switched}")
    print(f"The switched tensor type is {t_switched.dtype}")
    return t_switched


@beautifier
def type_changer(t: Tensor, category) -> Tensor:
    """ Change the type of tensor """
    print(f"The original tensor is:\n{t}")
    print(f"The original tensor type is {t.dtype}")

    t_changed = t.type(category)

    print(f"The changed tensor is:\n{t_changed}")
    print(f"The changed tensor type is {t_changed.dtype}")
    return t_changed


@beautifier
def type_half(t: Tensor) -> Tensor:
    """ Change the type of tensor to half precision """
    print(f"The original tensor is:\n{t}")
    print(f"The original tensor type is {t.dtype}")

    t_half = t.half()

    print(f"The half precision tensor is:\n{t_half}")
    print(f"The half precision tensor type is {t_half.dtype}")
    return t_half


def main() -> None:
    """ Main Function """
    t1 = tensor([[1.5, 2.3, 3.7], [4.1, 5.9, 6.6]])
    type_transformer(t1, "int")

    t2 = tensor([[1.5, 2.3, 3.7], [4.1, 5.9, 6.6]])
    type_switcher(t2, complex64)

    t3 = tensor([[1, 2, 3], [4, 5, 6]])
    type_transformer(t3, "float")

    t4 = tensor([[1.5, 2.3, 3.7], [4.1, 5.9, 6.6]])
    type_changer(t4, int64)

    t5 = tensor([[1, 2, 3], [4, 5, 6]])
    type_half(t5)


if __name__ == "__main__":
    main()
