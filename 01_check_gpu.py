#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/14 23:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_check_gpu.py
# @Desc     :   

from utils.PT import device_checker


def main() -> None:
    """ Main Function """
    device = device_checker()
    print(f"Using device: {device}")


if __name__ == "__main__":
    main()
