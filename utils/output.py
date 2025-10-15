#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 11:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   output.py
# @Desc     :   


class Highlighter(object):
    """ Console Text Highlighter in the Colors of the Pycharm IDE """

    def __init__(self):
        pass

    @staticmethod
    def black(text):
        """ Highlight text with Black
        :param text: text to be highlighted
        :return: text is highlighted with Black
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;30m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;30m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;30m{text}\033[0m"

    @staticmethod
    def red(text):
        """ Highlight text with Red
        :param text: text to be highlighted
        :return: text is highlighted with Red
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;31m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;31m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;31m{text}\033[0m"

    @staticmethod
    def green(text):
        """ Highlight text with Green
        :param text: text to be highlighted
        :return: text is highlighted with Green
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;32m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;32m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;32m{text}\033[0m"

    @staticmethod
    def yellow(text):
        """ Highlight text with Yellow
        :param text: text to be highlighted
        :return: text is highlighted with Yellow
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;33m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;33m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;33m{text}\033[0m"

    @staticmethod
    def blue(text):
        """ Highlight text with Blue
        :param text: text to be highlighted
        :return: text is highlighted with Blue
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;34m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;34m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;34m{text}\033[0m"

    @staticmethod
    def purple(text):
        """ Highlight text with Purple
        :param text: text to be highlighted
        :return: text is highlighted with Purple
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;35m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;35m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;35m{text}\033[0m"

    @staticmethod
    def cyan(text):
        """ Highlight text with Cyan
        :param text: text to be highlighted
        :return: text is highlighted with Cyan
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;36m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;36m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;36m{text}\033[0m"

    @staticmethod
    def white(text):
        """ Highlight text with White
        :param text: text to be highlighted
        :return: text is highlighted with White
        """
        match text:
            case _ if isinstance(text, int):
                return f"\033[1;37m{text:12d}\033[0m"
            case _ if isinstance(text, float):
                return f"\033[1;37m{text:9.5f}\033[0m"
            case _:
                return f"\033[1;37m{text}\033[0m"

    @staticmethod
    def bold(text):
        """ Bold text
        :param text: text to be bolded
        :return: text is bolded
        """
        return f"\033[1m{text}\033[0m"

    @staticmethod
    def underline(text):
        """ Underline text
        :param text: text to be underlined
        :return: text is underlined
        """
        return f"\033[4m{text}\033[0m"

    @staticmethod
    def invert(text):
        """ Invert text color
        :param text: text to be inverted
        :return: text color is inverted
        """
        return f"\033[7m{text}\033[0m"

    @staticmethod
    def strikethrough(text):
        """ Strikethrough text
        :param text: text to be strikethrough
        :return: text is strikethrough
        """
        return f"\033[9m{text}\033[0m"
