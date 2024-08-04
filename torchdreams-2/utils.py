from __future__ import absolute_import, division, print_function

import numpy as np


def _make_arg_str(arg):
    """
    Convert an argument to a string for logging.
    args:
        arg: The argument to be converted.
    Returns:
        str: The converted argument.
    """
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg