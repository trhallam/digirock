"""Common decorators for functions.
"""

from functools import wraps
import numpy as np

def mutually_exclusive(keyword, *keywords):
    """[summary]

    Args:
        keyword ([type]): [description]

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]

    Ref:
        https://stackoverflow.com/a/54487188
    """
    keywords = (keyword,) + keywords
    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if kwargs and sum([k in keywords for k, val in kwargs.items() if val is not None]) > 1:
                raise ValueError('Expected no more than one of {}'.format(', '.join(keywords)))
            return func(*args, **kwargs)
        return inner
    return wrapper

def mutually_inclusive(keyword, *keywords):
    """

    Args:
        keyword ([type]): [description]
    """
    keywords = (keyword,) + keywords
    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            i = 0
            for k, val in kwargs.items():
                if isinstance(val, np.ndarray):
                    val = True
                if k in keywords and val not in (None, False):
                    i = i + 1
            if not (i == 0 or i == len(keywords)):
                raise ValueError("Expected none or all of keywords"
                                 f"{', '.join(keywords)} to be defined.")
            return func(*args, **kwargs)
        return inner
    return wrapper

# pylint: disable=all
if __name__ == '__main__':
    # @mutually_exclusive('a', 'b')
    # def f(a=None, b=None):
    #     print(a, b)

    # f()
    # f(a=1)
    # f(b=1)
    # f(a=1, b=None)

    @mutually_inclusive('a', 'b')
    def f(a=None, b=None):
        print(a, b)

    f()
    f(a=None, b=False)
