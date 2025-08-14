"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# TODO: Implement for Task 0.1.
# - mul
def mul(x: float, y: float) -> float:
    return x * y

# - id
def id(x: float) -> float:
    return x

# - add
def add(x: float, y: float) -> float:
    return x + y

# - neg
def neg(x: float) -> float:
    return -x

# - lt
def lt(x: float, y: float) -> bool:
    return x < y

# - eq
def eq(x: float, y: float) -> bool:
    return x == y

# - max
def max(x: float, y: float) -> float:
    return x if x > y else y

# - is_close
def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2

# - sigmoid
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

# - relu
def relu(x: float) -> float:
    return max(0, x)
# - computes the derivative of relu times a second argument
def relu_back(x: float, y: float) -> float:
    return (1.0 if x > 0 else 0.0) * y

# - log
def log(x: float) -> float:
    return math.log(x)
# - exp
def exp(x: float) -> float:
    return math.exp(x)
# - log_back
def log_back(grad: float, x: float) -> float:
    if grad == 0:
        raise ValueError("Cannot compute log_back at zero")
    return x / grad
# - inv
def inv(x: float) -> float:
    return 1.0 / x
# - inv_back. Computes the derivate of a reciprocal times a second arg
def inv_back( grad: float,x: float) -> float:
    if grad == 0:
        raise ValueError("Cannot compute derivative at zero")
    return -x / (grad * grad)
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$





# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
def map(func: Callable[[float], float], lst: Iterable[float]) -> list[float]:
    return [func(x) for x in lst]

# - zipWith
def zipWith(func: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]) -> list[float]:
    return [func(x, y) for x, y in zip(lst1, lst2)]

# - reduce
def reduce(func: Callable[[float, float], float], lst: Iterable[float], initial: float) -> float:
    result = initial
    for x in lst:
        result = func(result, x)
    return result

# Use these to implement
# - negList : negate a list
def negList(lst: Iterable[float]) -> list[float]:
    return map(neg, lst)
# - addLists : add two lists together
def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> list[float]:
    return zipWith(add, lst1, lst2)
# - sum: sum lists
def sum(lst: Iterable[float]) -> float:
    return reduce(add, lst, 0.0)
# - prod: take the product of lists
def prod(lst: Iterable[float]) -> float:
    return reduce(mul, lst, 1.0)

# TODO: Implement for Task 0.3.
