# Author: Carson G Ray
# Language: Python


import numpy as np
from flextensor import FlexTensor

print("Movie Ratings:")
print()

# Creates flextensor with customers on the 0 axis and their associated values on the 1 axis
customers = FlexTensor(
    [[0, 0.5, 1],
    [1, 0.5, 0.5],
    [0, 1, 0],
    [1, 1, 1]], "customers", "values")

# Creates flextensor with movies on the 0 axis and their associated values on the 1 axis 
movies = FlexTensor(
    [[0, 1, 0.5],
    [1, 0, 0],
    [0.5, 0.5, 0.5]], "movies", "values")

# Describes axTensors
def describe(tensor):
    print(tensor)
    print("Shape: {}".format(tensor.shape()))
    print("Size: {}".format(tensor.size()))


describe(customers)

print()

describe(movies)

print()

# Performes the dot product of the two matrices where you can intuitively see that the end result is movies vs. customers
ratings = customers["customers", "values"] @ movies["values", "movies"]

describe(ratings)

print()


print("Multiplication Table:")
print()

# Creates vectors that will become the x and y axes of the multiplication table
x = FlexTensor(np.arange(10), "x")
y = FlexTensor(np.arange(10), "y")

describe(x)

print()

describe(y)

print()

# Performes a cartesian product by adding placeholder axes at the right locations
table = y["y", ""] * x["", "x"]

describe(table)

print()

# Creates cubic table through the same process
table = FlexTensor(table["", "y", "x"] * x["x", "", ""], "z", "y", "x")

describe(table)

print()

# Demonstrates use of ellipsis
foo = FlexTensor(np.arange(16).reshape(2, 2, 2, 2), "a", "b", "c", "d")

describe(foo)

# Note: result is returned as flextensor instead of numpy array
foo = foo["d", ..., "", "a"]

describe(foo)

print()

# Demonstrates fancy indexing
foo = FlexTensor(np.arange(27).reshape(3, 3, 3), "z", "y", "x")

describe(foo)
print()

foo = foo[..., ("y", 0), ("x", slice(2))]

describe(foo)

# Demonstrates flattening multiple axes into each other from right to left in the list argument
foo = FlexTensor(np.arange(16).reshape(2,2,2,2), "a", "b", "c", "d")

describe(foo)
print()

foo = foo[("c", 0), ["b", "d", "a"]]

describe(foo)

