# Author: Carson G Ray
# Language: Python


import numpy as np
from ax_tensor import AxTensor

# Creates axTensors for movie ratings
print("Movie Ratings:")
print()

customers = AxTensor(
    [[0, 0.5, 1],
    [1, 0.5, 0.5],
    [0, 1, 0],
    [1, 1, 1]], "customers", "values")

movies = AxTensor(
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

# Performes tensor operation
ratings = customers.by("customers", "values") @ movies.by("values", "movies")
ratings = AxTensor(ratings, "customers", "movies")

describe(ratings)

print()



# Creates axTensors for multiplication table
print("Multiplication Table:")
print()

x = AxTensor(np.arange(10), "x")
y = AxTensor(np.arange(10), "y")

describe(x)

print()

describe(y)

print()

# Multiplies in cartesian product form
table = AxTensor(y.by("y", [1]) * x.by([1], "x"), "y", "x")

describe(table)

print()

# Creates cubic table
table = AxTensor(table.by([1], "y", "x") * x.by("x", [2]), "z", "y", "x")

describe(table)

print()


    
