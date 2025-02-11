# pycharm shortcuts
# run line: shift+alt+e
# comment: ctrl + /

# Chap1: introduction
# # Numpy - for efficient operations on data structures like vectors,
# matrices and tensors

import numpy as np
from Tools.scripts.make_ctype import values

# a vector as a row
v_row = np.array([1,2,3])
# a vector as a column
v_col = np.array([[1],[2],[3]])
# matrix
matrix = np.array([[1,2], [1,2], [1,2]])

# create a sparse matrix
from  scipy import sparse
mat = np.array([[0,0], [0,1], [1,0]])
sparse_mat = sparse.csr_matrix(mat)
print(sparse_mat)
# Generating vectors of zeros and ones
v = np.zeros(shape = 5)
# a matrix of shape (3,3) with all ones
m = np.full(shape = (3,3), fill_value=1)
print(m)

#1.5 selecting elements

# select third element of a vector
v[2]
# select second row, second column
m[1,1]
# select all elements of a vector
v[:]
# select everything up to and
# including 2nd element
v[:2]
# select everything after 2nd element
v[2:]
# select the last element
v[-1]
# reverse the vector
v[::-1]
# select first 2 rows and all columns
m[:2, :]
# select all rows and the second col
m[:, 1:2]

#1.6 describing a matrix

# view number of rows and cols
m.shape
# view no. of elements, row*col
m.size
# view number of dimensions
m.ndim

# 1.7 Applying functions over each
# element - lambda(anonymous) functions

m = np.array([[1,2,3], [4,5,6], [7,8,9]])
# create a func tha adds 100 to item
add_100 = lambda i: i+100
# create a vectorized func
vectorized_add_100 = np.vectorize(add_100)

# apply function to elements of a matrix
vectorized_add_100(m)

# 1.8 finding max and min values
np.min(m)
np.max(m)

# maximum in each column
np.max(m, axis=0)
# maximum in each row
np.max(m, axis=1)

# 1.9 calculating summary statistics
# mean
np.mean(m)
# var
np.var(m)
# sd
np.std(m)

# 1.10 reshaping arrays
m.reshape(1,9)
# in reshape, -1 meaning "as many as needed"
m.reshape(1, -1) #1 row, as many cols

# 1.11 Transposing a vector matrix
m.T

# 1.12 Flattening a matrix into 1-dim array
m.flatten()

# ravel flattens list of arrays and speeds up code
ma =  np.array([[1,2], [3,4]])
mb = np.array([[4,5], [6,7]])
ml = [ma, mb]
np.ravel(ml)

# 1.13 finding the rank of a matrix
np.linalg.matrix_rank(m)
# 1.14 Getting diagonal of a matrix
m.diagonal()
## return diagonal one above the main diagonal
m.diagonal(offset=1)
# 1.15 Trace of a matrix
## trace is sum of diagonal elements
m.trace()
## or
sum(m.diagonal())
# 1.16 Calculating dot products
a  = np.array([1,2,3])
b = np.array([4,5,6])
np.dot(a, b)
# 1.18 Multiply matrices with np.dot
np.dot(ma, mb)
# 1.19 Inverting a matrix
np.linalg.inv(ma)
# 1.20 Generating random values
np.random.seed(1234)
np.random.random(3)

# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
np.random.normal(0.0, 1.0, 3)
# Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
np.random.logistic(0.0, 1.0, 3)
# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)
