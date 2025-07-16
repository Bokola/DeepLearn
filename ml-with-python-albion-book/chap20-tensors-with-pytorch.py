import torch
import numpy as np

# 20.1 Creating a Tensor

# create a vector as a row
tensor_row = torch.tensor([1, 2, 3])

# create a vector as a column
tensor_column = torch.tensor([
    [1],
    [2],
    [3]
])

# 20.2 Creating a Tensor from NumPy

# create a NumPy array
vector_row = np.array([1, 2, 3])

# create a tensor from a NumPy array
tensor_row = torch.from_numpy(vector_row)

# 20.3 Creating a Sparse Tensor

# create a tensor
tensor = torch.tensor([
    [0, 0],
    [0, 1],
    [3, 0]
])

# create a sparse tensor from regular tensor
sparse_tensor = tensor.to_sparse()

# 20.4 Selecting Elements from a tensor
tensor_row[0]
tensor[2,0]

# reverse a tensor using flip
tensor_row.flip(dims=(-1,))

# 20.5 Describing a Tensor

# get the shape of a tensor
tensor_column.shape

# get the data type
tensor_column.dtype

# get layout
tensor_column.layout

# get the device being used by the tensor
tensor_column.device

# 20.6 Applying Operations to Elements

# broadcast an operation to all elements
tensor_row * 100

# 20.7 Finding the Maximum and Minimum values
tensor_row.max()
tensor_row.min()

# 20.8 Reshaping tensors
tensor_row.reshape(3, 1)

# 20.9 Transposing a Tensor

# create a 2-dim tensor
tensor = torch.tensor([[[1, 2, 3]]])

# transpose
tensor.mT

# 20.10 Flattening a Tensor

tensor_column.flatten()

# 20.11 Calculating Dot Products
tensor_1 = torch.tensor([1, 2, 3])
tensor_2 = torch.tensor([4, 5, 6])
tensor_1.dot(tensor_2)

# 20.12 Multiply tensors
tensor_1 * tensor_2