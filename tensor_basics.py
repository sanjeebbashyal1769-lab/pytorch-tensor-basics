"""
PyTorch Tensor Basics
Author : Sanjeeb Bashyal
Purpose: Learn the fundamentals of PyTorch tensors

Topics Covered:
1. Tensor Creation
2. Tensor Shapes
3. Tensor Dimensions
4. Tensor Data Types
5. Tensor Indexing
6. Tensor Slicing
7. Tensor Reshaping
"""

# Import the PyTorch library
import torch

# --------------------------------------------------------
# 1. Creating Tensors
# --------------------------------------------------------

print("-" * 60)
print("1. Creating Tensors")
print("-" * 60)

# Create a scalar (0D tensor)
scalar = torch.tensor(5)

# Create a vector (1D tensor)
vector = torch.tensor([10, 20, 30, 40])

# Create a matrix (2D tensor)
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("Scalar:")
print(scalar)

print("\nVector:")
print(vector)

print("\nMatrix:")
print(matrix)

# --------------------------------------------------------
# 2. Tensor Shape
# --------------------------------------------------------

print("\n" + "-" * 60)
print("2. Tensor Shape")
print("-" * 60)

print("Shape of scalar :", scalar.shape)
print("Shape of vector :", vector.shape)
print("Shape of matrix :", matrix.shape)

# --------------------------------------------------------
# 3. Tensor Dimensions
# --------------------------------------------------------

print("\n" + "-" * 60)
print("3. Tensor Dimensions")
print("-" * 60)

print("Scalar dimensions :", scalar.ndim)
print("Vector dimensions :", vector.ndim)
print("Matrix dimensions :", matrix.ndim)

# --------------------------------------------------------
# 4. Tensor Data Type
# --------------------------------------------------------

print("\n" + "-" * 60)
print("4. Tensor Data Type")
print("-" * 60)

print("Data type of matrix :", matrix.dtype)

# --------------------------------------------------------
# 5. Tensor Indexing
# --------------------------------------------------------

print("\n" + "-" * 60)
print("5. Tensor Indexing")
print("-" * 60)

print("First element :", vector[0])

print("Last element :", vector[-1])

print("Element at row 2, column 3 :", matrix[1, 2])

# --------------------------------------------------------
# 6. Tensor Slicing
# --------------------------------------------------------

print("\n" + "-" * 60)
print("6. Tensor Slicing")
print("-" * 60)

print("First row:")
print(matrix[0])

print("\nSecond column:")
print(matrix[:, 1])

print("\nFirst two columns:")
print(matrix[:, :2])

# --------------------------------------------------------
# 7. Tensor Reshaping
# --------------------------------------------------------

print("\n" + "-" * 60)
print("7. Tensor Reshaping")
print("-" * 60)

# Create a sequence of numbers from 0 to 11
numbers = torch.arange(12)

print("Original Tensor:")
print(numbers)

# Reshape into a 3 × 4 matrix
reshaped = numbers.reshape(3, 4)

print("\nReshaped Tensor (3 x 4):")
print(reshaped)

# Flatten back into one dimension
flattened = reshaped.reshape(-1)

print("\nFlattened Tensor:")
print(flattened)

# --------------------------------------------------------
# End of Program
# --------------------------------------------------------

print("\n" + "-" * 60)
print("Program Completed Successfully!")
print("-" * 60)
