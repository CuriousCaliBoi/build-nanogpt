# Create a 100x2 tensor with random values
import torch

tensor = torch.randn(100, 2)
print("Original tensor shape:", tensor.shape)
print("\nFirst 5 rows:")
print(tensor[:5])
# Tensor indexing and slicing in PyTorch follows similar rules to NumPy arrays
print("\nTensor Indexing and Slicing Guide:")
print("-----------------------------------")
print("Basic syntax: tensor[start:end:step, dim1, dim2, ...]")
print("- Use : to select all elements in a dimension")
print("- Negative indices count from the end")
print("- Step parameter controls the stride")

# Examples
print("\nMore examples:")

# Negative indexing
print("\nLast row:", tensor[-1])  # Last row
print("Last 5 rows:", tensor[-5:])  # Last 5 rows

# Striding
print("\nEvery 2nd row of first 10 rows:")
print(tensor[:10:2])  # Start:end:step

# Boolean indexing
mask = tensor[:, 0] > 0  # Get mask of positive values in first column
print("\nRows where first column is positive:")
print(tensor[mask])

print("\nBest Resource:")
print("PyTorch's official documentation on tensor indexing:")
print("https://pytorch.org/docs/stable/tensors.html#indexing-slicing-joining-mutating-ops")






# Get first column
first_col = tensor[:, 0]
print("\nFirst column shape:", first_col.shape)
print("First 5 elements of first column:")
print(first_col[:5])

# Get second column 
second_col = tensor[:, 1]
print("\nSecond column shape:", second_col.shape)
print("First 5 elements of second column:")
print(second_col[:5])

# Get rows 10-20
rows_slice = tensor[10:20]
print("\nRows 10-20 shape:", rows_slice.shape)
print("Rows 10-20:")
print(rows_slice)

# Get a specific cell
cell = tensor[5, 1]  # Row 5, column 1
print("\nValue at row 5, column 1:", cell.item())

# Get multiple rows and columns
subset = tensor[5:10, 1]  # Rows 5-9, column 1
print("\nSubset shape (rows 5-9, column 1):", subset.shape)
print("Subset values:")
print(subset)
