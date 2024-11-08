# Create a 100x2 tensor with random values
import torch

tensor = torch.randn(100, 2)
print("Original tensor shape:", tensor.shape)
print("\nFirst 5 rows:")
print(tensor[:5])






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
