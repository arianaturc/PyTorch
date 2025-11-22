# tensor operations
# https://docs.pytorch.org/docs/stable/torch.html
import torch

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(f"{tensor}\n")


#concatenations of tensors
t1 = torch.cat([tensor, tensor, tensor], dim = 1)

print(f"{t1}\n")

t2 = torch.stack([tensor, tensor, tensor])
print(f"{t2}\n")

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(f"{y1}\n")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"{z1}\n")

#compute the sum of the values of a tensor
agg = tensor.sum()
agg_item = agg.item()
print(f"Sum: {agg_item}, Type: {type(agg_item)}\n")

#add 5 to all the values in the tensor
tensor.add_(5)
print(f"Add 5 to all values\n {tensor}")